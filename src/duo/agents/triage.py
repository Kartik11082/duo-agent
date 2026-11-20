"""The agent's LangGraph workflow.

START -> triage -> propose --any action needs approval--> approve --yes--> execute -> END
                          |                                       `--no---> END
                          `--all actions "auto"----------------------------> execute -> END

- triage:  the LLM picks a label (the only LLM call).
- propose: turns the triage into a list of actions, plain data.
- policy:  policy.py decides if a human must approve (GitHub manifest + agent.yaml).
- execute: runs each action through connectors/github/tools.ACTIONS. With no repo in state
           (the sample issue below, /demo) it's a dry run that changes nothing.

The checkpointer saves state to Postgres after every node, so a paused run
survives the process exiting and can be resumed later.

Run (dry run on the sample issue):
    uv run python -m duo.agents.triage start          # triage, pause, exit
    uv run python -m duo.agents.triage resume yes     # or: resume no
"""

import atexit
import sys
from typing import Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

from duo.config import settings
from duo.connectors.github.tools import ACTIONS
from duo.llm import get_llm
from duo.policy import policy_for

# The labels the agent may choose from: GitHub's default labels, so they
# already exist in new repos. A name that doesn't exist would be created.
Label = Literal["bug", "enhancement", "question", "documentation"]


class Triage(BaseModel):
    """What the LLM must return for an issue. Pydantic validates it."""

    label: Label = Field(description="The single best label for the issue.")
    reasoning: str = Field(description="One or two sentences explaining the label.")


class IssueInput(TypedDict):
    """What a run starts with: the issue. The nodes fill in the rest of IssueState."""

    repo: str  # "owner/name"; empty means dry run
    number: int
    title: str
    body: str


class IssueState(IssueInput):
    """The data that flows through the graph. Nodes read it and return updates.

    Only plain types (str, int, bool, lists/dicts of them) so the checkpointer
    can save it without issue.
    """

    label: str  # set by triage
    reasoning: str  # set by triage
    actions: list[dict]  # set by propose: [{"name": ..., "args": {...}}]
    approved: bool  # set by approve
    results: list[str]  # set by execute: one summary line per action


TRIAGE_PROMPT = """You triage GitHub issues for a software project.

Pick exactly one label:
- bug: something is broken or behaves differently than documented.
- enhancement: a request for new behavior or an improvement.
- question: the author asks how to do something; nothing is broken.
- documentation: documentation is missing, wrong, or unclear.

Issue title: {title}

Issue body:
{body}
"""


def triage_issue(state: IssueState) -> dict:
    """Ask the LLM to classify the issue. Returns only the keys it changes."""
    llm = get_llm().with_structured_output(Triage)
    prompt = TRIAGE_PROMPT.format(title=state["title"], body=state["body"])
    triage = llm.invoke(prompt)
    # Typed as dict | BaseModel; with a Pydantic schema it is always a Triage.
    assert isinstance(triage, Triage)
    return {"label": triage.label, "reasoning": triage.reasoning}


def propose_actions(state: IssueState) -> dict:
    """Turn the triage into actions: data that can be shown, checked, and run."""
    comment = (
        f"🤖 **duo-agent triage**: labeled `{state['label']}`.\n\n"
        f"> {state['reasoning']}"
    )
    return {
        "actions": [
            {"name": "add_label", "args": {"label": state["label"]}},
            {"name": "post_comment", "args": {"text": comment}},
        ],
        "results": [],  # clear results left by an earlier run on this thread
    }


def route_by_policy(state: IssueState) -> str:
    """Pause for approval unless every action is explicitly "auto"."""
    for action in state["actions"]:
        if policy_for("github", action["name"]) != "auto":
            return "approve"
    return "execute"


def ask_approval(state: IssueState) -> dict:
    """Pause until a human answers "yes" or "no".

    On resume, LangGraph runs this whole function again from the top, and this
    time interrupt() returns the human's answer. So no side effects here.
    """
    answer = interrupt({"actions": state["actions"], "reasoning": state["reasoning"]})
    return {"approved": answer == "yes"}


def route_after_approval(state: IssueState) -> str:
    return "execute" if state["approved"] else END


def execute_actions(state: IssueState) -> dict:
    """Run each action on GitHub, or only describe it when there's no repo."""
    if not state.get("repo"):
        return {
            "results": [f"(dry run) {a['name']} {a['args']}" for a in state["actions"]]
        }

    results = []
    for action in state["actions"]:
        run = ACTIONS[action["name"]]
        results.append(run(state["repo"], state["number"], **action["args"]))
    return {"results": results}


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Build the graph, saving state to Postgres unless given another checkpointer.

    Tests pass InMemorySaver() so they don't need a database.
    """
    builder = StateGraph(IssueState, input_schema=IssueInput)
    builder.add_node("triage", triage_issue)
    builder.add_node("propose", propose_actions)
    builder.add_node("approve", ask_approval)
    builder.add_node("execute", execute_actions)

    builder.add_edge(START, "triage")
    builder.add_edge("triage", "propose")
    builder.add_conditional_edges("propose", route_by_policy, ["approve", "execute"])
    builder.add_conditional_edges("approve", route_after_approval, ["execute", END])
    builder.add_edge("execute", END)

    if checkpointer is None:
        # A pool, because Telegram handlers run the graph from several threads.
        # PostgresSaver needs these three connection settings.
        pool: ConnectionPool[Connection[DictRow]] = ConnectionPool(
            settings.database_url,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
            open=True,
        )
        # Stop the pool's background threads before Python exits; otherwise exit
        # stalls about 5 seconds per thread and prints warnings.
        atexit.register(pool.close)
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()  # creates or migrates its tables; does nothing if up to date
    return builder.compile(checkpointer=checkpointer)


# No repo, so running it is always a dry run.
SAMPLE_ISSUE: IssueInput = {
    "repo": "",
    "number": 0,
    "title": "App crashes when uploading a file larger than 10MB",
    "body": "Steps: open the upload page, pick a 12MB PNG, click Upload. "
    "The page goes blank and the console shows 'RangeError: Maximum call stack'.",
}

# Identifies the CLI's run in the checkpointer. Real issues get "owner/repo#N".
CONFIG: RunnableConfig = {"configurable": {"thread_id": "issue-1"}}


if __name__ == "__main__":
    graph = build_graph()
    args = sys.argv[1:]

    if args == ["start"]:
        result = graph.invoke(SAMPLE_ISSUE, CONFIG)
    elif len(args) == 2 and args[0] == "resume" and args[1] in ("yes", "no"):
        result = graph.invoke(Command(resume=args[1]), CONFIG)
    else:
        print("usage: python -m duo.agents.triage start | resume yes|no")
        sys.exit(1)

    if "__interrupt__" in result:
        print("Paused. Proposed actions:")
        for action in result["__interrupt__"][0].value["actions"]:
            print(f"  - {action['name']}: {action['args']}")
        print("Approve with: uv run python -m duo.agents.triage resume yes   (or no)")
    elif result["results"]:
        print("Executed:")
        for line in result["results"]:
            print(f"  - {line}")
    else:
        print("Rejected, nothing done.")
