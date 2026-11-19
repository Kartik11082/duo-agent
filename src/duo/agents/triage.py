"""The agent's LangGraph workflow.

START -> triage -> propose --any action needs approval--> approve --yes--> execute -> END
                          |                                       `--no---> END
                          `--all actions "auto"----------------------------> execute -> END

- triage:  the LLM picks a label (the only LLM call).
- propose: turns the triage into a list of actions, plain data.
- policy:  config.PERMISSION_POLICY decides if a human must approve.
- execute: runs each action through connectors/github/tools.ACTIONS. With no repo in state
           (the sample issue below, /demo) it's a dry run that changes nothing.

The checkpointer saves state to duo.db after every node, so a paused run
survives the process exiting and can be resumed later.

Run (dry run on the sample issue):
    uv run python -m duo.agents.triage start          # triage, pause, exit
    uv run python -m duo.agents.triage resume yes     # or: resume no
"""

import sqlite3
import sys
from typing import Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from duo.config import PERMISSION_POLICY, settings
from duo.connectors.github.tools import ACTIONS
from duo.llm import get_llm

# The labels the agent may choose from: GitHub's default labels, so they
# already exist in new repos. A name that doesn't exist would be created.
Label = Literal["bug", "enhancement", "question", "documentation"]


class Triage(BaseModel):
    """What the LLM must return for an issue. Pydantic validates it."""

    label: Label = Field(description="The single best label for the issue.")
    reasoning: str = Field(description="One or two sentences explaining the label.")


class IssueState(TypedDict):
    """The data that flows through the graph. Nodes read it and return updates.

    Only plain types (str, int, bool, lists/dicts of them) so the checkpointer
    can save it without issue.
    """

    repo: str  # "owner/name"; empty means dry run
    number: int
    title: str
    body: str
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
        if PERMISSION_POLICY.get(action["name"]) != "auto":
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


def build_graph():
    builder = StateGraph(IssueState)
    builder.add_node("triage", triage_issue)
    builder.add_node("propose", propose_actions)
    builder.add_node("approve", ask_approval)
    builder.add_node("execute", execute_actions)

    builder.add_edge(START, "triage")
    builder.add_edge("triage", "propose")
    builder.add_conditional_edges("propose", route_by_policy, ["approve", "execute"])
    builder.add_conditional_edges("approve", route_after_approval, ["execute", END])
    builder.add_edge("execute", END)

    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    return builder.compile(checkpointer=SqliteSaver(conn))


# No repo, so running it is always a dry run.
SAMPLE_ISSUE = {
    "repo": "",
    "number": 0,
    "title": "App crashes when uploading a file larger than 10MB",
    "body": "Steps: open the upload page, pick a 12MB PNG, click Upload. "
    "The page goes blank and the console shows 'RangeError: Maximum call stack'.",
}

# Identifies the CLI's run in the checkpointer. Real issues get "owner/repo#N".
CONFIG = {"configurable": {"thread_id": "issue-1"}}


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
