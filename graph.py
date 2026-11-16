"""The agent's LangGraph workflow.

Milestone 3: triage, then pause for human approval, then (pretend to) act.
    START -> triage -> approve --yes--> apply_label -> END
                              `--no---> END
The checkpointer saves state to duo.db after every node, so a paused run
survives the process exiting and can be resumed later.

Run:
    uv run python graph.py start          # triage, pause, exit
    uv run python graph.py resume yes     # or: resume no
"""

import sqlite3
import sys
from typing import Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from config import settings
from llm import get_llm

# The labels the agent may choose from: GitHub's default labels, so they
# already exist in new repos. A name that doesn't exist would be created.
Label = Literal["bug", "enhancement", "question", "documentation"]


class Triage(BaseModel):
    """What the LLM must return for an issue. Pydantic validates it."""

    label: Label = Field(description="The single best label for the issue.")
    reasoning: str = Field(description="One or two sentences explaining the label.")


class IssueState(TypedDict):
    """The data that flows through the graph. Nodes read it and return updates.

    Only plain types (str, bool) so the checkpointer can save it without issue.
    """

    title: str
    body: str
    label: str  # set by triage
    reasoning: str  # set by triage
    approved: bool  # set by approve


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


def ask_approval(state: IssueState) -> dict:
    """Pause until a human answers "yes" or "no".

    On resume, LangGraph runs this whole function again from the top, and this
    time interrupt() returns the human's answer. So no side effects here.
    """
    answer = interrupt({"label": state["label"], "reasoning": state["reasoning"]})
    return {"approved": answer == "yes"}


def route_after_approval(state: IssueState) -> str:
    return "apply_label" if state["approved"] else END


def apply_label(state: IssueState) -> dict:
    """Placeholder for the real GitHub call in milestone 6."""
    print(f"Would apply label '{state['label']}' on GitHub.")
    return {}


def build_graph():
    builder = StateGraph(IssueState)
    builder.add_node("triage", triage_issue)
    builder.add_node("approve", ask_approval)
    builder.add_node("apply_label", apply_label)
    builder.add_edge(START, "triage")
    builder.add_edge("triage", "approve")
    builder.add_conditional_edges("approve", route_after_approval, ["apply_label", END])
    builder.add_edge("apply_label", END)

    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    return builder.compile(checkpointer=SqliteSaver(conn))


SAMPLE_ISSUE = {
    "title": "App crashes when uploading a file larger than 10MB",
    "body": "Steps: open the upload page, pick a 12MB PNG, click Upload. "
    "The page goes blank and the console shows 'RangeError: Maximum call stack'.",
}

# Identifies this run in the checkpointer. Later: one thread per GitHub issue.
CONFIG = {"configurable": {"thread_id": "issue-1"}}


if __name__ == "__main__":
    graph = build_graph()
    args = sys.argv[1:]

    if args == ["start"]:
        result = graph.invoke(SAMPLE_ISSUE, CONFIG)
    elif len(args) == 2 and args[0] == "resume" and args[1] in ("yes", "no"):
        result = graph.invoke(Command(resume=args[1]), CONFIG)
    else:
        print("usage: graph.py start | graph.py resume yes|no")
        sys.exit(1)

    if "__interrupt__" in result:
        proposal = result["__interrupt__"][0].value
        print(f"Paused. Proposed label: {proposal['label']}")
        print(f"Reason: {proposal['reasoning']}")
        print("Approve with: uv run python graph.py resume yes   (or no)")
    else:
        print("Finished:", "approved" if result["approved"] else "rejected")
