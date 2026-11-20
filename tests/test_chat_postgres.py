"""The chat agent on its real checkpointer, AsyncPostgresSaver.

The other chat tests use InMemorySaver, which takes sync and async calls alike,
so they would still pass if the async Postgres path broke. This one pauses for
approval, then reads the pause back and resumes it through a newly opened
checkpointer, as after a worker restart.

Skips without Postgres, except in CI (conftest.test_database_url).
"""

import asyncio
import uuid

from chat_fakes import call, fake_github, make_agent
from langchain_core.messages import AIMessage
from langgraph.types import Command

from duo.agents import chat
from duo.config import settings


def run(coroutine):
    # psycopg's async mode needs a selector loop on Windows (apps/worker.py).
    return asyncio.run(coroutine, loop_factory=asyncio.SelectorEventLoop)


def test_pause_survives_a_restart(monkeypatch, test_database_url):
    monkeypatch.setattr(settings, "database_url", test_database_url)
    github = fake_github(monkeypatch)
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:8]}"}}
    label_7 = call("add_label", repo="o/r", number=7, label="bug")

    async def pause_restart_resume():
        before = await chat.open_checkpointer()
        try:
            agent, _ = make_agent(monkeypatch, label_7, checkpointer=before)
            turn = {"messages": [{"role": "user", "content": "Label #7 as bug"}]}
            await agent.ainvoke(turn, config)
        finally:
            await before.conn.close()

        after = await chat.open_checkpointer()
        try:
            agent, _ = make_agent(
                monkeypatch, AIMessage("Labeled #7."), checkpointer=after
            )
            [pause] = (await agent.aget_state(config)).interrupts
            approve = Command(resume={"decisions": [{"type": "approve"}]})
            result = await agent.ainvoke(approve, config)
        finally:
            await after.conn.close()
        return pause, result

    pause, result = run(pause_restart_resume())
    assert pause.value["action_requests"][0]["name"] == "add_label"
    assert github == [("add_label", "o/r", 7, "bug")]
    assert result["messages"][-1].content == "Labeled #7."
