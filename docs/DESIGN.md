# duo: design

duo is a single-tenant, self-hostable agent that runs all the time. You talk to it
on Telegram, and it reacts to events from connected services and to schedules.
It acts through connectors, and every write goes through a permission policy that
can ask you for approval on Telegram. It runs locally with Ollama and deploys to
AWS with Terraform.

Agreed 2026-11-19. Change this file when a decision changes.

## Entry points

| Entry point | Source | Handler |
|---|---|---|
| Chat | Telegram messages | Chat agent |
| Events | Connector triggers, e.g. GitHub `issues.opened` | A workflow or the chat agent, per `routes:` |
| Schedules | Cron entries in `agent.yaml` | The chat agent, with the entry's prompt |

## Agents

- **Chat agent:** LangChain `create_agent` with `HumanInTheLoopMiddleware`. The
  LLM chooses which tools to call. Tools whose policy isn't `auto` pause for
  approval.
- **Workflows:** custom LangGraph graphs for fixed processes. The first is
  `triage`: triage → propose actions → policy → approve → execute.
- All runs are checkpointed, so a run waiting for approval survives restarts.

## Configuration: `agent.yaml`

The non-secret settings, validated with Pydantic at startup: an unknown key, an
unknown connector or tool, or a policy other than `auto`/`approve` stops duo with
the key named. A dashboard will edit the same settings later. `num_ctx` and
`temperature` are optional; without `num_ctx` the provider's default is used.
A connector is on only if it is listed under `connectors:` (`enabled` defaults to
true).

```yaml
model: {provider: ollama, name: "qwen2.5:7b", num_ctx: 8192}
connectors:
  github: {enabled: true, policy: {post_comment: approve}}
  fetch: {enabled: true}
routes:
  - on: github.issues.opened
    run: workflow:triage
  - on: github.issues.opened
    run: agent
    prompt: "New issue {repo}#{number}: {title}. Check for duplicates and suggest a label."
schedules:
  - cron: "0 8 * * *"
    prompt: "Morning brief: today's calendar, new issues on my repos, open reminders."
```

## Connectors

Each connector has a manifest, `connectors/<id>/manifest.yaml`, that describes it,
so the host (and later a dashboard) can set it up without knowing its code.
`connectors/registry.py` lists the manifests and, for native connectors, the Python
functions behind their tools and triggers, and checks they match at startup.
`credentials` are the names `get_secret()` takes.

```yaml
id: github
kind: native            # native (Python in this repo) | mcp (an MCP server)
auth: github_app        # github_app | api_key | oauth2 | none
credentials: [github_app_id, github_private_key_path, github_webhook_secret]
tools: {list_issues: auto, get_issue: auto, add_label: approve, post_comment: approve}
triggers: [issues.opened]
```

An MCP manifest adds `server:`, how to reach the server: a command the worker
starts (`transport: stdio`), or a hosted URL (`transport: streamable_http`, with
an optional `bearer_token` credential sent as a header). Only the tools listed
under `tools:` are given to the agent.

```yaml
id: web_search
kind: mcp
auth: api_key
credentials: [tavily_api_key]
server: {transport: streamable_http, url: "https://mcp.tavily.com/mcp/", bearer_token: tavily_api_key}
tools: {tavily_search: auto}
```

Order: GitHub (native), then MCP `fetch` and web search (Tavily; read-only), then
Google Calendar and Gmail via MCP.

## Permission policy

One `policy.py`, used by workflows and by the chat agent's middleware. A tool runs
on its own only if its policy is exactly `auto`. Anything else, including a missing
entry or a typo, needs approval. Actions are written to be idempotent, so retrying
them is safe.

## Memory

- **Conversation:** one thread per Telegram chat, `/new` starts a fresh one, and
  old messages are trimmed to fit the context window.
- **Long-term:** LangGraph `PostgresStore` with pgvector. Namespaces:
  `("user", "profile")` for facts and preferences, `("github", repo, "issues")`
  for duplicate detection. Written through explicit `save_memory` and
  `search_memory` tools; automatic extraction comes later.

## Models

| | Local | AWS |
|---|---|---|
| Chat | Ollama `qwen2.5:7b` | Bedrock |
| Embeddings | Ollama `nomic-embed-text` (768-d) | Bedrock Titan Text Embeddings v2 |

The embedding model name is stored with the data. `duo reindex` re-embeds after a
model change.

## Storage and queue

- Postgres with pgvector everywhere: Docker Compose locally, RDS on AWS. It holds
  the checkpoints, the memory store and the local job queue.
- `JobQueue` interface: Postgres with `SELECT ... FOR UPDATE SKIP LOCKED` locally,
  SQS on AWS.

## Secrets

`get_secret(name)` reads `.env` locally and SSM Parameter Store on AWS. Names are
the lowercase `.env` names, e.g. `github_webhook_secret`. Google
OAuth uses a one-time `duo connect google` command until the dashboard exists.

## AWS

| Piece | Service |
|---|---|
| Webhook intake | API Gateway HTTP API + Lambda (the FastAPI app wrapped with Mangum) |
| Queue | SQS, with a dead-letter queue |
| Worker and Telegram | ECS Fargate, one task, public subnet, no NAT Gateway; Telegram uses long polling |
| Schedules | EventBridge Scheduler → SQS |
| Database | RDS Postgres `db.t4g.micro` with pgvector |
| LLM | Bedrock |
| Secrets | SSM Parameter Store (SecureString) |
| Images and logs | ECR, CloudWatch Logs |

All of it is in Terraform (`infra/terraform/`). Deploy with `terraform apply`, and
run `terraform destroy` between demos to keep costs inside the $100 credit.

## Tooling

- CI (GitHub Actions) on every push: ruff, pyright, pytest.
- Deploy workflow, started by hand: build the image, push to ECR, `terraform apply`.
  Authenticates to AWS with OIDC, so no long-lived keys.
- Optional LangSmith tracing, on when `LANGSMITH_API_KEY` is set in `.env`. Runs
  go to the project `tracing: {project: duo}` in `agent.yaml`, or, when
  `LANGSMITH_AGENT_ID` and `LANGSMITH_AGENT_ENVIRONMENT` are set in `.env`, to
  that LangSmith agent instead (LangSmith refuses both at once).

## Layout

```
agent.yaml
pyproject.toml            # entry points: duo-server, duo-worker
src/duo/
  config.py  llm.py  policy.py  storage.py
  agents/      triage.py  chat.py
  connectors/  github/{auth,tools,webhook}.py  mcp.py
  channels/    telegram.py
  apps/        server.py  worker.py
tests/
infra/terraform/
docs/
```

## Build order

Each phase is small, works on its own, and is committed and pushed before the next.

| Phase | Scope | Done when |
|---|---|---|
| P0 | Move into `src/duo/`, no behavior change | The live GitHub flow still works |
| P1 | Postgres + pgvector in Docker Compose; checkpointer; jobs with SKIP LOCKED | Same flow on Postgres |
| P2 | CI: ruff, pyright, pytest | Checks pass on push |
| P3 | `agent.yaml`, manifests, `policy.py`, `get_secret` | GitHub runs from its manifest |
| P4 | Chat agent + HITL + GitHub tools + LangSmith | "Label #2 as question" → approve → done |
| P5 | MCP: `fetch` + Tavily search | "Summarize this URL" works |
| P6 | Memory store + duplicate detection | #3 flagged as a duplicate of #2 |
| P7 | Routes + schedules | The 8 AM brief arrives on its own |
| P8 | Terraform on AWS + Bedrock, compare models | `apply` → demo → `destroy` |
| P9 | Google Calendar + Gmail via MCP, `duo connect google` | "What's on my calendar?" works |
| P10 | Evals, README, architecture diagram, demo video | Ready to show |

Later: dashboard, PR summaries, a reconciliation sweep for missed webhooks, a
Retry button for failed actions.

## Not in scope

- Multiple users or accounts.
- Storing credentials in the database.
- Self-hosted vector databases (Chroma, Milvus).
