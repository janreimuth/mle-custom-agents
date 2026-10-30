# Building Custom Agents

This repository teaches how to build custom LLM agents as production-oriented
workflows rather than one-off prompts. The notebooks use
[PydanticAI](https://pydantic.dev/docs/ai/overview/) as the hands-on framework for typed
tools, memory, planning, orchestration, safety controls, and evaluation. They
also compare the implementation trade-offs against
LangGraph, Google Agent Development Kit, and CrewAI.

The lesson follows the earlier Prompt Engineering material by turning
ReAct-style ideas into a framework-backed agent loop.

## Learning Objectives

By the end of this repository, you should be able to:

- Build an LLM agent that performs multi-step reasoning with typed tool calls.
- Implement conversation memory so an agent can maintain context across turns.
- Design an agent workflow with planning, routing, bounded tool use, and human
  review points.
- Explain framework trade-offs between PydanticAI and alternatives such as
  LangGraph, Google ADK, and CrewAI.
- Evaluate an agent on trajectory quality, tool-use correctness, task
  completion, and failure recovery.

## Learning Path

| File / Folder | Description |
|---|---|
| [**01 - Agent Foundations and Tools**](01-agent-foundations-and-tools.ipynb) | Distinguish agents from prompts and fixed workflows, then build typed local tools. |
| [**02 - Memory and Conversation State**](02-memory-and-conversation-state.ipynb) | Reuse message history and filter supplied durable notes without changing storage. |
| [**03 - Planning and Orchestration**](03-planning-and-orchestration.ipynb) | Build a bounded plan-execute-verify workflow around the same operations assistant. |
| [**04 - Agent Evaluation and Framework Trade-offs**](04-agent-evaluation-and-framework-tradeoffs.ipynb) | Evaluate tool trajectories and compare PydanticAI with LangGraph, Google ADK, and CrewAI. |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**assets**](assets/) | Visual aids referenced in the repository. |
| [**data**](data/) | Local fixtures for tickets, policies, availability, memory, and evaluation. |
| [**support**](support/) | Shared tools and scripted or live PydanticAI setup. |
| [**tests**](tests/) | Regression checks for routing and human-review behavior. |
| [**.env.example**](.env.example) | Execution mode, provider credentials, and limits. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.13.5 or newer
and Python 3.13.17 or a newer 3.13 patch release, as defined in
[`.python-version`](.python-version).

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a **placeholder**. Replace it, including the `< >` brackets, with your own value. For example, `cd <repo-name>` becomes `cd mle-custom-agents`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

The command below installs all dependencies and creates a virtual environment in `.venv/`.

```bash
cd <repo-name>
uv sync
```

All four notebooks execute PydanticAI with typed tools, real message history, and enforced usage limits. The default `dry-run` mode supplies scripted replies without an API key or model download. Select a provider below to let a real LLM choose tools and produce the validated decision instead.

---

### 5. Set Up Environment Variables (Optional)

The notebooks run offline without `.env`. For real LLM calls, copy the template:

```bash
cp .env.example .env
```

Set `LLM_PROVIDER` and the matching API key; model defaults are in `.env.example`:

- **Groq (primary):** `LLM_PROVIDER=groq`, `GROQ_API_KEY`, optional `GROQ_MODEL`.
- **Mistral (alternative):** `LLM_PROVIDER=mistral`, `MISTRAL_API_KEY`, optional `MISTRAL_MODEL`.
- **OpenRouter (optional):** `LLM_PROVIDER=openrouter`, `OPENROUTER_API_KEY`, optional `OPENROUTER_MODEL`. `OPENROUTER_FALLBACK_MODEL=openrouter/free` enables a server-side model backup; leave it blank to disable. Free accounts currently share [50 requests/day across free models](https://openrouter.ai/pricing).
- **Google AI Studio (optional):** `LLM_PROVIDER=google`, `GOOGLE_API_KEY`, `GOOGLE_MODEL=gemma-4-26b-a4b-it` for direct [Gemma access](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api), independent of OpenRouter. Free [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) vary by project and model; use only non-sensitive exercise data on the free tier.
- **OpenAI (optional, paid):** `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, optional `OPENAI_MODEL`.

Restart the notebook kernel after changing `.env`. Exported variables take precedence. Only the selected provider is called; a missing key or provider error never falls back to simulation or another API provider.

> [!CAUTION]
> Never commit `.env`. It is git-ignored because environment files often hold secrets. Only `.env.example`, with placeholder or default values, belongs in the repository.

---

### 6. Open the Repository in VS Code

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open a notebook and select the Python environment created by `uv sync` as the kernel.

## Run Tests

```bash
uv run pytest tests
```

## Scripted And Live Runs

Both modes use the same `build_agent()` factory, tools, history, limits, and output schema. `LLM_PROVIDER=dry-run` keeps the deterministic `FunctionModel` for demonstrations and regression tests. Tests explicitly disable `.env` loading and live credentials. A native PydanticAI model can also be injected with `build_agent(model=...)`.

`MAX_AGENT_STEPS` bounds model requests per run; `MAX_TOOL_CALLS` separately bounds application-tool calls. Each request allows up to 2,048 output tokens and a 30-second timeout, with no HTTP retries. One tool/schema correction is allowed within the same request budget. A notebook contains several runs, so its total request count can exceed one run's budget.

Live tool order and wording vary; inspect traces and scores rather than expecting the scripted reference values. Schema validation and citation IDs do not prove factual correctness. The `human_review` tool marks review as pending, never approved, and no tool sends a customer update. These fixture tools assume authorized access; they are not production authorization controls. Provider-backed runs may incur costs or hit rate limits.

## Professional Context

PydanticAI is used here because typed tool signatures and
structured results make the agent mechanics inspectable. In production, the
same concepts often map to other frameworks:

- [LangGraph](https://langchain-ai.github.io/langgraph/) for graph-state
  orchestration and durable workflows.
- [Google ADK](https://adk.dev/) for vendor-native agent
  development on Google infrastructure.
- [CrewAI](https://docs.crewai.com/) for role-based crew patterns and
  multi-agent collaboration.

Choose a framework by operational constraints: state control, tool schema
discipline, deployment target, observability, and team familiarity.
