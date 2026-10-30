"""Typed operations tools shared by scripted and live PydanticAI workflows."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, AgentRunResult, ModelRetry
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel

from .provider import model_from_env, model_settings

DATA_ROOT = Path("data")


class AgentDecision(BaseModel):
    """Structured result returned by the operations assistant."""

    ticket_id: str = Field(min_length=1, pattern=r"\S")
    next_action: str = Field(min_length=1, pattern=r"\S")
    owner: str = Field(min_length=1, pattern=r"\S")
    customer_update: str = Field(min_length=1, pattern=r"\S")
    policy_ids: list[str] = Field(default_factory=list)
    requires_human_review: bool = False


class MemoryNote(BaseModel):
    """Small durable memory record with an explicit confidence score."""

    customer: str
    fact: str
    source: str
    confidence: float = Field(ge=0, le=1)


@dataclass(frozen=True)
class PlanStep:
    """One completed application-tool step in an agent trace."""

    tool: str
    reason: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentTrace:
    """Inspectable trace for evaluating an agent trajectory."""

    case_id: str
    request: str
    steps: list[PlanStep]
    decision: AgentDecision


class OperationsData:
    """Local data access layer that acts like a small tool backend."""

    def __init__(self, data_root: Path = DATA_ROOT) -> None:
        self.data_root = data_root
        self.tickets = self._read_json("operations/tickets.json")
        self.policies = self._read_json("operations/policies.json")
        self.availability = self._read_json("operations/availability.json")
        self.memory = self._read_json("memory/seed_memory.json")

    def _read_json(self, relative_path: str) -> list[dict[str, Any]]:
        path = self.data_root / relative_path
        return json.loads(path.read_text(encoding="utf-8"))

    def lookup_ticket(self, ticket_id: str) -> dict[str, Any]:
        """Read the ticket's current topic, priority and region."""

        for ticket in self.tickets:
            if ticket["ticket_id"] == ticket_id:
                return ticket
        raise ModelRetry(
            f"Unknown ticket_id: {ticket_id}. Use an ID from the request or history, not a guess."
        )

    def lookup_policy(self, topic: str) -> dict[str, Any]:
        """Read the policy matching the ticket topic."""

        for policy in self.policies:
            if policy["topic"] == topic:
                return policy
        raise ModelRetry(
            f"No policy for topic: {topic}. Use the ticket's returned topic."
        )

    def find_available_owner(self, role_hint: str, region: str) -> dict[str, Any]:
        """Prefer a matching role and region before comparing availability."""

        role_hint = role_hint.lower()
        role_matches = [
            row
            for row in self.availability
            if role_hint in row["role"] or row["role"] in role_hint
        ]
        regional_role_matches = [row for row in role_matches if row["region"] == region]
        candidates = regional_role_matches or role_matches
        if not candidates:
            candidates = [
                row for row in self.availability if row["region"] == region
            ] or self.availability
        return min(candidates, key=lambda row: row["available_in_minutes"])

    def recall_memory(self, customer: str) -> list[MemoryNote]:
        """Read stored notes; the agent tool filters which notes reach the model."""

        return [
            MemoryNote.model_validate(row)
            for row in self.memory
            if row["customer"].lower() == customer.lower()
        ]

    def human_review(self, ticket_id: str) -> dict[str, bool]:
        """Mark a review as pending; this tool never grants approval or sends updates."""

        return {
            "required": self.lookup_ticket(ticket_id)["priority"] == "critical",
            "approved": False,
        }


def load_operations_data() -> OperationsData:
    """Load the local tool backend from the repository data folder."""

    return OperationsData()


def extract_ticket_id(text: str) -> str:
    """Extract the first Northstar ticket identifier from a request."""

    match = re.search(r"\bT-\d+\b", text)
    if match:
        return match.group()
    raise ValueError("Request must contain a ticket id such as T-1001.")


def is_safe_memory(text: str) -> bool:
    """Illustrative lexical guard, not a complete sensitive-data detector."""

    return not any(
        term in text.lower() for term in ("password", "token", "api key", "credit card")
    )


def scenario_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Script model replies while PydanticAI executes real tools, history and limits."""

    prompts = [
        (i, part.content)
        for i, message in enumerate(messages)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]
    start, request = prompts[-1]
    returned: dict[str, Any] = {
        part.tool_name: part.content
        for message in messages[start:]
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    # Reuse a ticket mentioned in history when the follow-up omits its ID.
    ticket_prompt = next(
        (text for _, text in reversed(prompts) if re.search(r"\bT-\d+\b", text)), None
    )
    memory_request = "memory" in request.lower() and "for " in request
    if memory_request:
        if "recall_memory" not in returned:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "recall_memory",
                        {"customer": request.rsplit("for ", 1)[-1].rstrip(".")},
                    )
                ]
            )
        notes = [MemoryNote.model_validate(note) for note in returned["recall_memory"]]
        facts = " ".join(note.fact for note in notes)
        decision = AgentDecision(
            ticket_id="memory-only",
            next_action="Review approved memory.",
            owner="support",
            customer_update=f"{facts} Do not store credentials.",
        )
    elif ticket_prompt is not None:
        if "lookup_ticket" not in returned:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "lookup_ticket", {"ticket_id": extract_ticket_id(ticket_prompt)}
                    )
                ]
            )
        ticket = returned["lookup_ticket"]
        if "lookup_policy" not in returned:
            return ModelResponse(
                parts=[
                    ToolCallPart("lookup_policy", {"topic": ticket["topic"]}),
                    ToolCallPart(
                        "find_available_owner",
                        {"role_hint": ticket["topic"], "region": ticket["region"]},
                    ),
                    ToolCallPart("recall_memory", {"customer": ticket["customer"]}),
                ]
            )
        if ticket["priority"] == "critical" and "human_review" not in returned:
            return ModelResponse(
                parts=[ToolCallPart("human_review", {"ticket_id": ticket["ticket_id"]})]
            )
        policy, owner = (
            returned["lookup_policy"],
            returned["find_available_owner"]["owner"],
        )
        notes = [MemoryNote.model_validate(note) for note in returned["recall_memory"]]
        facts = " ".join(note.fact for note in notes)
        decision = AgentDecision(
            ticket_id=ticket["ticket_id"],
            next_action=f"Escalate {ticket['topic']} case to {owner}.",
            owner=owner,
            customer_update=f"Use policy {policy['policy_id']}: {policy['text']} {facts}".strip(),
            policy_ids=[policy["policy_id"]],
            requires_human_review=returned.get("human_review", {}).get(
                "required", False
            ),
        )
    else:
        raise ValueError(
            "Provide a ticket ID or ask about memory for a named customer."
        )
    return ModelResponse(
        parts=[ToolCallPart(info.output_tools[0].name, decision.model_dump())]
    )


def build_agent(
    data: OperationsData | None = None, *, model: Model | None = None
) -> Agent[None, AgentDecision]:
    """Use the configured provider, or scripted replies in explicit dry-run mode."""

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    model = model if model is not None else model_from_env()
    data = data or load_operations_data()
    agent = Agent[None, AgentDecision](
        model or FunctionModel(scenario_response),
        output_type=AgentDecision,
        instructions=(
            "For a memory-only request, call recall_memory for the named customer, "
            "do not look up a ticket, and return ticket_id=memory-only and owner=support. "
            "Otherwise look up the ticket named in the request or conversation history. "
            "If neither names a ticket, ask for its ID without guessing. "
            "Use tools to look up that ticket, its policy, an available owner "
            "matching its topic and region, and safe customer memory before proposing "
            "an action. Use only facts returned by tools, never invent policy IDs or "
            "claim an action was performed. Critical incidents must call human_review "
            "for that ticket and set requires_human_review=true; review remains "
            "pending, never approved."
        ),
        model_settings=model_settings(),
        retries=1,
    )
    agent.tool_plain(data.lookup_ticket)
    agent.tool_plain(data.lookup_policy)
    agent.tool_plain(data.find_available_owner)

    @agent.tool_plain
    def recall_memory(customer: str) -> list[MemoryNote]:
        """Expose only sufficiently confident, non-sensitive notes to the model."""

        return [
            note
            for note in data.recall_memory(customer)
            if note.confidence >= 0.85 and is_safe_memory(note.fact)
        ]

    agent.tool_plain(data.human_review)
    return agent


def trace_from_result(
    result: AgentRunResult[AgentDecision], request: str, case_id: str = "ad-hoc"
) -> AgentTrace:
    """Record completed framework tool calls rather than inventing a planned trace."""

    tools = {
        "lookup_ticket",
        "lookup_policy",
        "find_available_owner",
        "recall_memory",
        "human_review",
    }
    messages = result.new_messages()
    calls = {
        part.tool_call_id: part.args_as_dict()
        for message in messages
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name in tools
    }
    steps = [
        PlanStep(
            part.tool_name, "Tool returned successfully.", calls[part.tool_call_id]
        )
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
        and part.tool_name in tools
        and part.outcome == "success"
    ]
    return AgentTrace(
        case_id=case_id, request=request, steps=steps, decision=result.output
    )


def verify_trace(
    trace: AgentTrace, data: OperationsData, max_steps: int = 5
) -> dict[str, bool]:
    """Check a routing proposal against its requested ticket, not model claims."""

    ticket = data.lookup_ticket(extract_ticket_id(trace.request))
    review_required = ticket["priority"] == "critical"
    reviews = [step for step in trace.steps if step.tool == "human_review"]
    review_matches_ticket = bool(reviews) and all(
        step.arguments.get("ticket_id") == ticket["ticket_id"] for step in reviews
    )
    return {
        "within_step_budget": len(trace.steps) <= max_steps,
        "human_review_gate": (
            trace.decision.ticket_id == ticket["ticket_id"]
            and trace.decision.requires_human_review == review_required
            and (review_matches_ticket if review_required else not reviews)
        ),
    }


def evaluate_trace(trace: AgentTrace, expected_case: dict[str, Any]) -> dict[str, Any]:
    """Check expected tool names and lexical terms, not semantic correctness."""

    actual_tools = [step.tool for step in trace.steps]
    expected_tools = expected_case["expected_tools"]
    answer_text = (
        f"{trace.decision.next_action} "
        f"{trace.decision.customer_update} {trace.decision.owner}"
    ).lower()
    tool_recall = len(set(actual_tools) & set(expected_tools)) / len(expected_tools)
    required_terms_found = [
        term
        for term in expected_case["required_terms"]
        if re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", answer_text)
    ]
    term_recall = len(required_terms_found) / len(expected_case["required_terms"])
    review_match = (
        trace.decision.requires_human_review == expected_case["requires_human_review"]
    )
    request_match = trace.request == expected_case["request"]
    ticket = re.search(r"\bT-\d+\b", expected_case["request"])
    ticket_match = ticket is None or trace.decision.ticket_id == ticket.group()
    if expected_case["requires_human_review"] and ticket is not None:
        reviews = [step for step in trace.steps if step.tool == "human_review"]
        review_match = (
            review_match
            and bool(reviews)
            and all(
                step.arguments.get("ticket_id") == ticket.group() for step in reviews
            )
        )
    return {
        "case_id": expected_case["case_id"],
        "tool_recall": round(tool_recall, 2),
        "term_recall": round(term_recall, 2),
        "review_match": review_match,
        "request_match": request_match,
        "ticket_match": ticket_match,
        "passed": (
            tool_recall == 1
            and term_recall >= 2 / 3
            and review_match
            and request_match
            and ticket_match
        ),
    }
