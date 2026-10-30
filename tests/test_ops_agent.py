"""Regression tests for deterministic operations-agent behavior."""

import asyncio
import json
import unittest
from dataclasses import replace

from pydantic import ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import UsageLimits

from support import (
    AgentDecision,
    build_agent,
    evaluate_trace,
    load_operations_data,
    trace_from_result,
    verify_trace,
)


class OperationsAgentTests(unittest.TestCase):
    def test_unknown_ticket_gets_one_bounded_correction(self) -> None:
        attempts = 0

        def response(messages, info):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "lookup_ticket",
                            {"ticket_id": "12345" if attempts == 1 else "T-1001"},
                        )
                    ]
                )
            self.assertTrue(
                any(
                    isinstance(part, ToolReturnPart) and part.outcome == "retried"
                    for message in messages
                    for part in message.parts
                )
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        info.output_tools[0].name,
                        {
                            "ticket_id": "T-1001",
                            "next_action": "Review the ticket.",
                            "owner": "support",
                            "customer_update": "Review is pending.",
                        },
                    )
                ]
            )

        result = asyncio.run(
            build_agent(model=FunctionModel(response)).run(
                "Plan T-1001.",
                usage_limits=UsageLimits(request_limit=3, tool_calls_limit=2),
            )
        )
        self.assertEqual(result.output.ticket_id, "T-1001")
        self.assertEqual(attempts, 3)

    def test_repeated_unknown_ticket_is_not_a_success(self) -> None:
        def response(messages, info):
            return ModelResponse(
                parts=[ToolCallPart("lookup_ticket", {"ticket_id": "12345"})]
            )

        with self.assertRaises(UnexpectedModelBehavior):
            asyncio.run(
                build_agent(model=FunctionModel(response)).run(
                    "Plan T-1001.",
                    usage_limits=UsageLimits(request_limit=3, tool_calls_limit=2),
                )
            )

    def test_owner_selection_prioritizes_role_before_speed(self) -> None:
        data = load_operations_data()

        self.assertEqual(data.find_available_owner("billing", "US")["owner"], "Leo")
        self.assertEqual(data.find_available_owner("incident", "APAC")["owner"], "Sam")

    def test_critical_trace_routes_to_review(self) -> None:
        request = "Handle the critical incident in T-1003."
        result = asyncio.run(build_agent().run(request))
        trace = trace_from_result(result, request)

        self.assertEqual(trace.decision.owner, "Sam")
        self.assertTrue(trace.decision.requires_human_review)
        self.assertIn("human_review", [step.tool for step in trace.steps])
        self.assertEqual(len(trace.steps), result.usage.tool_calls)

    def test_critical_evaluation_requires_the_review_tool(self) -> None:
        data = load_operations_data()
        case = json.loads(
            (data.data_root / "evaluation/agent_cases.json").read_text(encoding="utf-8")
        )[1]
        result = asyncio.run(build_agent(data).run(case["request"]))
        trace = trace_from_result(result, case["request"])

        self.assertTrue(evaluate_trace(trace, case)["passed"])
        without_review = replace(
            trace, steps=[step for step in trace.steps if step.tool != "human_review"]
        )
        self.assertTrue(without_review.decision.requires_human_review)
        self.assertFalse(evaluate_trace(without_review, case)["passed"])

    def test_trace_verification_uses_trusted_ticket_priority(self) -> None:
        data = load_operations_data()
        request = "Handle the critical incident in T-1003."
        result = asyncio.run(build_agent(data).run(request))
        trace = trace_from_result(result, request)
        self.assertTrue(all(verify_trace(trace, data).values()))
        self.assertFalse(verify_trace(trace, data, max_steps=4)["within_step_budget"])

        without_review = replace(
            trace, steps=[step for step in trace.steps if step.tool != "human_review"]
        )
        for flag in (True, False):
            with self.subTest(review_flag=flag):
                changed = replace(
                    without_review,
                    decision=trace.decision.model_copy(
                        update={"requires_human_review": flag}
                    ),
                )
                self.assertFalse(verify_trace(changed, data)["human_review_gate"])

        wrong_ticket = replace(
            trace,
            decision=trace.decision.model_copy(update={"ticket_id": "T-1001"}),
        )
        self.assertFalse(verify_trace(wrong_ticket, data)["human_review_gate"])

    def test_routine_trace_does_not_require_review(self) -> None:
        data = load_operations_data()
        request = "Plan T-1001."
        result = asyncio.run(build_agent(data).run(request))
        trace = trace_from_result(result, request)
        self.assertTrue(all(verify_trace(trace, data).values()))

    def test_review_of_another_ticket_does_not_satisfy_the_gate(self) -> None:
        data = load_operations_data()
        case = json.loads((data.data_root / "evaluation/agent_cases.json").read_text())[
            1
        ]
        result = asyncio.run(build_agent(data).run(case["request"]))
        trace = trace_from_result(result, case["request"])
        changed = replace(
            trace,
            steps=[
                replace(step, arguments={"ticket_id": "T-1001"})
                if step.tool == "human_review"
                else step
                for step in trace.steps
            ],
        )
        self.assertFalse(verify_trace(changed, data)["human_review_gate"])
        self.assertFalse(evaluate_trace(changed, case)["passed"])

    def test_evaluation_rejects_wrong_requests_tickets_and_substrings(self) -> None:
        data = load_operations_data()
        case = json.loads((data.data_root / "evaluation/agent_cases.json").read_text())[
            1
        ]
        result = asyncio.run(build_agent(data).run(case["request"]))
        trace = trace_from_result(result, case["request"])
        self.assertFalse(
            evaluate_trace(replace(trace, request="Plan T-1001."), case)["passed"]
        )
        wrong_ticket = replace(
            trace, decision=trace.decision.model_copy(update={"ticket_id": "T-1001"})
        )
        self.assertFalse(evaluate_trace(wrong_ticket, case)["passed"])
        wrong_terms = replace(
            trace,
            decision=trace.decision.model_copy(
                update={
                    "next_action": "Review it.",
                    "customer_update": "incident commandment in 160 minutes",
                    "owner": "Samson",
                }
            ),
        )
        self.assertEqual(evaluate_trace(wrong_terms, case)["term_recall"], 0.0)

    def test_term_recall_threshold_accepts_two_of_three_terms(self) -> None:
        data = load_operations_data()
        case = json.loads((data.data_root / "evaluation/agent_cases.json").read_text())[
            1
        ]
        result = asyncio.run(build_agent(data).run(case["request"]))
        trace = trace_from_result(result, case["request"])

        for update, expected_recall, passed in [
            ("Incident command review is pending.", 0.67, True),
            ("Review is pending.", 0.33, False),
        ]:
            with self.subTest(customer_update=update):
                changed = replace(
                    trace,
                    decision=trace.decision.model_copy(
                        update={"customer_update": update}
                    ),
                )
                score = evaluate_trace(changed, case)
                self.assertEqual(score["term_recall"], expected_recall)
                self.assertEqual(score["tool_recall"], 1.0)
                self.assertTrue(score["review_match"])
                self.assertTrue(score["request_match"])
                self.assertTrue(score["ticket_match"])
                self.assertEqual(score["passed"], passed)

    def test_output_schema_rejects_blank_decision_fields(self) -> None:
        values = {
            "ticket_id": "T-1001",
            "next_action": "Review the shipment.",
            "owner": "Priya",
            "customer_update": "We are reviewing your shipment.",
        }
        for key in values:
            for blank in ("", "   "):
                with (
                    self.subTest(field=key, blank=blank),
                    self.assertRaises(ValidationError),
                ):
                    AgentDecision(**{**values, key: blank})

    def test_follow_up_reuses_history_but_refreshes_tool_results(self) -> None:
        async def conversation():
            agent = build_agent()
            first = await agent.run("Plan T-1001.")
            follow_up = await agent.run(
                "Who owns this ticket?", message_history=first.all_messages()
            )
            changed = await agent.run(
                "Now handle T-1003.", message_history=follow_up.all_messages()
            )
            memory = await agent.run(
                "Summarize what memory changes are safe for Aisha Khan.",
                message_history=changed.all_messages(),
            )
            return follow_up.output, changed.output, memory.output

        same_ticket, changed_ticket, memory = asyncio.run(conversation())
        self.assertEqual(same_ticket.ticket_id, "T-1001")
        self.assertEqual(changed_ticket.ticket_id, "T-1003")
        self.assertTrue(changed_ticket.requires_human_review)
        self.assertEqual(memory.ticket_id, "memory-only")

    def test_framework_enforces_request_and_tool_limits(self) -> None:
        for limits in [UsageLimits(request_limit=1), UsageLimits(tool_calls_limit=1)]:
            with self.subTest(limits=limits), self.assertRaises(UsageLimitExceeded):
                asyncio.run(build_agent().run("Plan T-1001.", usage_limits=limits))

    def test_memory_case_uses_a_real_framework_tool_call(self) -> None:
        case = {
            "case_id": "memory",
            "request": "Summarize what memory changes are safe for Aisha Khan.",
            "expected_tools": ["recall_memory"],
            "required_terms": ["production exports", "do not store credentials"],
            "requires_human_review": False,
        }
        request = "Summarize what memory changes are safe for Aisha Khan."
        result = asyncio.run(build_agent().run(request))
        trace = trace_from_result(result, request, case_id="memory")
        self.assertEqual([step.tool for step in trace.steps], ["recall_memory"])
        self.assertTrue(evaluate_trace(trace, case)["passed"])


if __name__ == "__main__":
    unittest.main()
