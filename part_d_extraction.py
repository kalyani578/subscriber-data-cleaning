"""Extract and validate structured fields from every support ticket."""

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import mock_llm
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field, ValidationError

from llm_client import call_llm


ROOT = Path(__file__).resolve().parent
TICKETS_PATH = ROOT / "data" / "raw" / "tickets.csv"
OUTPUT_PATH = ROOT / "outputs" / "ticket_extractions.jsonl"


class TicketRecord(BaseModel):
    """Validated fields extracted from one support ticket."""

    subscriber_id: str | None = None
    category: Literal["billing", "signal", "plan_change", "cancellation", "other"]
    urgency: int = Field(ge=1, le=5)
    wants_refund: bool


def call_prompt(prompt_value: Any) -> str:
    """Translate LangChain messages to the shared chat client format."""
    role_names = {"system": "system", "human": "user", "ai": "assistant"}
    messages = [
        {"role": role_names[message.type], "content": message.content}
        for message in prompt_value.to_messages()
    ]
    return call_llm(messages)


def build_chain() -> tuple[Any, PydanticOutputParser]:
    """Build the required system/history/client/Pydantic parser chain."""
    parser = PydanticOutputParser(pydantic_object=TicketRecord)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "TICKET EXTRACTOR\nExtract the ticket fields. Follow this schema:\n"
                "{format_instructions}",
            ),
            MessagesPlaceholder("history"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | call_prompt | parser, parser


def run_one_ticket(
    chain: Any, ticket_text: str, first_failures: Counter[str]
) -> tuple[str, int, TicketRecord | None]:
    """Return status, total model attempts, and a validated record if possible."""
    history = [HumanMessage(content=ticket_text)]
    attempts = 1
    try:
        return "valid", attempts, chain.invoke({"history": history})
    except OutputParserException as error:
        last_error = error
        if last_error.llm_output is None:
            raise RuntimeError("Parser failure did not include the model reply") from last_error
        count_first_failure(last_error.llm_output, first_failures)

    for _ in range(2):
        if last_error.llm_output is None:
            raise RuntimeError("Parser failure did not include the model reply") from last_error
        history.extend(
            [
                AIMessage(content=last_error.llm_output),
                HumanMessage(content=f"VALIDATION ERROR: {last_error}"),
            ]
        )
        attempts += 1
        try:
            return "valid", attempts, chain.invoke({"history": history})
        except OutputParserException as error:
            last_error = error

    return "manual_review", attempts, None


def count_first_failure(
    raw_reply: str, counts: Counter[str]
) -> None:
    """Classify a failed first response as invalid JSON or a schema violation."""
    try:
        data = json.loads(raw_reply)
    except json.JSONDecodeError:
        if raw_reply.startswith("Sure!"):
            counts["chatty_prose"] += 1
        else:
            counts["unreadable_ticket"] += 1
        return
    if not isinstance(data, dict):
        counts["schema_payload"] += 1
        return

    try:
        TicketRecord.model_validate(data)
    except ValidationError as error:
        for item in error.errors():
            counts[f"schema_{item['loc'][0]}"] += 1


def demonstrate_pydantic() -> None:
    """Print the requested coercion and validation-error examples."""
    converted = TicketRecord(
        subscriber_id=None,
        category="billing",
        urgency="3",
        wants_refund=False,
    )
    print(f'urgency "3" converts to {converted.urgency} ({type(converted.urgency).__name__})')

    for label, data in (
        (
            "urgency=9",
            {
                "subscriber_id": "TP00001",
                "category": "billing",
                "urgency": 9,
                "wants_refund": False,
            },
        ),
        (
            'category="refund"',
            {
                "subscriber_id": "TP00001",
                "category": "refund",
                "urgency": 3,
                "wants_refund": True,
            },
        ),
    ):
        try:
            TicketRecord.model_validate(data)
        except ValidationError as error:
            first = error.errors()[0]
            print(f"{label}: field={first['loc'][0]}, error_type={first['type']}")


def run() -> None:
    """Process all tickets, retry invalid outputs, and write JSONL results."""
    os.environ["LLM_MODE"] = "mock"
    demonstrate_pydantic()
    with TICKETS_PATH.open(newline="", encoding="utf-8-sig") as file:
        tickets = list(csv.DictReader(file))
    if len(tickets) != 150:
        raise ValueError(f"Expected 150 tickets, got {len(tickets)}")

    chain, _ = build_chain()
    mock_llm.reset()
    outcome_counts: Counter[str] = Counter()
    first_failures: Counter[str] = Counter()
    results: list[dict[str, Any]] = []
    valid_records: list[TicketRecord] = []

    for row in tickets:
        status, attempts, record = run_one_ticket(
            chain, row["text"], first_failures
        )
        outcome_counts[status] += 1
        if record is not None:
            valid_records.append(record)
            record_data = json.loads(record.model_dump_json())
        else:
            record_data = None
        results.append(
            {
                "ticket_id": row["ticket_id"],
                "status": status,
                "attempts": attempts,
                "record": record_data,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    total_attempts = sum(result["attempts"] for result in results)
    recovered_429 = mock_llm._state["calls"] - total_attempts
    refund_count = sum(record.wants_refund for record in valid_records)
    null_subscriber_ids = sum(record.subscriber_id is None for record in valid_records)

    print(
        f"outcomes: first_attempt={sum(r['attempts'] == 1 for r in results)}, "
        f"after_one_retry={sum(r['attempts'] == 2 for r in results)}, "
        f"manual_review={outcome_counts['manual_review']}"
    )
    print(
        "first-attempt failures: "
        f"invalid_json={first_failures['chatty_prose'] + first_failures['unreadable_ticket']} "
        f"(chatty_prose={first_failures['chatty_prose']}, "
        f"unreadable={first_failures['unreadable_ticket']}), "
        f"schema_urgency={first_failures['schema_urgency']}, "
        f"schema_category={first_failures['schema_category']}"
    )
    print(
        f"recovered_429={recovered_429}; valid_records={len(valid_records)}; "
        f"wants_refund={refund_count}; null_subscriber_id={null_subscriber_ids}"
    )
    print(f"saved {len(results)} records to {OUTPUT_PATH.relative_to(ROOT)}")

    assert outcome_counts == Counter({"valid": 146, "manual_review": 4})
    assert sum(result["attempts"] == 1 for result in results) == 118
    assert sum(result["attempts"] == 2 for result in results) == 28
    assert first_failures == Counter(
        {
            "chatty_prose": 12,
            "unreadable_ticket": 4,
            "schema_urgency": 6,
            "schema_category": 10,
        }
    )
    assert recovered_429 == 23
    assert refund_count == 31
    assert null_subscriber_ids == 75


if __name__ == "__main__":
    run()
