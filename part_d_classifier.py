"""Classify support tickets with weak, zero-shot, and few-shot prompts."""

import csv
import os
from pathlib import Path
from typing import Any

import mock_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm_client import call_llm


ROOT = Path(__file__).resolve().parent
TICKETS_PATH = ROOT / "data" / "raw" / "tickets.csv"
EXAMPLES_PATH = ROOT / "data" / "raw" / "few_shot_examples.csv"
OUTPUT_PATH = ROOT / "outputs" / "ticket_classification.csv"
LABELS = {"billing", "signal", "plan_change", "cancellation"}

WEAK_PROMPT = "TICKET CLASSIFIER. Classify the ticket."
ZERO_SHOT_PROMPT = """TICKET CLASSIFIER
You are a customer-support ticket classifier. Classify each support ticket into exactly one of these labels: billing, signal, plan_change, cancellation.
Respond with only the label. Do not create new labels."""


def call_prompt(prompt_value: Any) -> str:
    """Translate LangChain messages to the shared chat client format."""
    role_names = {"system": "system", "human": "user", "ai": "assistant"}
    messages = [
        {"role": role_names[message.type], "content": message.content}
        for message in prompt_value.to_messages()
    ]
    return call_llm(messages)


def load_rows(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV into dictionaries while preserving file order."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def make_chain():
    """Build the required prompt-to-client-to-parser LangChain pipeline."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", "{system_prompt}"), ("human", "{ticket}")]
    )
    return prompt | call_prompt | StrOutputParser()


def few_shot_prompt(examples: list[dict[str, str]]) -> str:
    """Append the examples in the exact line format used by the mock."""
    formatted = "\n".join(
        f"Ticket: {row['text']}\nLabel: {row['label']}" for row in examples
    )
    return f"{ZERO_SHOT_PROMPT}\n\nExamples:\n{formatted}"


def classify(
    chain: Any, tickets: list[dict[str, str]], system_prompt: str
) -> tuple[list[str], int]:
    """Run one prompt across all tickets and return labels and recovered 429s."""
    mock_llm.reset()
    labels = [
        chain.invoke({"system_prompt": system_prompt, "ticket": row["text"]})
        for row in tickets
    ]
    return labels, mock_llm._state["calls"] - len(tickets)


def summarize(
    name: str, predictions: list[str], tickets: list[dict[str, str]], recovered: int
) -> tuple[int, int]:
    """Print valid-label and accuracy metrics for one prompt run."""
    valid = sum(prediction in LABELS for prediction in predictions)
    correct = sum(
        prediction == row["true_category"]
        for prediction, row in zip(predictions, tickets)
    )
    accuracy = correct / len(tickets) * 100
    print(
        f"{name}: valid={valid}/{len(tickets)}, "
        f"correct={correct}/{len(tickets)} ({accuracy:.2f}%), "
        f"recovered_429={recovered}"
    )
    return valid, correct


def run() -> None:
    """Generate the classifier metrics and the required output CSV."""
    os.environ["LLM_MODE"] = "mock"
    tickets = [row for row in load_rows(TICKETS_PATH) if row["text"].strip() != "..."]
    examples = load_rows(EXAMPLES_PATH)
    if len(tickets) != 146 or len(examples) != 13:
        raise ValueError(
            f"Expected 146 readable tickets and 13 examples; got "
            f"{len(tickets)} tickets and {len(examples)} examples"
        )

    chain = make_chain()
    weak, weak_recovered = classify(chain, tickets, WEAK_PROMPT)
    zero_shot, zero_recovered = classify(chain, tickets, ZERO_SHOT_PROMPT)
    few_shot, few_recovered = classify(chain, tickets, few_shot_prompt(examples))

    weak_metrics = summarize("weak", weak, tickets, weak_recovered)
    zero_metrics = summarize("zero_shot", zero_shot, tickets, zero_recovered)
    few_metrics = summarize("few_shot", few_shot, tickets, few_recovered)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["ticket_id", "true_category", "zero_shot", "few_shot"])
        writer.writerows(
            (row["ticket_id"], row["true_category"], zero, few)
            for row, zero, few in zip(tickets, zero_shot, few_shot)
        )

    print("Few-shot errors:")
    errors = [
        (row, predicted)
        for row, predicted in zip(tickets, few_shot)
        if predicted != row["true_category"]
    ]
    for row, predicted in errors:
        print(
            f"{row['ticket_id']} true={row['true_category']} "
            f"predicted={predicted}: {row['text']}"
        )

    assert weak_metrics == (0, 0)
    assert zero_metrics == (146, 123)
    assert few_metrics == (146, 142)
    assert (weak_recovered, zero_recovered, few_recovered) == (18, 18, 18)
    assert len(errors) == 4
    print(f"Saved {len(tickets)} predictions to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    run()
