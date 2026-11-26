"""Build policy-grounded retention briefs for high-priority subscribers."""

import json
import os
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import mock_llm
import numpy as np
import pandas as pd
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

from llm_client import call_llm
from policy_rag import retrieve
from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PROJECT_DIR,
    build_model_table,
)


DATABASE_PATH = PROJECT_DIR / "data" / "tataplay.db"
OUTPUT_DIR = PROJECT_DIR / "outputs"
PRIORITY_PATH = OUTPUT_DIR / "retention_priority.csv"
SEGMENTS_PATH = OUTPUT_DIR / "segments.csv"
MODEL_PATH = OUTPUT_DIR / "churn_model.joblib"
POLICY_DIR = PROJECT_DIR / "data" / "policies"
SNAPSHOT_DATE = "2026-03-31"


class RetentionBrief(BaseModel):
    """Validated, policy-grounded recommendation for one subscriber."""

    subscriber_id: str
    risk_band: Literal["High", "Medium", "Low"]
    recommended_action: str
    policy_source: str


def call_prompt(prompt_value: Any) -> str:
    """Translate LangChain messages to the shared chat client format."""
    role_names = {"system": "system", "human": "user", "ai": "assistant"}
    messages = [
        {"role": role_names[message.type], "content": message.content}
        for message in prompt_value.to_messages()
    ]
    return call_llm(messages)


@lru_cache(maxsize=1)
def build_retention_chain() -> tuple[Any, PydanticOutputParser]:
    """Build the prompt, shared LLM function, and Pydantic parser chain."""
    parser = PydanticOutputParser(pydantic_object=RetentionBrief)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "RETENTION BRIEF\nReply with one JSON object only. Use only the "
                "policy context in the user message; do not invent policy details.\n"
                "{format_instructions}",
            ),
            MessagesPlaceholder("history"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | call_prompt | parser, parser


@lru_cache(maxsize=1)
def load_churn_model() -> Any:
    """Load the saved Part B pipeline once for all subscriber rescoring."""
    return joblib.load(MODEL_PATH)


def read_database_inputs(subscriber_id: str) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Read one subscriber, their observed recharges, and SQL-derived facts."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        subscriber = pd.read_sql_query(
            "SELECT * FROM subscribers WHERE subscriber_id = ?",
            connection,
            params=(subscriber_id,),
        )
        recharges = pd.read_sql_query(
            """
            SELECT *
            FROM recharges
            WHERE subscriber_id = ? AND recharge_date <= ?
            """,
            connection,
            params=(subscriber_id, SNAPSHOT_DATE),
        )
        summary = pd.read_sql_query(
            """
            SELECT
                s.tenure_months,
                COALESCE(
                    SUM(
                        CASE
                            WHEN r.status = 'failed' AND r.recharge_date <= ? THEN 1
                        END
                    ),
                    0
                ) AS n_failed_6m
            FROM subscribers AS s
            LEFT JOIN recharges AS r
                ON r.subscriber_id = s.subscriber_id
            WHERE s.subscriber_id = ?
            GROUP BY s.subscriber_id, s.tenure_months
            """,
            connection,
            params=(SNAPSHOT_DATE, subscriber_id),
        )

    if len(subscriber) != 1 or len(summary) != 1:
        raise KeyError(f"Expected exactly one database row for {subscriber_id}")
    tenure_months = int(summary.iloc[0]["tenure_months"])
    failed_recharges = int(summary.iloc[0]["n_failed_6m"])
    return subscriber, recharges, tenure_months, failed_recharges


def determine_risk_band(probability: float) -> str:
    """Map the probability to the assignment's fixed risk bands."""
    if probability >= 0.40:
        return "High"
    if probability >= 0.20:
        return "Medium"
    return "Low"


def expected_action(
    risk_band: str, tenure_months: int, failed_recharges: int
) -> str:
    """Apply the retention-offer precedence stated in the policy."""
    if risk_band == "High" and failed_recharges >= 3:
        return "Offer an auto-renew setup call and a 7 day grace extension."
    if risk_band == "High" and tenure_months >= 12:
        return "Offer a loyalty discount of 15 percent on the next three recharges."
    if risk_band == "High":
        return "Offer a free add-on pack for 2 months."
    return "No retention offer applies; keep monitoring the account."


def validate_consistency(
    brief: RetentionBrief,
    subscriber_id: str,
    risk_band: str,
    required_action: str,
) -> None:
    """Reject schema-valid model replies that contradict the supplied facts."""
    policy_files = {path.name for path in POLICY_DIR.glob("*.txt")}
    if brief.subscriber_id != subscriber_id:
        raise ValueError("subscriber_id does not match the supplied facts")
    if brief.risk_band != risk_band:
        raise ValueError("risk_band does not match the supplied facts")
    if brief.policy_source not in policy_files:
        raise ValueError("policy_source is not a real policy file")
    if brief.policy_source != "retention_offers_policy.txt":
        raise ValueError("policy_source must be retention_offers_policy.txt")
    if brief.recommended_action != required_action:
        raise ValueError("recommended_action does not follow the retention rule")


def parse_brief(
    facts_and_context: str,
    subscriber_id: str,
    risk_band: str,
    required_action: str,
) -> tuple[RetentionBrief, int]:
    """Parse one brief, retrying invalid or inconsistent replies up to twice."""
    chain, _ = build_retention_chain()
    history = [HumanMessage(content=facts_and_context)]

    for attempt in range(1, 4):
        try:
            brief = chain.invoke({"history": history})
        except OutputParserException as error:
            if error.llm_output is None:
                raise RuntimeError("Parser failure did not include the model reply") from error
            history.extend(
                [
                    AIMessage(content=error.llm_output),
                    HumanMessage(
                        content=f"VALIDATION ERROR: {error}\n\n{facts_and_context}"
                    ),
                ]
            )
            continue

        try:
            validate_consistency(
                brief, subscriber_id, risk_band, required_action
            )
        except ValueError as error:
            history.extend(
                [
                    AIMessage(content=brief.model_dump_json()),
                    HumanMessage(
                        content=f"VALIDATION ERROR: {error}\n\n{facts_and_context}"
                    ),
                ]
            )
            continue
        return brief, attempt

    raise RuntimeError(
        f"Could not produce a consistent retention brief for {subscriber_id}"
    )


def build_brief(subscriber_id: str) -> dict[str, Any]:
    """Build and validate one subscriber's policy-grounded retention brief."""
    priorities = pd.read_csv(PRIORITY_PATH)
    priority_rows = priorities.loc[priorities["subscriber_id"] == subscriber_id]
    if len(priority_rows) != 1:
        raise KeyError(f"Expected exactly one priority row for {subscriber_id}")
    priority_row = priority_rows.iloc[0]
    saved_probability = float(priority_row["churn_probability"])

    subscriber, recharges, tenure_months, failed_recharges = read_database_inputs(
        subscriber_id
    )
    feature_row = build_model_table(subscribers=subscriber, recharges=recharges)
    row_failed = int(feature_row.iloc[0]["n_failed_6m"])
    if row_failed != failed_recharges:
        raise AssertionError(
            f"SQL and shared feature builder disagree on failed recharges for {subscriber_id}"
        )

    model_features = feature_row[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ].replace({None: np.nan})
    rescored_probability = float(load_churn_model().predict_proba(model_features)[0, 1])
    if round(saved_probability, 4) != round(rescored_probability, 4):
        raise AssertionError(
            f"Saved and rescored probabilities differ for {subscriber_id}: "
            f"{saved_probability:.6f} vs {rescored_probability:.6f}"
        )

    risk_band = determine_risk_band(saved_probability)
    segment_rows = pd.read_csv(SEGMENTS_PATH)
    subscriber_segments = segment_rows.loc[
        segment_rows["subscriber_id"] == subscriber_id, "segment_name"
    ]
    if len(subscriber_segments) != 1:
        raise KeyError(f"Expected exactly one segment row for {subscriber_id}")
    segment = str(subscriber_segments.iloc[0])

    query = (
        f"retention offer for a {risk_band} risk subscriber with "
        f"{tenure_months} months tenure and {failed_recharges} failed recharges"
    )
    policy_chunks = retrieve(query, k=3)
    if len(policy_chunks) != 3:
        raise AssertionError(f"Expected three policy chunks for {subscriber_id}")
    facts = {
        "subscriber_id": subscriber_id,
        "churn_probability": saved_probability,
        "rescored_churn_probability": rescored_probability,
        "risk_band": risk_band,
        "tenure_months": tenure_months,
        "failed_recharges_6m": failed_recharges,
        "segment": segment,
    }
    context = "\n\n".join(
        f"[{chunk['source']}]\n{chunk['text']}" for chunk in policy_chunks
    )
    facts_and_context = (
        "FACTS:\n"
        f"subscriber_id: {subscriber_id}\n"
        f"risk_band: {risk_band}\n"
        f"tenure_months: {tenure_months}\n"
        f"failed_recharges_6m: {failed_recharges}\n"
        f"segment: {segment}\n\n"
        f"POLICY CONTEXT:\n{context}"
    )
    action = expected_action(risk_band, tenure_months, failed_recharges)
    brief, attempts = parse_brief(
        facts_and_context, subscriber_id, risk_band, action
    )
    return {
        "facts": facts,
        "brief": brief.model_dump(),
        "attempts": attempts,
    }


def run() -> None:
    """Create all 15 retention briefs and report validated counts."""
    os.environ["LLM_MODE"] = "mock"
    mock_llm.reset()
    subscribers = pd.read_csv(PRIORITY_PATH)["subscriber_id"].astype(str).tolist()
    if len(subscribers) != 15 or len(set(subscribers)) != 15:
        raise ValueError("Expected 15 distinct subscribers in retention_priority.csv")

    results = [build_brief(subscriber_id) for subscriber_id in subscribers]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "retention_briefs.jsonl"
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    risk_counts = Counter(result["brief"]["risk_band"] for result in results)
    action_counts = Counter(
        result["brief"]["recommended_action"] for result in results
    )
    print(f"valid and consistent briefs: {len(results)}/15")
    print(
        "risk bands: "
        + str({band: risk_counts.get(band, 0) for band in ("High", "Medium", "Low")})
    )
    print("recommended actions:")
    for action, count in sorted(action_counts.items()):
        print(f"  {count}: {action}")
    print(f"saved {len(results)} briefs to {output_path.relative_to(PROJECT_DIR)}")

    assert len(results) == 15
    assert all(result["attempts"] <= 3 for result in results)


if __name__ == "__main__":
    run()
