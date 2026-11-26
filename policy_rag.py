"""Chunk and retrieve policy documents for the Part D support assistant."""

import csv
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

import mock_llm
from llm_client import call_llm


ROOT = Path(__file__).resolve().parent
POLICY_DIR = ROOT / "data" / "policies"
EVAL_PATH = ROOT / "data" / "raw" / "policy_eval.csv"
FALLBACK = "I don't have enough information to answer that."
REJECTION_THRESHOLD = 0.32
_DEFAULT_STORE: tuple[Any, Any] | None = None


def chunk_text(text: str, size: int = 400, overlap: int = 60) -> list[str]:
    """Split text into fixed-size overlapping chunks using the project rule."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")

    step = size - overlap
    chunks = []
    start = 0
    while start + overlap < len(text):
        chunks.append(text[start : start + size])
        start += step
    return chunks


def create_policy_collection(size: int = 400, overlap: int = 60) -> tuple[Any, Any]:
    """Create an in-memory cosine collection and add uniquely identified chunks."""
    client = chromadb.Client()
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.create_collection(
        name=f"policy_chunks_{size}_{overlap}",
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
    )
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    for path in sorted(POLICY_DIR.glob("*.txt")):
        for index, chunk in enumerate(
            chunk_text(path.read_text(encoding="utf-8"), size=size, overlap=overlap)
        ):
            ids.append(f"{path.name}_{index}")
            documents.append(chunk)
            metadatas.append({"source": path.name, "chunk": index})
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return client, collection


def retrieve_from(collection: Any, query: str, k: int = 3) -> list[dict[str, Any]]:
    """Return the nearest policy chunks with cosine similarity and source."""
    if k < 1:
        raise ValueError("k must be at least 1")
    result = collection.query(
        query_texts=[query],
        n_results=k,
        include=["metadatas", "documents", "distances"],
    )
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {
            "source": metadata["source"],
            "chunk": metadata["chunk"],
            "similarity": 1.0 - float(distance),
            "text": document,
        }
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]


def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Search the default 400/60 policy collection."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = create_policy_collection()
    return retrieve_from(_DEFAULT_STORE[1], query, k=k)


def load_eval_rows() -> list[dict[str, str]]:
    """Read the fixed policy retrieval evaluation set."""
    with EVAL_PATH.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def keyword_baseline(question: str, policy_paths: list[Path]) -> str:
    """Select the file with the most shared non-stopword tokens."""
    question_words = set(re.findall(r"[a-z]+", question.lower())) - mock_llm.STOP
    sorted_paths = sorted(policy_paths, key=lambda path: path.name)
    return max(
        sorted_paths,
        key=lambda path: len(
            (set(re.findall(r"[a-z]+", path.read_text(encoding="utf-8").lower()))
             - mock_llm.STOP)
            & question_words
        ),
    ).name


def evaluate_retrieval(collection: Any) -> None:
    """Print reference retrieval metrics for answerable and unanswerable questions."""
    eval_rows = load_eval_rows()
    answerable = [row for row in eval_rows if row["expected_file"]]
    unanswerable = [row for row in eval_rows if not row["expected_file"]]
    retrieved = [
        (row, retrieve_from(collection, row["question"], k=3)) for row in eval_rows
    ]
    answerable_results = [
        (row, items) for row, items in retrieved if row["expected_file"]
    ]
    unanswerable_results = [
        items for row, items in retrieved if not row["expected_file"]
    ]
    hits_at_3 = sum(
        any(item["source"] == row["expected_file"] for item in items)
        for row, items in answerable_results
    )
    top1_hits = sum(
        items[0]["source"] == row["expected_file"]
        for row, items in answerable_results
    )
    answerable_similarities = [items[0]["similarity"] for _, items in answerable_results]
    unanswerable_similarities = [items[0]["similarity"] for items in unanswerable_results]

    policy_paths = list(POLICY_DIR.glob("*.txt"))
    baseline_hits = sum(
        keyword_baseline(row["question"], policy_paths) == row["expected_file"]
        for row in answerable
    )
    print(
        f"retrieval file hits: top3={hits_at_3}/{len(answerable)}, "
        f"top1={top1_hits}/{len(answerable)}"
    )
    print(
        "answerable top1 similarity: "
        f"{min(answerable_similarities):.3f} to {max(answerable_similarities):.3f}; "
        f"mean={sum(answerable_similarities) / len(answerable_similarities):.3f}"
    )
    print(
        "unanswerable top1 similarities: "
        + ", ".join(f"{value:.3f}" for value in unanswerable_similarities)
    )
    print(f"keyword baseline top1: {baseline_hits}/{len(answerable)}")

    assert len(eval_rows) == 16
    assert hits_at_3 == 12
    assert top1_hits == 12
    assert 0.28 <= min(answerable_similarities) <= 0.39
    assert 0.58 <= max(answerable_similarities) <= 0.69
    assert baseline_hits == 9


def _answer_from_results(
    question: str, results: list[dict[str, Any]], threshold: float
) -> tuple[str, list[str]]:
    """Answer only when retrieval clears the threshold, returning source names."""
    if not results:
        raise RuntimeError("Policy retrieval returned no chunks")
    sources = [item["source"] for item in results]
    if results[0]["similarity"] < threshold:
        return FALLBACK, sources

    context = "\n\n".join(
        f"[{item['source']}]\n{item['text']}" for item in results
    )
    user_message = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    answer_text = call_llm(
        [
            {
                "role": "system",
                "content": (
                    "POLICY ASSISTANT\nAnswer using only the context below. "
                    f"If it does not contain the answer, say exactly: {FALLBACK}"
                ),
            },
            {"role": "user", "content": user_message},
        ]
    )
    return answer_text, sources


def answer(question: str) -> tuple[str, list[str]]:
    """Return a thresholded, context-grounded policy answer and its sources."""
    return _answer_from_results(question, retrieve(question, k=3), REJECTION_THRESHOLD)


def ungrounded_answer(question: str) -> str:
    """Run the intentionally ungrounded baseline prompt without a threshold."""
    return call_llm(
        [
            {"role": "system", "content": "POLICY ASSISTANT. Answer the question."},
            {"role": "user", "content": question},
        ]
    )


def normalized(text: str) -> str:
    """Collapse whitespace before checking facts in chunk-boundary fragments."""
    return " ".join(text.split())


def evaluate_answers(collection: Any, eval_rows: list[dict[str, str]]) -> None:
    """Write grounded/ungrounded results and check all exact mock references."""
    mock_llm.reset()
    grounded_results = []
    retrieval_results = []
    for row in eval_rows:
        hits = retrieve_from(collection, row["question"], k=3)
        retrieval_results.append(hits)
        grounded_results.append(
            _answer_from_results(row["question"], hits, REJECTION_THRESHOLD)
        )

    mock_llm.reset()
    ungrounded_results = [ungrounded_answer(row["question"]) for row in eval_rows]

    rows_out = []
    grounded_fact_hits = 0
    retrieved_fact_hits = 0
    fallback_count = 0
    for row, hits, (grounded, sources), ungrounded in zip(
        eval_rows, retrieval_results, grounded_results, ungrounded_results
    ):
        expected_fact = row["expected_fact"]
        if expected_fact:
            if any(expected_fact in normalized(hit["text"]) for hit in hits):
                retrieved_fact_hits += 1
            if expected_fact in normalized(grounded):
                grounded_fact_hits += 1
        else:
            fallback_count += grounded == FALLBACK

        rows_out.append(
            {
                "question": row["question"],
                "expected_file": row["expected_file"],
                "top1_source": hits[0]["source"],
                "top1_similarity": f"{hits[0]['similarity']:.6f}",
                "hit_at_3": any(
                    hit["source"] == row["expected_file"] for hit in hits
                )
                if row["expected_file"]
                else False,
                "grounded_answer": grounded,
                "ungrounded_answer": ungrounded,
            }
        )

    output_path = ROOT / "outputs" / "rag_eval.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question",
                "expected_file",
                "top1_source",
                "top1_similarity",
                "hit_at_3",
                "grounded_answer",
                "ungrounded_answer",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    ungrounded_unique = set(ungrounded_results)
    print(
        f"400/60 expected fact in retrieved top3: "
        f"{retrieved_fact_hits}/12; grounded expected fact: {grounded_fact_hits}/12"
    )
    print(
        f"grounded fallback: {fallback_count}/4; "
        f"unique ungrounded answers: {len(ungrounded_unique)}"
    )
    print(f"saved {len(rows_out)} rows to {output_path.relative_to(ROOT)}")
    assert grounded_fact_hits == 12
    assert retrieved_fact_hits == 12
    assert fallback_count == 4
    assert ungrounded_unique == {
        "As of my last update, this is usually processed within 3 working days and the fee is typically INR 500."
    }


def evaluate_small_chunks(eval_rows: list[dict[str, str]]) -> None:
    """Compare fact retrieval and grounded answers with 200/30 character chunks."""
    client, collection = create_policy_collection(size=200, overlap=30)
    chunk_count = sum(
        len(chunk_text(path.read_text(encoding="utf-8"), size=200, overlap=30))
        for path in POLICY_DIR.glob("*.txt")
    )
    answerable = [row for row in eval_rows if row["expected_file"]]
    mock_llm.reset()
    retrieved_fact_hits = 0
    grounded_fact_hits = 0
    for row in answerable:
        hits = retrieve_from(collection, row["question"], k=3)
        expected_fact = row["expected_fact"]
        if any(expected_fact in normalized(hit["text"]) for hit in hits):
            retrieved_fact_hits += 1
        grounded, _ = _answer_from_results(
            row["question"], hits, REJECTION_THRESHOLD
        )
        if expected_fact in normalized(grounded):
            grounded_fact_hits += 1

    print(
        f"200/30 chunks: {chunk_count}; expected fact in top3="
        f"{retrieved_fact_hits}/12; grounded expected fact={grounded_fact_hits}/12"
    )
    assert chunk_count == 25
    assert retrieved_fact_hits == 11
    assert grounded_fact_hits == 8
    del collection, client


def main() -> None:
    """Run chunking, retrieval, grounded-answer, and chunk-size evaluations."""
    global _DEFAULT_STORE
    os.environ["LLM_MODE"] = "mock"
    counts: Counter[str] = Counter()
    for path in sorted(POLICY_DIR.glob("*.txt")):
        chunks = chunk_text(path.read_text(encoding="utf-8"))
        counts[path.name] = len(chunks)
        print(f"{path.name}: {len(chunks)} chunks")

    expected = {
        "cancellation_policy.txt": 2,
        "plan_change_policy.txt": 3,
        "recharge_and_refund_policy.txt": 3,
        "retention_offers_policy.txt": 3,
        "technical_support_policy.txt": 2,
    }
    if counts != Counter(expected):
        raise AssertionError(f"Unexpected policy chunk counts: {dict(counts)}")
    print(f"total: {sum(counts.values())} chunks")
    client, collection = create_policy_collection()
    _DEFAULT_STORE = (client, collection)
    eval_rows = load_eval_rows()
    evaluate_retrieval(collection)
    evaluate_answers(collection, eval_rows)
    evaluate_small_chunks(eval_rows)


if __name__ == "__main__":
    main()
