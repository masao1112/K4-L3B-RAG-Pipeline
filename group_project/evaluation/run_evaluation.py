"""Reproducible A/B evaluation for dense-only versus hybrid RAG retrieval.

The evaluator intentionally uses local BGE-M3 cosine similarity so both
configurations are judged by the same deterministic metric implementation.
Generation still uses the provider configured in ``.env``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.task10_generation import (  # noqa: E402
    SAFE_REFUSAL,
    SYSTEM_PROMPT,
    call_llm,
    format_context,
    reorder_for_llm,
)
from src.task4_chunking_indexing import EMBEDDING_MODEL  # noqa: E402
from src.task5_semantic_search import semantic_search  # noqa: E402
from src.task6_lexical_search import lexical_search  # noqa: E402
from src.task7_reranking import rerank_rrf  # noqa: E402


TOP_K = 5
RRF_K = 60
RELEVANCE_THRESHOLD = 0.20


def similarity_matrix(texts: list[str]):
    """Build a deterministic lexical-semantic proxy matrix for one case."""
    vectors = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
    ).fit_transform(texts)
    return cosine_similarity(vectors)


def split_claims(text: str) -> list[str]:
    """Split Vietnamese/English prose into claim-sized units."""
    claims = re.split(r"(?<=[.!?;:])\s+|\n+", text.strip())
    return [claim.strip(" -•\t") for claim in claims if len(claim.strip()) >= 12]


def retrieve_configs(question: str) -> dict[str, tuple[list[dict], float]]:
    """Retrieve both configurations while keeping every other setting equal."""
    dense_started = time.perf_counter()
    dense_candidates = semantic_search(question, top_k=TOP_K * 2)
    dense_latency = time.perf_counter() - dense_started

    hybrid_started = time.perf_counter()
    sparse_candidates = lexical_search(question, top_k=TOP_K * 2)
    hybrid = rerank_rrf(
        [dense_candidates, sparse_candidates],
        top_k=TOP_K,
        k=RRF_K,
    )
    hybrid_latency = dense_latency + (time.perf_counter() - hybrid_started)
    return {
        "dense_only": (dense_candidates[:TOP_K], dense_latency),
        "hybrid_rrf": (hybrid, hybrid_latency),
    }


def generate_answer(question: str, chunks: list[dict]) -> tuple[str, float, str | None]:
    """Generate one grounded answer and retain latency/error evidence."""
    context = format_context(reorder_for_llm(chunks))
    user_message = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Trích dẫn nguồn theo dạng [Document N]."
    )
    started = time.perf_counter()
    retry_delays = (10, 30, 60, 90)
    for attempt in range(5):
        try:
            answer = call_llm(SYSTEM_PROMPT, user_message)
            return answer, time.perf_counter() - started, None
        except Exception as exc:  # provider/network errors are evidence, not crashes
            if attempt == 4:
                error = f"{type(exc).__name__}: {str(exc)[:300]}"
                return SAFE_REFUSAL, time.perf_counter() - started, error
            time.sleep(retry_delays[attempt])
    return SAFE_REFUSAL, time.perf_counter() - started, "UnknownError"


def score_case(case: dict, answer: str, chunks: list[dict]) -> dict[str, float]:
    """Calculate four documented [0, 1] TF-IDF proxy metrics."""
    contexts = [chunk["content"] for chunk in chunks]
    answer_claims = split_claims(answer) or [answer]
    expected_claims = split_claims(case["expected_context"]) or [case["expected_context"]]

    texts = (
        [case["question"], case["expected_answer"], answer]
        + answer_claims
        + expected_claims
        + contexts
        + [case["expected_context"]]
    )
    similarities = similarity_matrix(texts)
    question_index, reference_answer_index, answer_index = 0, 1, 2
    cursor = 3
    answer_claim_indices = list(range(cursor, cursor + len(answer_claims)))
    cursor += len(answer_claims)
    expected_claim_indices = list(range(cursor, cursor + len(expected_claims)))
    cursor += len(expected_claims)
    context_indices = list(range(cursor, cursor + len(contexts)))
    cursor += len(contexts)
    expected_context_index = cursor

    if answer == SAFE_REFUSAL or not contexts:
        faithfulness = 0.0
        answer_relevance = 0.0
    else:
        claim_scores = [
            max(similarities[claim, context] for context in context_indices)
            for claim in answer_claim_indices
        ]
        faithfulness = sum(claim_scores) / len(claim_scores)
        answer_relevance = (
            similarities[answer_index, question_index]
            + similarities[answer_index, reference_answer_index]
        ) / 2

    recalled_claims = [
        max(similarities[claim, context] for context in context_indices)
        >= RELEVANCE_THRESHOLD
        for claim in expected_claim_indices
    ]
    context_recall = (
        sum(recalled_claims) / len(recalled_claims) if recalled_claims else 0.0
    )

    relevant = [
        similarities[expected_context_index, context] >= RELEVANCE_THRESHOLD
        for context in context_indices
    ]
    precision_sum = 0.0
    relevant_seen = 0
    for rank, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            relevant_seen += 1
            precision_sum += relevant_seen / rank
    context_precision = precision_sum / max(1, relevant_seen)

    return {
        "faithfulness": round(float(faithfulness), 4),
        "answer_relevance": round(float(answer_relevance), 4),
        "context_recall": round(float(context_recall), 4),
        "context_precision": round(float(context_precision), 4),
    }


def evaluate(
    dataset: list[dict],
    workers: int,
    existing_rows: list[dict] | None = None,
) -> dict:
    """Run retrieval, concurrent generation and deterministic scoring."""
    rows = [row for row in (existing_rows or []) if not row.get("provider_error")]
    completed_keys = {(row["case_index"], row["config"]) for row in rows}
    generation_jobs: list[tuple[int, str, dict, list[dict], float]] = []
    retrieved: dict[tuple[int, str], list[dict]] = {}

    for case_index, case in enumerate(dataset):
        for config, (chunks, retrieval_latency) in retrieve_configs(
            case["question"]
        ).items():
            key = (case_index, config)
            retrieved[key] = chunks
            for row in rows:
                if (row["case_index"], row["config"]) == key:
                    row["retrieval_latency_seconds"] = round(retrieval_latency, 4)
            if key not in completed_keys:
                generation_jobs.append(
                    (case_index, config, case, chunks, retrieval_latency)
                )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(generate_answer, case["question"], chunks): (
                case_index,
                config,
                case,
                chunks,
                retrieval_latency,
            )
            for case_index, config, case, chunks, retrieval_latency in generation_jobs
        }
        completed = 0
        for future in as_completed(futures):
            case_index, config, case, chunks, retrieval_latency = futures[future]
            answer, latency, error = future.result()
            rows.append(
                {
                    "case_index": case_index,
                    "config": config,
                    "question": case["question"],
                    "answer": answer,
                    "source_ids": [chunk["id"] for chunk in chunks],
                    "retrieval_latency_seconds": round(retrieval_latency, 4),
                    "latency_seconds": round(latency, 3),
                    "provider_error": error,
                    "metrics": {},
                }
            )
            completed += 1
            print(f"Completed {completed}/{len(generation_jobs)}", flush=True)

    rows.sort(key=lambda row: (row["case_index"], row["config"]))
    for row in rows:
        chunks = retrieved[(row["case_index"], row["config"])]
        row["metrics"] = score_case(
            dataset[row["case_index"]],
            row["answer"],
            chunks,
        )
    metric_names = (
        "faithfulness",
        "answer_relevance",
        "context_recall",
        "context_precision",
    )
    overall: dict[str, dict] = {}
    for config in ("dense_only", "hybrid_rrf"):
        config_rows = [row for row in rows if row["config"] == config]
        metrics = {
            name: round(
                sum(row["metrics"][name] for row in config_rows) / len(config_rows),
                4,
            )
            for name in metric_names
        }
        metrics["average"] = round(sum(metrics.values()) / len(metrics), 4)
        metrics["mean_latency_seconds"] = round(
            sum(row["latency_seconds"] for row in config_rows) / len(config_rows),
            3,
        )
        metrics["mean_retrieval_latency_seconds"] = round(
            sum(row["retrieval_latency_seconds"] for row in config_rows)
            / len(config_rows),
            4,
        )
        metrics["provider_errors"] = sum(
            row["provider_error"] is not None for row in config_rows
        )
        overall[config] = metrics

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "run": {
            "date": date.today().isoformat(),
            "framework": "custom-tfidf-evaluator-v1",
            "ragas_installed": version("ragas"),
            "generator_model": os.getenv("LLM_MODEL", ""),
            "embedding_model": os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL),
            "corpus_commit": commit,
            "golden_size": len(dataset),
            "top_k": TOP_K,
            "rrf_k": RRF_K,
            "relevance_threshold": RELEVANCE_THRESHOLD,
        },
        "overall": overall,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "group_project/evaluation/golden_dataset.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "group_project/evaluation/metrics.json",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep successful rows from an existing output and rerun failures only.",
    )
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    existing_rows = None
    if args.resume and args.output.exists():
        existing_rows = json.loads(args.output.read_text(encoding="utf-8")).get("cases")
    results = evaluate(
        dataset,
        workers=max(1, args.workers),
        existing_rows=existing_rows,
    )
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
