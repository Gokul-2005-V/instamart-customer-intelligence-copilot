
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# Allow Python to import mock_llm.py from the project root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mock_llm  # noqa: E402


POLICY_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs"

CHUNK_SIZE = 40
DEFAULT_OVERLAP = 10
TOP_K = 3
MAX_LLM_ATTEMPTS = 4

FALLBACK_TEXT = (
    "I could not find that information in the policy documents."
)


# ============================================================
# 2. POLICY CHUNKING
# ============================================================

def load_chunks(
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Split policy documents into 40-word chunks.

    With overlap=10, chunk starts are 0, 30, 60, etc.
    Each chunk retains its source filename.
    """

    if not 0 <= overlap < CHUNK_SIZE:
        raise ValueError(
            f"overlap must be between 0 and {CHUNK_SIZE - 1}"
        )

    if not POLICY_DIR.exists():
        raise FileNotFoundError(
            f"Policy directory not found: {POLICY_DIR}"
        )

    files = sorted(POLICY_DIR.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"No policy .txt files found in {POLICY_DIR}"
        )

    chunks: list[dict[str, Any]] = []
    step = CHUNK_SIZE - overlap

    for file_path in files:
        words = file_path.read_text(
            encoding="utf-8"
        ).split()

        if not words:
            continue

        for start in range(0, len(words), step):
            chunk_words = words[start:start + CHUNK_SIZE]

            chunks.append({
                "source": file_path.name,
                "start_word": start,
                "end_word": start + len(chunk_words),
                "text": " ".join(chunk_words),
            })

            # Stop once the current chunk reaches the end.
            if start + CHUNK_SIZE >= len(words):
                break

    return chunks


# ============================================================
# 3. TF-IDF RETRIEVER
# ============================================================

class PolicyRetriever:
    """
    Fit TF-IDF on policy chunks and rank them by cosine similarity.
    """

    def __init__(
        self,
        overlap: int = DEFAULT_OVERLAP,
    ):
        self.overlap = overlap
        self.chunks = load_chunks(overlap)

        if not self.chunks:
            raise ValueError("No policy chunks available.")

        self.vectorizer = TfidfVectorizer()

        self.matrix = self.vectorizer.fit_transform([
            chunk["text"] for chunk in self.chunks
        ])

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Return the top-k chunks as dictionaries containing:
        source, score, and text.
        """

        if not isinstance(query, str) or not query.strip():
            return []

        if k <= 0:
            return []

        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.matrix,
        ).ravel()

        # Stable sorting makes equal-score results deterministic.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: (-scores[i], i),
        )

        results = []

        for index in ranked_indices[:k]:
            chunk = self.chunks[index]

            results.append({
                "source": chunk["source"],
                "score": float(scores[index]),
                "text": chunk["text"],
            })

        return results


# ============================================================
# 4. MOCK LLM RESPONSE PARSING
# ============================================================

def extract_response(response: Any) -> tuple[str, int]:
    """
    Read the actual mock_llm.chat() response structure:

        response.choices[0].message.content
        response.usage.total_tokens
    """

    try:
        message = response.choices[0].message
        content = message.content or ""
    except (AttributeError, IndexError, TypeError):
        content = ""

    try:
        tokens_used = int(response.usage.total_tokens)
    except (AttributeError, TypeError, ValueError):
        tokens_used = 0

    return str(content).strip(), tokens_used


def call_mock_llm(
    messages: list[dict[str, Any]],
) -> tuple[str, int]:
    """Call the offline mock LLM with bounded retry handling."""

    last_error: Exception | None = None

    for attempt in range(MAX_LLM_ATTEMPTS):
        try:
            response = mock_llm.chat(messages)
            return extract_response(response)

        except Exception as exc:
            last_error = exc

            if attempt < MAX_LLM_ATTEMPTS - 1:
                time.sleep(0.2 * (attempt + 1))

    raise RuntimeError(
        f"mock_llm.chat failed after {MAX_LLM_ATTEMPTS} "
        f"attempts: {last_error}"
    ) from last_error


# ============================================================
# 5. RAG ANSWER GENERATION
# ============================================================

def answer(
    question: str,
    retriever: PolicyRetriever,
) -> dict[str, Any]:
    """
    Retrieve the top three chunks and ask the policy assistant
    to answer exclusively from the retrieved context.
    """

    retrieved = retriever.retrieve(question, k=TOP_K)

    context_lines = [
        f"[{item['source']}] {item['text']}"
        for item in retrieved
    ]

    context = "\n".join(context_lines)

    system_message = (
        "POLICY ASSISTANT\n"
        "Answer only from the supplied policy context. "
        "Do not invent facts or policy rules. "
        "If the answer is not present in the context, "
        "respond with a sentence beginning exactly with "
        "'I could not find'."
    )

    # Required user prompt structure.
    user_message = (
        f"CONTEXT:\n{context}\n"
        f"QUESTION:\n{question}"
    )

    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    try:
        answer_text, tokens_used = call_mock_llm(messages)

        if not answer_text:
            answer_text = FALLBACK_TEXT

        return {
            "question": question,
            "answer": answer_text,
            "retrieved": retrieved,
            "tokens_used": tokens_used,
        }

    except RuntimeError as exc:
        # Do not silently return an empty answer if the LLM fails.
        return {
            "question": question,
            "answer": FALLBACK_TEXT,
            "retrieved": retrieved,
            "tokens_used": 0,
            "error": str(exc),
        }


# ============================================================
# 6. EVALUATION HELPERS
# ============================================================

def clean_text(value: Any) -> str:
    """Convert CSV values to trimmed strings safely."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def policy_source_for_label(
    expected_doc: str,
    policy_sources: list[str],
) -> str | None:
    """
    Resolve an expected-document label to a real policy filename.

    Accepts labels such as:
        refund_policy.txt
        refund_policy
        A path ending in refund_policy.txt

    A blank or unrelated label is not considered answerable.
    """

    label = clean_text(expected_doc).replace("\\", "/").lower()

    if not label:
        return None

    label_basename = label.rsplit("/", 1)[-1]

    for source in policy_sources:
        source_lower = source.lower()
        stem_lower = Path(source).stem.lower()

        if label_basename in {
            source_lower,
            stem_lower,
        }:
            return source

    return None


def to_bool(value: Any) -> bool:
    """Convert a Boolean or Boolean-like value to bool."""

    if isinstance(value, bool):
        return value

    value = clean_text(value).lower()

    return value in {"true", "1", "yes"}


def evaluate(
    overlap: int = DEFAULT_OVERLAP,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Evaluate policy questions using policy_eval.csv.

    Answerability is determined by whether expected_doc maps
    to an actual policy filename. This correctly distinguishes
    the two out-of-scope questions from the 12 policy questions.
    """

    if not EVAL_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EVAL_FILE}"
        )

    # Preserve empty strings so unanswerable labels remain visible.
    eval_df = pd.read_csv(
        EVAL_FILE,
        keep_default_na=False,
    )

    eval_df.columns = [
        str(column).strip().lower()
        for column in eval_df.columns
    ]

    required = {
        "question",
        "expected_doc",
        "expected_fact",
    }

    missing = required - set(eval_df.columns)

    if missing:
        raise ValueError(
            f"Missing evaluation columns: {sorted(missing)}. "
            f"Available columns: {list(eval_df.columns)}"
        )

    policy_sources = sorted(
        path.name for path in POLICY_DIR.glob("*.txt")
    )

    if not policy_sources:
        raise FileNotFoundError(
            f"No policy files found in {POLICY_DIR}"
        )

    retriever = PolicyRetriever(overlap=overlap)

    # Reset the deterministic mock LLM counters for this batch.
    mock_llm.reset()

    rows: list[dict[str, Any]] = []

    for _, csv_row in eval_df.iterrows():
        question = clean_text(csv_row["question"])
        expected_doc = clean_text(csv_row["expected_doc"])
        expected_fact = clean_text(csv_row["expected_fact"])

        if not question:
            raise ValueError(
                "policy_eval.csv contains an empty question."
            )

        expected_source = policy_source_for_label(
            expected_doc,
            policy_sources,
        )

        is_answerable = expected_source is not None

        result = answer(question, retriever)

        answer_text = result["answer"]
        retrieved = result["retrieved"]

        retrieved_sources = [
            item["source"] for item in retrieved
        ]

        # Expected document must be present in the top three.
        doc_found = (
            expected_source in retrieved_sources
            if is_answerable
            else ""
        )

        # The assignment requires a case-sensitive substring check.
        fact_found = (
            expected_fact in answer_text
            if is_answerable and expected_fact
            else ""
        )

        # Unanswerable questions must use the specified fallback.
        fallback_correct = (
            answer_text.startswith("I could not find")
            if not is_answerable
            else ""
        )

        rows.append({
            "overlap": overlap,
            "question": question,
            "expected_doc": expected_doc,
            "expected_source": expected_source or "",
            "expected_fact": expected_fact,
            "answerable": is_answerable,
            "answer": answer_text,
            "retrieved_sources": " | ".join(
                retrieved_sources
            ),
            "retrieved_scores": " | ".join(
                f"{item['score']:.4f}" for item in retrieved
            ),
            "expected_doc_retrieved": doc_found,
            "expected_fact_found": fact_found,
            "fallback_correct": fallback_correct,
            "tokens_used": result["tokens_used"],
            "llm_error": result.get("error", ""),
        })

    results = pd.DataFrame(rows)

    answerable_rows = results[
        results["answerable"].map(to_bool)
    ]

    unanswerable_rows = results[
        ~results["answerable"].map(to_bool)
    ]

    doc_hits = sum(
        to_bool(value)
        for value in answerable_rows["expected_doc_retrieved"]
    )

    fact_hits = sum(
        to_bool(value)
        for value in answerable_rows["expected_fact_found"]
    )

    fallback_hits = sum(
        to_bool(value)
        for value in unanswerable_rows["fallback_correct"]
    )

    n_answerable = len(answerable_rows)
    n_unanswerable = len(unanswerable_rows)

    metrics = {
        "overlap": overlap,
        "chunk_count": len(retriever.chunks),
        "question_count": len(results),
        "answerable_questions": n_answerable,
        "unanswerable_questions": n_unanswerable,
        "expected_doc_in_top_3": int(doc_hits),
        "expected_doc_accuracy": (
            round(doc_hits / n_answerable, 4)
            if n_answerable else 0.0
        ),
        "expected_fact_in_answer": int(fact_hits),
        "expected_fact_accuracy": (
            round(fact_hits / n_answerable, 4)
            if n_answerable else 0.0
        ),
        "correct_unanswerable_fallbacks": int(fallback_hits),
        "unanswerable_fallback_accuracy": (
            round(fallback_hits / n_unanswerable, 4)
            if n_unanswerable else 0.0
        ),
        "llm_errors": int(
            results["llm_error"].astype(str).str.len().gt(0).sum()
        ),
    }

    return results, metrics


# ============================================================
# 7. SAVE CHUNKS
# ============================================================

def save_chunks(overlap: int) -> list[dict[str, Any]]:
    """Save the policy chunks and their source metadata."""

    chunks = load_chunks(overlap=overlap)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(chunks).to_csv(
        OUTPUT_DIR / f"policy_chunks_overlap_{overlap}.csv",
        index=False,
    )

    return chunks


# ============================================================
# 8. MAIN PIPELINE
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("PART D — RETRIEVAL-AUGMENTED GENERATION (RAG)")
    print("=" * 65)

    # --------------------------------------------------------
    # Check 1: 40-word chunks with 10-word overlap
    # --------------------------------------------------------

    chunks = save_chunks(DEFAULT_OVERLAP)

    chunk_df = pd.DataFrame(chunks)

    print(f"\nChunks with {DEFAULT_OVERLAP}-word overlap: {len(chunks)}")
    print("Chunks by document:")
    print(chunk_df["source"].value_counts().sort_index().to_string())

    expected_counts = {
        "delivery_policy.txt": 3,
        "escalation_policy.txt": 3,
        "payment_policy.txt": 3,
        "privacy_policy.txt": 2,
        "refund_policy.txt": 4,
    }

    actual_counts = chunk_df["source"].value_counts().to_dict()

    assert len(chunks) == 15, (
        f"Expected 15 chunks, found {len(chunks)}."
    )

    assert actual_counts == expected_counts, (
        f"Unexpected chunk distribution: {actual_counts}"
    )

    # --------------------------------------------------------
    # Check 2: Top-three refund retrieval
    # --------------------------------------------------------

    retriever = PolicyRetriever(overlap=DEFAULT_OVERLAP)

    query = "refund policy late delivery"
    top_results = retriever.retrieve(query, k=TOP_K)

    print(f"\nTop {TOP_K} sources for '{query}':")

    for item in top_results:
        print(
            f"  {item['source']} "
            f"(score={item['score']:.4f})"
        )

    assert len(top_results) == TOP_K
    assert all(
        item["source"] == "refund_policy.txt"
        for item in top_results
    ), "Refund query did not return refund_policy.txt in all top-three results."

    # --------------------------------------------------------
    # Check 3: Evaluate normal overlap
    # --------------------------------------------------------

    print("\nRunning evaluation with 10-word overlap...")

    normal_results, normal_metrics = evaluate(
        overlap=DEFAULT_OVERLAP
    )

    normal_path = OUTPUT_DIR / "policy_eval_results.csv"
    normal_results.to_csv(normal_path, index=False)

    print("\nMetrics — 10-word overlap:")
    print(json.dumps(normal_metrics, indent=2))

    # --------------------------------------------------------
    # Check 4: Evaluate zero overlap
    # --------------------------------------------------------

    zero_chunks = save_chunks(0)

    print(f"\nChunks with zero overlap: {len(zero_chunks)}")
    print("Running evaluation with zero overlap...")

    zero_results, zero_metrics = evaluate(overlap=0)

    zero_path = (
        OUTPUT_DIR / "policy_eval_zero_overlap_results.csv"
    )
    zero_results.to_csv(zero_path, index=False)

    print("\nMetrics — zero overlap:")
    print(json.dumps(zero_metrics, indent=2))

    # --------------------------------------------------------
    # Check 5: Compare expected-fact failures
    # --------------------------------------------------------

    zero_answerable = zero_results[
        zero_results["answerable"].map(to_bool)
    ]

    zero_fact_failures = zero_answerable[
        ~zero_answerable["expected_fact_found"].map(to_bool)
    ]

    print("\nZero-overlap answerable fact failures:")

    if zero_fact_failures.empty:
        print("  None")
    else:
        for _, row in zero_fact_failures.iterrows():
            print(f"\n  Question: {row['question']}")
            print(f"  Expected fact: {row['expected_fact']}")
            print(f"  Answer: {row['answer']}")

    # --------------------------------------------------------
    # Check 6: Save combined metrics
    # --------------------------------------------------------

    combined_metrics = {
        "overlap_10": normal_metrics,
        "overlap_0": zero_metrics,
        "zero_overlap_fact_failure_count": len(zero_fact_failures),
    }

    metrics_path = OUTPUT_DIR / "part_d_rag_metrics.json"

    metrics_path.write_text(
        json.dumps(combined_metrics, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Check 7: Assignment checkpoints
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("PART D RAG CHECKPOINTS")
    print("=" * 65)

    assert len(normal_results) == 14, (
        f"Expected 14 evaluation questions, got {len(normal_results)}."
    )

    assert normal_metrics["answerable_questions"] == 12, (
        "Expected 12 answerable questions. Check expected_doc "
        "labels in policy_eval.csv."
    )

    assert normal_metrics["unanswerable_questions"] == 2, (
        "Expected 2 unanswerable questions. Check expected_doc "
        "labels in policy_eval.csv."
    )

    assert normal_metrics["expected_doc_in_top_3"] == 12, (
        "Not all answerable questions retrieved their expected "
        "document in the top three."
    )

    assert normal_metrics["expected_fact_in_answer"] == 12, (
        "Not all 12 expected facts were found in the answers. "
        "Inspect policy_eval_results.csv."
    )

    assert normal_metrics["correct_unanswerable_fallbacks"] == 2, (
        "Both unanswerable questions must begin with "
        "'I could not find'."
    )

    assert len(zero_results) == 14

    assert len(zero_fact_failures) >= 1, (
        "Expected at least one expected-fact failure with zero "
        "overlap. Inspect the zero-overlap results and chunk "
        "boundaries if this checkpoint fails."
    )

    assert normal_metrics["llm_errors"] == 0, (
        "Some mock LLM calls failed. Inspect the evaluation CSV."
    )

    print("PASS: 15 policy chunks with 10-word overlap.")
    print("PASS: Expected per-policy chunk distribution.")
    print("PASS: Top three refund retrieval sources.")
    print("PASS: 12 answerable and 2 unanswerable questions.")
    print("PASS: All 12 expected documents retrieved in top three.")
    print("PASS: All 12 expected facts found.")
    print("PASS: Both unanswerable questions use the fallback.")
    print("PASS: Zero-overlap evaluation exposes a fact failure.")
    print("PASS: No mock LLM errors.")

    print("\nOutput files:")
    print(f"  {OUTPUT_DIR / 'policy_chunks_overlap_10.csv'}")
    print(f"  {OUTPUT_DIR / 'policy_chunks_overlap_0.csv'}")
    print(f"  {normal_path}")
    print(f"  {zero_path}")
    print(f"  {metrics_path}")

    print("\nALL PART D RAG CHECKPOINTS PASSED.")


if __name__ == "__main__":
    main()
