
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------------------------------------------------
# Project paths and import setup
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

# Allow this script to import mock_llm.py from the repository root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mock_llm


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
RAW_TICKETS = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS_FILE = ROOT / "outputs" / "customer_segments.csv"
OUTPUTS = ROOT / "outputs"

MAX_INPUT_TOKENS = 400
TOKEN_BUDGET_DEMO = 6000
TOKEN_BUDGET_FINAL = 100000

BASE_DELAY = 0.05
MAX_RETRIES = 3

ALLOWED_CATEGORIES = ["billing", "delivery", "technical", "fraud"]

# Required patterns and placeholders, applied in this exact order.
PII_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]

ZERO_SHOT_PROMPT = """
CLASSIFY customer support tickets.
Allowed categories: billing, delivery, technical, fraud.
Return JSON only, with exactly these fields:
{"category": "...", "reason": "..."}
The category must be one of the four allowed categories.
Do not include markdown or any text outside the JSON object.
""".strip()

FEW_SHOT_PROMPT = """
CLASSIFY customer support tickets.
Allowed categories: billing, delivery, technical, fraud.
Return JSON only, with exactly these fields:
{"category": "...", "reason": "..."}
The category must be one of the four allowed categories.
Do not include markdown or any text outside the JSON object.

EXAMPLES:
Ticket: I see an unauthorized transaction that I did not make.
Output: {"category": "fraud", "reason": "unauthorized transaction"}
Ticket: My app crashes and I cannot log in.
Output: {"category": "technical", "reason": "app and login issue"}
Ticket: My invoice shows the wrong amount.
Output: {"category": "billing", "reason": "incorrect invoice"}
""".strip()


# ------------------------------------------------------------
# Load ticket data
# ------------------------------------------------------------
def load_tickets():
    # Preserve empty strings instead of converting them to NaN.
    tickets = pd.read_csv(RAW_TICKETS, keep_default_na=False)

    required = {
        "ticket_id",
        "customer_id",
        "order_id",
        "text",
        "true_category",
    }
    missing = required - set(tickets.columns)

    if missing:
        raise ValueError(f"Missing ticket columns: {sorted(missing)}")

    return tickets


# ------------------------------------------------------------
# Privacy: redact personal information
# ------------------------------------------------------------
def redact_text(text):
    """Return redacted text and substitution counts."""
    result = str(text)
    counts = {}

    for name, pattern, placeholder in PII_PATTERNS:
        result, count = pattern.subn(placeholder, result)
        counts[name] = count

    # Prove that no targeted PII pattern remains.
    for name, pattern, _ in PII_PATTERNS:
        assert pattern.search(result) is None, (
            f"PII remains after redaction: {name}"
        )

    return result, counts


def prepare_redacted_tickets(tickets):
    tickets = tickets.copy()

    redacted_texts = []
    total_counts = {
        "email": 0,
        "phone": 0,
        "order_id": 0,
    }

    for text in tickets["text"]:
        redacted, counts = redact_text(text)
        redacted_texts.append(redacted)

        for name, count in counts.items():
            total_counts[name] += count

    tickets["redacted_text"] = redacted_texts

    total_counts["total"] = sum(total_counts.values())

    return tickets, total_counts


def assert_prompt_has_no_pii(messages):
    """Assert raw email addresses, phone numbers, and order IDs are absent."""
    full_prompt = "\n".join(
        message.get("content", "") for message in messages
    )

    for name, pattern, _ in PII_PATTERNS:
        assert pattern.search(full_prompt) is None, (
            f"Privacy assertion failed: raw {name} found in prompt"
        )


# ------------------------------------------------------------
# Model calls: token budget and retries
# ------------------------------------------------------------
def call_with_retries(
    messages,
    ticket_id,
    running_tokens,
    token_budget,
    rng,
):
    """
    Check the budget before each model call.

    Returns:
        response, running_tokens, ticket_tokens, call_status
    """
    ticket_tokens = 0

    for retry_number in range(MAX_RETRIES + 1):
        # Do not call the model if the budget has been exhausted.
        if running_tokens >= token_budget:
            return (
                None,
                running_tokens,
                ticket_tokens,
                "budget_exhausted",
            )

        assert_prompt_has_no_pii(messages)

        try:
            response = mock_llm.chat(messages)
            used = int(response.usage.total_tokens)

            running_tokens += used
            ticket_tokens += used

            return response, running_tokens, ticket_tokens, "ok"

        except mock_llm.TransientLLMError:
            if retry_number >= MAX_RETRIES:
                print(
                    f"PERMANENT_ERROR ticket={ticket_id} "
                    f"after {MAX_RETRIES} retries"
                )
                return (
                    None,
                    running_tokens,
                    ticket_tokens,
                    "error_permanent",
                )

            # Retry number 0: 0.045–0.0575 seconds.
            # Exponential backoff is retained for later retry numbers.
            wait = (
                BASE_DELAY
                * (2 ** retry_number)
                * rng.uniform(0.9, 1.15)
            )

            print(
                f"RETRY ticket={ticket_id} "
                f"retry_number={retry_number} "
                f"wait_seconds={wait:.4f}"
            )
            time.sleep(wait)

    return None, running_tokens, ticket_tokens, "error_permanent"


# ------------------------------------------------------------
# Structured output validation
# ------------------------------------------------------------
def validate_response(content):
    """
    Valid JSON category -> classified_json.
    Non-JSON with one recognised category word -> classified_keyword.
    Invalid JSON category or ambiguous output -> flagged_invalid_output.
    """
    if content is None:
        return "", "flagged_invalid_output"

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if isinstance(parsed, dict):
        category = parsed.get("category")

        if (
            isinstance(category, str)
            and category in ALLOWED_CATEGORIES
        ):
            return category, "classified_json"

        # Never silently accept an invalid category such as "shipping".
        return "", "flagged_invalid_output"

    # Word extraction correctly recognises "billing-related".
    words = set(re.findall(r"[a-z]+", str(content).lower()))
    found = words.intersection(ALLOWED_CATEGORIES)

    if len(found) == 1:
        return next(iter(found)), "classified_keyword"

    return "", "flagged_invalid_output"


# ------------------------------------------------------------
# Run the triage pipeline
# ------------------------------------------------------------
def run_pipeline(tickets, prompt, token_budget, run_name):
    """Process tickets in their original file order."""
    mock_llm.reset()

    rng = random.Random(42)
    running_tokens = 0
    rows = []

    for row in tickets.itertuples(index=False):
        ticket_id = row.ticket_id
        customer_id = row.customer_id
        original_text = row.text
        redacted_text = row.redacted_text

        category = ""
        ticket_tokens = 0

        # Input guardrails must happen before any model call.
        if not str(original_text).strip():
            status = "skipped_empty"

        elif mock_llm.est_tokens(original_text) > MAX_INPUT_TOKENS:
            status = "flagged_long"

        else:
            # The user message must contain exactly the redacted text.
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": redacted_text},
            ]

            (
                response,
                running_tokens,
                ticket_tokens,
                call_status,
            ) = call_with_retries(
                messages=messages,
                ticket_id=ticket_id,
                running_tokens=running_tokens,
                token_budget=token_budget,
                rng=rng,
            )

            if call_status == "budget_exhausted":
                status = "budget_exhausted"

            elif call_status == "error_permanent":
                status = "error_permanent"

            else:
                category, status = validate_response(
                    response.choices[0].message.content
                )

        rows.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "true_category": row.true_category,
            "tokens_used": ticket_tokens,
        })

    result = pd.DataFrame(rows)

    classified_statuses = {
        "classified_json",
        "classified_keyword",
    }
    classified = result[
        result["status"].isin(classified_statuses)
    ]

    accuracy = None

    if not classified.empty:
        accuracy = float(
            accuracy_score(
                classified["true_category"],
                classified["category"],
            )
        )

    total_tokens = int(result["tokens_used"].sum())

    metrics = {
        "run_name": run_name,
        "token_budget": token_budget,
        "tickets_total": int(len(result)),
        "tickets_classified": int(len(classified)),
        "accuracy_on_classified": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_classified_ticket": (
            round(total_tokens / len(classified), 2)
            if len(classified)
            else None
        ),
        "status_counts": {
            str(key): int(value)
            for key, value in result["status"].value_counts().items()
        },
        "budget_exhausted_count": int(
            (result["status"] == "budget_exhausted").sum()
        ),
        "permanent_error_count": int(
            (result["status"] == "error_permanent").sum()
        ),
    }

    print(f"\n--- {run_name} ---")
    print(json.dumps(metrics, indent=2))

    return result, metrics


# ------------------------------------------------------------
# Part B segments and business routing
# ------------------------------------------------------------
def load_segments():
    segments = pd.read_csv(
        SEGMENTS_FILE,
        keep_default_na=False,
    )

    required = {
        "customer_id",
        "segment_name",
        "is_premium",
    }
    missing = required - set(segments.columns)

    if missing:
        raise ValueError(
            f"Missing segment columns: {sorted(missing)}"
        )

    segments = segments.drop_duplicates("customer_id").copy()

    def parse_bool(value):
        return str(value).strip().lower() in {
            "true", "1", "yes"
        }

    segments["is_premium_bool"] = segments["is_premium"].map(
        parse_bool
    )

    return segments.set_index("customer_id").to_dict("index")


def apply_routing(result, segment_lookup):
    human_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
        "error_permanent",
    }

    output_rows = []

    for row in result.itertuples(index=False):
        segment = segment_lookup.get(row.customer_id, {})
        segment_name = segment.get("segment_name", "")
        is_premium = segment.get("is_premium_bool", False)

        if row.status in human_statuses:
            route = "human_review"
            priority = "NORMAL"

        elif row.category == "fraud":
            route = "security_team"
            priority = "HIGH"

        elif is_premium:
            route = "premium_desk"
            priority = "HIGH"

        else:
            route = "support_queue"
            priority = "NORMAL"

        output_rows.append({
            "ticket_id": row.ticket_id,
            "customer_id": row.customer_id,
            "status": row.status,
            "category": row.category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": row.tokens_used,
        })

    return pd.DataFrame(
        output_rows,
        columns=[
            "ticket_id",
            "customer_id",
            "status",
            "category",
            "segment_name",
            "priority",
            "route",
            "tokens_used",
        ],
    )


# ------------------------------------------------------------
# Reports
# ------------------------------------------------------------
def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def save_zero_shot_confusion(result):
    classified = result[
        result["status"].isin({
            "classified_json",
            "classified_keyword",
        })
    ]

    matrix = confusion_matrix(
        classified["true_category"],
        classified["category"],
        labels=ALLOWED_CATEGORIES,
    )

    confusion = pd.DataFrame(
        matrix,
        index=ALLOWED_CATEGORIES,
        columns=ALLOWED_CATEGORIES,
    )

    confusion.index.name = "true_category"
    confusion.to_csv(
        OUTPUTS / "part_c_zero_shot_confusion.csv"
    )


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    tickets = load_tickets()
    tickets, redaction_counts = prepare_redacted_tickets(tickets)

    save_json(
        OUTPUTS / "part_c_redaction_report.json",
        redaction_counts,
    )

    print("\n--- PII redaction counts across all tickets ---")
    print(json.dumps(redaction_counts, indent=2))

    # Comparison runs: no practical token budget.
    zero_result, zero_metrics = run_pipeline(
        tickets,
        ZERO_SHOT_PROMPT,
        10**9,
        "zero_shot_comparison",
    )

    few_result, few_metrics = run_pipeline(
        tickets,
        FEW_SHOT_PROMPT,
        10**9,
        "few_shot_comparison",
    )

    save_zero_shot_confusion(zero_result)

    # Reference status counts for each complete comparison run.
    expected_statuses = {
        "skipped_empty": 4,
        "flagged_long": 4,
        "classified_json": 135,
        "classified_keyword": 12,
        "flagged_invalid_output": 9,
    }

    for run_name, result in [
        ("zero-shot", zero_result),
        ("few-shot", few_result),
    ]:
        counts = result["status"].value_counts().to_dict()

        for status, expected in expected_statuses.items():
            actual = counts.get(status, 0)

            assert actual == expected, (
                f"{run_name}: expected {expected} {status} "
                f"tickets, got {actual}"
            )

        assert len(result) == 164
        assert counts.get("error_permanent", 0) == 0

    # These assertions verify all 164 tickets were redacted as expected.
    expected_redaction_counts = {
        "email": 55,
        "phone": 91,
        "order_id": 132,
        "total": 278,
    }

    assert redaction_counts == expected_redaction_counts, (
        f"Unexpected redaction counts: {redaction_counts}"
    )

    # Zero-shot budget demonstration.
    budget_result, budget_metrics = run_pipeline(
        tickets,
        ZERO_SHOT_PROMPT,
        TOKEN_BUDGET_DEMO,
        "zero_shot_budget_demo",
    )

    budget_result.to_csv(
        OUTPUTS / "triage_budget_demo.csv",
        index=False,
    )

    assert budget_metrics["budget_exhausted_count"] > 0, (
        "No tickets exhausted the demo budget. "
        "Lower TOKEN_BUDGET_DEMO and rerun."
    )

    # Final few-shot run with segment-aware routing.
    final_result, final_metrics = run_pipeline(
        tickets,
        FEW_SHOT_PROMPT,
        TOKEN_BUDGET_FINAL,
        "few_shot_final",
    )

    segment_lookup = load_segments()
    routed = apply_routing(final_result, segment_lookup)

    routed.to_csv(
        OUTPUTS / "triage_results.csv",
        index=False,
    )

    save_json(
        OUTPUTS / "part_c_metrics.json",
        {
            "redaction": redaction_counts,
            "zero_shot": zero_metrics,
            "few_shot": few_metrics,
            "budget_demo": budget_metrics,
            "final_few_shot": final_metrics,
            "zero_shot_confusion_file":
                "outputs/part_c_zero_shot_confusion.csv",
        },
    )

    assert len(routed) == 164

    assert set(routed["route"]).issubset({
        "human_review",
        "security_team",
        "premium_desk",
        "support_queue",
    })

    print("\nPART C CHECKPOINTS PASSED")
    print("Created outputs/triage_results.csv")
    print("Created outputs/triage_budget_demo.csv")
    print("Created outputs/part_c_metrics.json")
    print("Created outputs/part_c_redaction_report.json")
    print("Created outputs/part_c_zero_shot_confusion.csv")


if __name__ == "__main__":
    main()
