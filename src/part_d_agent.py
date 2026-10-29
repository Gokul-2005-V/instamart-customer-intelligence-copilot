"""Part D: offline ReAct refund agent with guardrails and resumable checkpoints."""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mock_llm
from src.part_d_rag import PolicyRetriever

DB_FILE = ROOT / "data" / "instamart.db"
SCENARIOS_FILE = ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order in the local SQLite database.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Retrieve relevant policy chunks.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Calculate the eligible refund amount.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": ["number", "null"]}}, "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Record an approved automatic refund of INR 1000 or less.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Record a refund above INR 1000 for human review.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}},
]

_RETRIEVER = None


def get_order(order_id: str) -> dict:
    """Read an order from SQLite."""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=? LIMIT 1",
                (str(order_id),),
            ).fetchone()
    except sqlite3.Error as exc:
        return {"error": "database error", "detail": str(exc)}
    if row is None:
        return {"error": "order not found"}
    return {
        "order_id": str(row["order_id"]),
        "status": str(row["status"]).strip().lower(),
        "order_amount": None if row["order_amount"] is None else float(row["order_amount"]),
        "delivery_minutes": None if row["delivery_minutes"] is None else float(row["delivery_minutes"]),
    }


def search_policy(query: str) -> dict:
    """Return retrieved policy chunks under the required results key."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = PolicyRetriever(overlap=10)
    return {"results": [
        {"source": item["source"], "score": round(float(item["score"]), 6), "text": item["text"]}
        for item in _RETRIEVER.retrieve(str(query), k=3)
    ]}


def calculate_refund(order_amount: float, delivery_minutes: float | None) -> dict:
    """Completed-order eligibility is checked by the agent; late delivery means >45 minutes."""
    try:
        amount = max(0.0, float(order_amount))
        minutes = None if delivery_minutes is None else float(delivery_minutes)
    except (TypeError, ValueError):
        return {"refund_inr": 0.0}
    return {"refund_inr": round(amount if minutes is not None and minutes > 45 else 0.0, 2)}


def _ledger_action(tool_name: str, order_id: str, refund_inr: float) -> dict:
    """Append a ledger record, avoiding an identical duplicate side effect."""
    amount = round(float(refund_inr), 2)
    if amount <= 0:
        return {"error": "refund amount must be positive", "refund_inr": amount}
    ledger = OUTPUT_DIR / "refund_ledger.jsonl"
    if ledger.exists():
        try:
            for line in ledger.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                old = json.loads(line)
                if (old.get("order_id") == str(order_id)
                    and old.get("tool_name") == tool_name
                    and round(float(old.get("refund_inr", -1)), 2) == amount):
                    return {"refund_inr": amount, "ledger_status": "already_recorded", "tool_name": tool_name}
        except (OSError, ValueError, TypeError):
            pass
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": str(order_id),
        "refund_inr": amount,
    }
    with ledger.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry) + "\n")
    return {"refund_inr": amount, "ledger_status": "recorded", "tool_name": tool_name}


def issue_refund(order_id: str, refund_inr: float) -> dict:
    if float(refund_inr) > 1000:
        return {"error": "refund above INR 1000 requires human escalation"}
    return _ledger_action("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id: str, refund_inr: float) -> dict:
    if float(refund_inr) <= 1000:
        return {"error": "human escalation is reserved for refunds above INR 1000"}
    return _ledger_action("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


class SimulatedCrash(RuntimeError):
    """Raised only to test checkpoint recovery."""


def _checkpoint_path(run_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(run_id))
    return STATE_DIR / f"{safe}.json"


def _save_checkpoint(run_id, scenario_id, messages, tokens_used, llm_calls, steps, tools_executed):
    payload = {
        "run_id": run_id, "scenario_id": scenario_id, "messages": messages,
        "tokens_used": tokens_used, "llm_calls": llm_calls,
        "step_number": len(steps), "steps": steps, "tools_executed": tools_executed,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _checkpoint_path(run_id)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def _load_checkpoint(run_id):
    path = _checkpoint_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found for run_id={run_id!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def _fallback(status, steps, llm_calls, tokens_used):
    done = ", ".join(step["tool"] for step in steps) or "no tools completed"
    return {
        "status": status,
        "final_answer": f"{status}: processing stopped safely. Completed tools: {done}. Please hand this case to a human support agent.",
        "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
        "tools_executed": [step["tool"] for step in steps],
    }


def _parse_arguments(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Tool arguments must be a JSON object")


def _continue_run(run_id, scenario_id, messages, tokens_used, llm_calls, steps, tools_executed,
                  *, max_iterations, token_limit, time_limit_seconds, start_time,
                  crash_after_tool_executions=None):
    while True:
        if time.monotonic() - start_time >= time_limit_seconds:
            return _fallback("FALLBACK_TIMEOUT", steps, llm_calls, tokens_used)
        if llm_calls >= max_iterations:
            return _fallback("FALLBACK_MAX_ITERATIONS", steps, llm_calls, tokens_used)

        try:
            response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        except Exception as exc:
            response = None
            last_error = exc
            for attempt in range(3):
                if time.monotonic() - start_time >= time_limit_seconds:
                    return _fallback("FALLBACK_TIMEOUT", steps, llm_calls, tokens_used)
                time.sleep(0.05 * (attempt + 1))
                try:
                    response = mock_llm.chat(messages, tools=TOOLS_SPEC)
                    break
                except Exception as retry_exc:
                    last_error = retry_exc
            if response is None:
                return _fallback(
                    "FALLBACK_LLM_ERROR",
                    steps + [{"tool": "mock_llm.chat", "arguments": {}, "observation": str(last_error)}],
                    llm_calls, tokens_used,
                )

        llm_calls += 1
        try:
            tokens_used += int(response.usage.total_tokens)
        except (AttributeError, TypeError, ValueError):
            pass
        # Never execute a tool requested by a call that exceeds the token budget.
        if tokens_used > token_limit:
            return _fallback("FALLBACK_TOKEN_LIMIT", steps, llm_calls, tokens_used)

        message = response.choices[0].message
        tool_calls = list(getattr(message, "tool_calls", []) or [])
        if not tool_calls:
            final_answer = str(getattr(message, "content", "") or "").strip()
            status = final_answer.split(":", 1)[0].strip().upper() if ":" in final_answer else "COMPLETED"
            return {
                "status": status,
                "final_answer": final_answer or "FALLBACK_EMPTY_RESPONSE: human handoff required.",
                "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                "tools_executed": tools_executed,
            }

        calls = []
        for call in tool_calls:
            calls.append({
                "id": str(getattr(call, "id", f"call_{len(tools_executed) + 1}")),
                "name": str(getattr(call, "name", "")),
                "arguments": _parse_arguments(getattr(call, "arguments", {})),
            })
        messages.append({"role": "assistant", "content": None, "tool_calls": calls})

        for call in calls:
            name, args = call["name"], call["arguments"]
            try:
                observation = TOOL_MAP[name](**args) if name in TOOL_MAP else {"error": f"unknown tool: {name}"}
                if not isinstance(observation, dict):
                    observation = {"result": observation}
            except Exception as exc:
                observation = {"error": f"{type(exc).__name__}: {exc}"}
            messages.append({
                "role": "tool", "tool_call_id": call["id"], "name": name,
                "content": json.dumps(observation, ensure_ascii=False, default=str),
            })
            steps.append({"tool": name, "arguments": args, "observation": observation})
            tools_executed.append(name)
            # Checkpoint immediately after each tool observation.
            _save_checkpoint(run_id, scenario_id, messages, tokens_used, llm_calls, steps, tools_executed)
            if crash_after_tool_executions is not None and len(tools_executed) >= crash_after_tool_executions:
                raise SimulatedCrash(f"Simulated crash after tool {len(tools_executed)}; checkpoint saved.")


def run_agent(user_message: str, scenario_id: str, max_iterations: int = 6,
              token_limit: int = 6000, time_limit_seconds: float = 10,
              *, run_id: str | None = None, crash_after_tool_executions: int | None = None) -> dict:
    """Run a fresh agent workflow with iteration, token, and time guardrails."""
    run_id = run_id or scenario_id
    messages = [
        {"role": "system", "content": (
            "AGENT\nYou are a cautious Instamart refund agent. Follow policy exactly. "
            "Only completed orders can be refunded. Delivery over 45 minutes means 100% of order_amount; "
            "delivery in 45 minutes or less means zero refund. Refunds above INR 1000 must be escalated "
            "to a human and never issued automatically. Use tools and do not invent order facts."
        )},
        {"role": "user", "content": str(user_message)},
    ]
    mock_llm.reset()
    _save_checkpoint(run_id, scenario_id, messages, 0, 0, [], [])
    return _continue_run(
        run_id, scenario_id, messages, 0, 0, [], [],
        max_iterations=max_iterations, token_limit=token_limit,
        time_limit_seconds=float(time_limit_seconds), start_time=time.monotonic(),
        crash_after_tool_executions=crash_after_tool_executions,
    )


def resume_agent(run_id: str, max_iterations: int = 6, token_limit: int = 6000,
                 time_limit_seconds: float = 10) -> dict:
    """Continue from the last saved tool observation; do not replay completed tools."""
    state = _load_checkpoint(run_id)
    return _continue_run(
        run_id, state["scenario_id"], state["messages"], int(state["tokens_used"]),
        int(state["llm_calls"]), state["steps"], state["tools_executed"],
        max_iterations=max_iterations, token_limit=token_limit,
        time_limit_seconds=float(time_limit_seconds), start_time=time.monotonic(),
    )


def _refund_amount(result):
    for step in reversed(result.get("steps", [])):
        if step["tool"] == "calculate_refund" and "refund_inr" in step["observation"]:
            return float(step["observation"]["refund_inr"])
    return None


def run_scenarios():
    rows = pd.read_csv(SCENARIOS_FILE, keep_default_na=False).to_dict(orient="records")
    results = []
    with (OUTPUT_DIR / "agent_runs.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            result = run_agent(row["message"], row["scenario_id"], run_id=f"scenario-{row['scenario_id']}")
            record = {
                "scenario_id": row["scenario_id"], "status": result["status"],
                "final_answer": result["final_answer"], "refund_inr": _refund_amount(result),
                "llm_calls": result["llm_calls"], "tokens_used": result["tokens_used"],
                "steps": result["steps"], "tools_executed": result["tools_executed"],
                "expected_outcome": row["expected_outcome"],
                "expected_refund_inr": float(row["expected_refund_inr"]),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)
    return results


def run_safety_demos():
    rows = pd.read_csv(SCENARIOS_FILE, keep_default_na=False)
    s1 = rows.loc[rows["scenario_id"] == "S1"].iloc[0]
    token = run_agent(s1["message"], "S1_TOKEN_LIMIT", token_limit=200, run_id="demo-token-limit")
    timeout = run_agent(s1["message"], "S1_TIMEOUT", time_limit_seconds=0, run_id="demo-timeout")
    data = {
        "token_limit_demo": {key: token[key] for key in ("status", "llm_calls", "tokens_used", "final_answer")},
        "timeout_demo": {key: timeout[key] for key in ("status", "llm_calls", "tokens_used", "final_answer")},
    }
    (OUTPUT_DIR / "part_d_agent_safety_demos.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def run_resume_demo():
    rows = pd.read_csv(SCENARIOS_FILE, keep_default_na=False)
    s1 = rows.loc[rows["scenario_id"] == "S1"].iloc[0]
    run_id = "demo-resume-S1"
    checkpoint = _checkpoint_path(run_id)
    if checkpoint.exists():
        checkpoint.unlink()
    try:
        run_agent(s1["message"], "S1_RESUME_DEMO", run_id=run_id, crash_after_tool_executions=2)
    except SimulatedCrash:
        pass
    before = _load_checkpoint(run_id)["tools_executed"]
    result = resume_agent(run_id)
    data = {
        "status": result["status"], "final_answer": result["final_answer"],
        "llm_calls": result["llm_calls"], "tokens_used": result["tokens_used"],
        "tools_before_resume": before, "tools_across_both_runs": result["tools_executed"],
        "refund_inr": _refund_amount(result),
    }
    (OUTPUT_DIR / "part_d_agent_resume_demo.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def validate_scenarios(results):
    expected = {"S1": "REFUND", "S2": "REFUND", "S3": "ESCALATE", "S4": "DENIED",
                "S5": "DENIED", "S6": "NOT_FOUND", "S7": "FALLBACK_MAX_ITERATIONS"}
    failures = []
    for row in results:
        sid = row["scenario_id"]
        if row["status"] != expected[sid]:
            failures.append(f"{sid}: expected {expected[sid]}, got {row['status']}")
        if sid in ("S1", "S2", "S3"):
            expected_amount = row["expected_refund_inr"]
            actual = row["refund_inr"]
            if actual is None or abs(actual - expected_amount) > 0.01:
                failures.append(f"{sid}: expected refund {expected_amount:.2f}, got {actual}")
        if sid == "S1" and row["llm_calls"] != 5:
            failures.append(f"S1: expected 5 model calls, got {row['llm_calls']}")
    return failures


def main():
    results = run_scenarios()
    print("\nPART D REFUND AGENT SCENARIOS")
    print("ID       STATUS                   REFUND_INR  CALLS  TOKENS")
    for row in results:
        amount = "—" if row["refund_inr"] is None else f"{row['refund_inr']:.2f}"
        print(f"{row['scenario_id']:<8} {row['status']:<24} {amount:>10} {row['llm_calls']:>6} {row['tokens_used']:>7}")
    safety = run_safety_demos()
    resume = run_resume_demo()
    print("\nSAFETY DEMOS\n" + json.dumps(safety, indent=2))
    print("\nRESUME DEMO\n" + json.dumps(resume, indent=2))
    failures = validate_scenarios(results)
    if safety["token_limit_demo"]["status"] != "FALLBACK_TOKEN_LIMIT":
        failures.append("token-limit demo did not return FALLBACK_TOKEN_LIMIT")
    if safety["timeout_demo"]["status"] != "FALLBACK_TIMEOUT":
        failures.append("timeout demo did not return FALLBACK_TIMEOUT")
    if resume["status"] != "REFUND" or abs(float(resume["refund_inr"] or 0) - 128.25) > 0.01:
        failures.append("resume demo did not complete the S1 refund")
    if failures:
        print("\nFAILURES\n" + "\n".join("- " + item for item in failures))
        return 1
    print("\nALL PART D REFUND AGENT CHECKPOINTS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
