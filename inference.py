"""
inference.py — DataGym OpenEnv inference client.

Mandatory stdout format (panel specification):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Required environment variables:
    API_BASE_URL   — LLM endpoint base URL
    MODEL_NAME     — Model identifier
    HF_TOKEN       — Hugging Face / API key
    ENV_BASE_URL   — FastAPI env server (default: http://localhost:8000)
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Environment variables ─────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY: str      = os.getenv("HF_TOKEN")
MODEL_NAME: str   = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:8000")

# ── Benchmark metadata ────────────────────────────────────────────────────────
BENCHMARK: str = "datagym"

# ── Per-task step budgets ─────────────────────────────────────────────────────
TASK_MAX_STEPS: Dict[str, int] = {
    "task1_easy":   10,
    "task2_medium": 12,
    "task3_hard":   15,
}

# ── LLM hyperparameters ───────────────────────────────────────────────────────
TEMPERATURE: float = 0.2
MAX_TOKENS:  int   = 512
SUCCESS_SCORE_THRESHOLD: float = 0.5

# ── Session state ─────────────────────────────────────────────────────────────
SESSION_ID: Optional[str] = None

# ── OpenAI-compatible client ──────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = r"""You are an Autonomous Data Quality Engineer.
Your job is to repair a corrupted dataset to match a hidden ground truth.
Read the observation carefully: dataset_preview, schema_info, issues_detected,
and current F1/Precision/Recall scores.  Choose ONE action per step.

Respond ONLY with a valid JSON object. No markdown, no extra text.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE ACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLEANING:
1. {"action_type": "drop_nulls",       "column": "col", "params": {}}
2. {"action_type": "fill_nulls",       "column": "col", "params": {"strategy": "mode"}}
   (strategy: "mode" | "mean" | omit to use "value":"x")
3. {"action_type": "cast_type",        "column": "col", "params": {"target_type": "int"}}
   (target_type: int | float | str | datetime)
4. {"action_type": "normalize_values", "column": "col", "params": {"mapping": {"PATTERN": "REPLACEMENT"}}}
   SAFE EXAMPLES:
     Remove " USD"         : {" USD": ""}
     Remove $ prefix       : {"\$": ""}    <- dollar MUST be \$ (regex anchor otherwise)
     Strip trailing spaces : {"   ": ""}   <- literal spaces, NOT \s+
     Fix casing            : {"electronics": "Electronics"}
5. {"action_type": "rename_column",    "column": "old", "params": {"new_name": "new"}}
6. {"action_type": "deduplicate",      "column": null,  "params": {}}

SCHEMA REPAIR (task3):
7.  {"action_type": "drop_column",         "column": "col",  "params": {}}
8.  {"action_type": "split_column",        "column": "full_name",
     "params": {"delimiter": " ", "new_columns": ["first_name", "last_name"], "drop_original": true}}
9.  {"action_type": "parse_json_column",   "column": "tags",
     "params": {"key": "t1", "new_column": "tag_1"}}
10. {"action_type": "arithmetic_transform","column": "cost",
     "params": {"operator": "/", "value": 1.3, "new_name": "price_usd"}}

TERMINAL:
11. {"action_type": "submit", "column": null, "params": {}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION POLICY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. READ issues_detected first — every step. Do not guess.

2. ACTION ORDER — follow exactly:
   a. Schema first (task3): drop spurious cols, reconstruct missing ones.
   b. Replace outliers BEFORE fill_nulls — outliers inflate the mean.
      Use normalize_values with the exact mapping from issues_detected.
   c. Normalize strings BEFORE cast. Strip "$" or " USD", then cast.
   d. Fill nulls AFTER casting and outlier removal (clean data = better mean).
   e. Deduplicate ONLY if issues_detected mentions duplicate rows.
   f. Submit when issues_detected is empty or F1 > 0.90.

3. DOLLAR SIGN: use {"\$": ""} NOT {"$": ""}. Bare $ is a regex anchor.

4. DATES: use cast_type datetime on transaction_date directly — it handles
   DD/MM/YYYY, MM-DD-YY, and Unix timestamps. Do not normalize dates with regex.

5. drop_nulls is BLOCKED if it removes more than 15 percent of rows.
   Use fill_nulls to preserve row count.

6. If the same column gave reward 0.000 twice in a row, move on.

TASK 1 sequence:
  fill_nulls department mode -> normalize_values salary {" USD": ""} ->
  cast_type salary int -> fill_nulls salary mode -> deduplicate -> submit

TASK 2 sequence:
  normalize_values amount {"\$": ""} -> cast_type transaction_date datetime ->
  normalize_values category (spaces + all 6 casing variants in one call) ->
  normalize_values amount outliers (use exact mapping from issues_detected) ->
  cast_type amount float -> fill_nulls amount mean ->
  fill_nulls category mode -> fill_nulls transaction_date mode -> submit

TASK 3 sequence:
  drop_column etl_timestamp -> drop_column _internal_row_uuid ->
  split_column full_name -> arithmetic_transform cost /1.3 ->
  parse_json_column tags key=t1 new_column=tag_1 ->
  parse_json_column tags key=t2 new_column=tag_2 ->
  drop_column tags -> submit
"""


# ── Mandatory stdout loggers ──────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    # Keep short task name (easy/medium/hard) matching evaluator's expected format
    short = task.replace("task1_", "").replace("task2_", "").replace("task3_", "")
    print(f"[START] task={short} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    task: str,
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} score={score:.3f} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ── Observation unwrapper ─────────────────────────────────────────────────────

def _unwrap_observation(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    create_app double-wraps the observation:
        data["observation"]["observation"] = real DatagymObservation dict
    Detect by checking for our sentinel field f1_score.
    """
    outer = data.get("observation", data)
    if isinstance(outer, dict) and "observation" in outer and "f1_score" not in outer:
        return outer["observation"]
    return outer


# ── API helpers ───────────────────────────────────────────────────────────────

def call_reset(task_id: str = "task1_easy", seed: int = 42) -> Dict[str, Any]:
    """POST /reset → returns unwrapped observation dict."""
    global SESSION_ID

    episode_id = f"eval-{task_id}-{int(time.time())}"
    SESSION_ID  = episode_id

    payload = {"task_id": task_id, "seed": seed, "episode_id": episode_id}
    response = requests.post(f"{ENV_BASE_URL}/reset", json=payload, timeout=30)
    response.raise_for_status()

    data = response.json()
    return _unwrap_observation(data)


def call_step(action_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /step → returns normalised dict:
        {"observation": {...}, "reward": float, "done": bool}
    """
    clean_action = {k: v for k, v in action_dict.items() if v is not None}

    # Thread session key via action.metadata for concurrent judge isolation
    if SESSION_ID:
        meta = clean_action.get("metadata") or {}
        meta["session_id"] = SESSION_ID
        clean_action["metadata"] = meta

    payload: Dict[str, Any] = {"action": clean_action}
    if SESSION_ID:
        payload["session_id"] = SESSION_ID

    response = requests.post(f"{ENV_BASE_URL}/step", json=payload, timeout=30)
    if response.status_code != 200:
        response.raise_for_status()

    data = response.json()
    obs  = _unwrap_observation(data)

    return {
        "observation": obs,
        "reward":      data.get("reward", 0.0),
        "done":        data.get("done", False),
    }


# ── Agent ─────────────────────────────────────────────────────────────────────

def get_agent_action(
    observation:    Dict[str, Any],
    recent_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Call the LLM and return a parsed action dict."""
    prompt = (
        f"CURRENT OBSERVATION:\n{json.dumps(observation, indent=2)}\n\n"
        f"RECENT ACTION HISTORY (last 4):\n{json.dumps(recent_history, indent=2)}\n\n"
        "Decide the best next action. If the dataset is clean, submit."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return {"action_type": "submit", "column": None, "params": {}}


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_id: str, seed: int = 42) -> None:
    """
    Run one full RL episode, emitting mandatory [START] / [STEP] / [END] lines.
    [END] is always emitted, even if an exception occurs mid-episode.
    """
    max_steps = TASK_MAX_STEPS.get(task_id, 10)

    rewards:      List[float]         = []
    recent_history: List[Dict]        = []
    steps_taken:  int                 = 0
    score:        float               = 0.0
    success:      bool                = False
    observation:  Optional[Dict]      = None
    done:         bool                = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = call_reset(task_id=task_id, seed=seed)

        for step in range(1, max_steps + 1):
            if done:
                break

            action = get_agent_action(observation, recent_history)

            # Compact action string for [STEP] log
            act_type = action.get("action_type", "unknown")
            act_col  = action.get("column") or ""
            act_str  = f"{act_type}({act_col})" if act_col else act_type

            # Extract last error from issues for [STEP] error field
            last_error: Optional[str] = None
            issues = observation.get("issues_detected", [])
            failed = [i for i in issues if "FAILED" in i]
            if failed:
                last_error = failed[-1][:120].replace("\n", " ")

            try:
                step_response = call_step(action)
                observation   = step_response["observation"]
                reward_val    = float(step_response.get("reward") or 0.0)
                done          = bool(step_response.get("done", False))

                # Error = any new FAILED issue in this step's observation
                new_issues  = observation.get("issues_detected", [])
                new_failed  = [i for i in new_issues if "FAILED" in i]
                step_error  = new_failed[-1][:120].replace("\n", " ") if new_failed else None

            except Exception as exc:
                reward_val  = 0.0
                done        = True
                step_error  = str(exc)[:120]
                print(f"[DEBUG] Step {step} error: {exc}", flush=True)

            rewards.append(reward_val)
            steps_taken = step

            log_step(
                step=step,
                action=act_str,
                reward=reward_val,
                done=done,
                error=step_error,
            )

            # Rolling 4-action history for LLM context
            recent_history.append(action)
            if len(recent_history) > 4:
                recent_history.pop(0)

        # Score = final F1 (already in [0, 1])
        if observation:
            score = float(observation.get("f1_score", 0.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(
            task=task_id,
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    missing = [v for v in ("API_BASE_URL", "HF_TOKEN", "MODEL_NAME") if not os.getenv(v)]
    if missing:
        print(f"[DEBUG] Missing env vars: {', '.join(missing)}", flush=True)

    # Run one episode per task — demonstrates full environment range.
    # Seed is fixed for reproducibility.
    tasks = [
        ("task1_easy",   42),
        ("task2_medium", 42),
        ("task3_hard",   42),
    ]

    try:
        for task_id, seed in tasks:
            run_episode(task_id, seed=seed)
    except requests.exceptions.ConnectionError:
        print(
            f"[DEBUG] Cannot connect to env server at {ENV_BASE_URL}. "
            "Is the FastAPI server running?",
            flush=True,
        )