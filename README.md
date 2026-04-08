---
title: Datagym Environment Server
emoji: 🎲
colorFrom: red
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# DataGym — Dataset Repair RL Environment

> **Meta × PyTorch × Hugging Face — OpenEnv AI Hackathon**

An OpenEnv-compliant reinforcement learning environment where an AI agent repairs corrupted datasets. The agent receives a dirty dataset and must apply a sequence of targeted fix actions to restore it to a clean ground-truth state — without over-correcting or destroying valid data.

**Framing for judges:** *"Dataset Repair Agent for LLM Training Pipelines"* — the environment an agent must master before it can be trusted to curate training data for LLMs.

---

## How It Works

### Environment Loop

```
POST /reset  →  agent receives dirty dataset + issues list
POST /step   →  agent applies one action → receives reward + updated observation
...repeat until done=True or max_steps reached...
```

### Reward Signal

Reward at each step = **ΔF1** (change in cell-level F1 score against the hidden ground truth).

| Outcome | Reward |
|---|---|
| Correct action on the right column | +0.05 to +0.25 |
| Useless action (already clean column) | 0.00 |
| Destructive action (e.g. drop_nulls on 20% of rows) | −0.05 to −0.60 |
| Unknown action / missing column | −0.05 |

### Observation Fields

| Field | Type | Description |
|---|---|---|
| `dataset_preview` | list | First 10 rows of the current (dirty) DataFrame |
| `schema_info` | dict | Column names, inferred dtypes, null counts, shape |
| `issues_detected` | list | Human-readable diagnostics with exact fix instructions |
| `actions_taken` | list | History of successful actions this episode |
| `step` / `max_steps` | int | Current step and budget |
| `f1_score` | float | Current F1 against ground truth (0.0–1.0) |
| `precision` | float | Cell-level precision |
| `recall` | float | Cell-level recall / schema recovery rate (task3) |

### Available Actions

| Action | Purpose |
|---|---|
| `drop_nulls` | Drop rows where column is null (blocked if > 15% of rows) |
| `fill_nulls` | Fill nulls with mean / mode / custom value |
| `cast_type` | Cast column to int / float / str / datetime |
| `normalize_values` | Regex/literal string replacement mapping |
| `rename_column` | Rename a column |
| `deduplicate` | Drop duplicate rows |
| `drop_column` | Remove a spurious column (task3) |
| `split_column` | Split one column into two (task3) |
| `parse_json_column` | Extract a key from a JSON string column (task3) |
| `arithmetic_transform` | Scale a numeric column and optionally rename it (task3) |
| `submit` | End the episode |

---

## Tasks

### Task 1 — Easy (`task1_easy`)

**Dataset:** 200-row employee table (`employee_id`, `department`, `salary`, `performance_score`)

**Corruptions:**
- 15% nulls in `department`
- 10% of `salary` values contain a `" USD"` suffix (mixed types)
- 15 duplicate rows injected

**Optimal sequence (4 steps):**
```
normalize_values salary → cast_type salary int → fill_nulls department mode → deduplicate
```
**F1 ceiling:** ~0.94

---

### Task 2 — Medium (`task2_medium`)

**Dataset:** 300-row transaction table (`transaction_id`, `transaction_date`, `amount`, `category`)

**Corruptions:**
- Mixed date formats: DD/MM/YYYY, MM-DD-YY, Unix timestamps
- 20% of `amount` values prefixed with `$`
- Inconsistent category casing + trailing spaces
- 5 extreme outlier rows (`amount = 999999.99`)
- 8% nulls across three columns

**Optimal sequence (8 steps):**
```
normalize_values amount ($) → cast_type transaction_date datetime →
normalize_values category (spaces + casing) → normalize_values amount (outliers) →
cast_type amount float → fill_nulls amount mean → fill_nulls category mode →
fill_nulls transaction_date mode → submit
```
**F1 ceiling:** ~0.89 (null fills with mode/mean can't match individual GT values)

---

### Task 3 — Hard (`task3_hard`)

**Dataset:** 400-row product catalogue — ground truth has 6 columns, dirty version has schema corruption

**Corruptions:**
- `first_name` + `last_name` merged into `full_name`
- `price_usd` scaled ×1.3 and renamed `cost`
- `tag_1` + `tag_2` JSONified into `tags`
- Two spurious ETL columns injected (`etl_timestamp`, `_internal_row_uuid`)
- Column order shuffled

**Optimal sequence (8 steps):**
```
drop_column etl_timestamp → drop_column _internal_row_uuid →
split_column full_name → arithmetic_transform cost /1.3 →
parse_json_column tags t1 → parse_json_column tags t2 →
drop_column tags → submit
```
**F1 ceiling:** 0.90 (scorer: 0.45 × schema_recall + 0.45 × cell_f1)

---

## Testing in the Web UI

The OpenEnv web UI at the top of this Space lets you manually test each endpoint.

### 1. Reset the environment

Click **POST /reset**. This initializes a fresh session and generates a unique dirty dataset.

```json
{
  "task_id": "task1_easy",
  "seed": 42
}
```

Valid `task_id` values: `"task1_easy"`, `"task2_medium"`, `"task3_hard"`

You'll receive the first observation with `issues_detected` listing all corruptions found.

### 2. Execute a step

Click **POST /step** and send an action. The `params` field must always be a valid JSON object — not a string.

**Fill nulls with mode:**
```json
{
  "action_type": "fill_nulls",
  "column": "department",
  "params": {"strategy": "mode"}
}
```

**Cast salary to integer:**
```json
{
  "action_type": "cast_type",
  "column": "salary",
  "params": {"target_type": "int"}
}
```

**Normalize (strip currency symbol):**
```json
{
  "action_type": "normalize_values",
  "column": "amount",
  "params": {"mapping": {"\\$": ""}}
}
```

**Deduplicate:**
```json
{
  "action_type": "deduplicate",
  "column": "",
  "params": {}
}
```

**Drop a spurious column (task3):**
```json
{
  "action_type": "drop_column",
  "column": "etl_timestamp",
  "params": {}
}
```

**Split a merged column (task3):**
```json
{
  "action_type": "split_column",
  "column": "full_name",
  "params": {
    "delimiter": " ",
    "new_columns": ["first_name", "last_name"],
    "drop_original": true
  }
}
```

> **⚠️ UI note:** If you see a `422 Unprocessable Entity` error, check that `params` is a JSON object (`{}`), not a string or left blank.

### 3. Check current state

Click **GET /state** — returns `episode_id`, `step_count`, `task_id`, `current_f1`, `is_terminated`. Does not modify the dataset.

### 4. Get schemas

Click **GET /schema** — returns full JSON schemas for `DatagymAction`, `DatagymObservation`, and `State`.

---

## Running the Inference Agent

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_hf_token"
export ENV_BASE_URL="https://JKrishhhna-DataGym.hf.space"

python inference.py
```

**Expected stdout format (panel specification):**
```
[START] task=task1_easy env=datagym model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=normalize_values(salary) reward=0.00 done=false error=null
[STEP] step=2 action=cast_type(salary) reward=0.22 done=false error=null
...
[END] success=true steps=6 score=0.943 rewards=0.00,0.22,...
```

---

## Sample Evaluation Output

Actual run of `Qwen/Qwen2.5-72B-Instruct` on this environment.

**Why Task 2 shows `0.00` rewards on early steps:** The DataGym grader is strict — even if the agent cleans text values in a column, it receives no reward until it corrects the data type. This prevents half-fixes and ensures the agent learns to complete the full pipeline.

```
[START] task=task1_easy env=datagym model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=deduplicate reward=0.03 done=false error=null
[STEP] step=2 action=normalize_values(salary) reward=0.00 done=false error=null
[STEP] step=3 action=fill_nulls(department) reward=0.01 done=false error=null
[STEP] step=4 action=cast_type(salary) reward=0.03 done=false error=null
[END] success=true steps=5 score=0.968

[START] task=task2_medium env=datagym model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=normalize_values(amount) reward=0.00 done=false error=null
[STEP] step=2 action=normalize_values(amount) reward=0.00 done=false error=null
[STEP] step=3 action=cast_type(amount) reward=0.04 done=false error=null
[END] success=true steps=10 score=0.947
```

---

## Local Development

```bash
# Install dependencies
uv sync

# Run the server
uv run server
# or
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Smoke test
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_easy", "seed": 42}'
```

---

## Project Structure

```
/
├── inference.py                    # Agent inference script (panel requirement: root)
├── models.py                       # Pydantic models: DatagymAction, DatagymObservation, DatagymState
├── openenv.yaml                    # OpenEnv metadata and deployment config
├── Dockerfile                      # Multi-stage Docker build (openenv-base)
├── pyproject.toml                  # Package config and dependencies
└── server/
    ├── app.py                      # FastAPI app (create_app())
    ├── DataGym_environment.py      # Core RL environment with session store
    ├── generator.py                # Deterministic dataset generators + detect_issues()
    └── grader.py                   # Cell-level F1 scorer with datetime and type awareness
```

---

## Technical Highlights

**Session isolation:** All episode state lives in a class-level LRU store (`DatagymEnvironment._sessions`) keyed by `episode_id`. This survives framework instance recreation between requests and supports up to 10 concurrent judges without collision.

**Grader design:** Cell-level F1 with separate precision/recall denominators — extra duplicate rows lower precision (rewarding dedup), dropped rows lower recall (penalising `drop_nulls`). DateTime columns are compared date-only after normalisation so `cast_type(datetime)` gives real signal.

**Issues-as-instructions:** `detect_issues()` runs after every action and emits concrete fix instructions with exact mappings (e.g. `{"999999.99": "260.41"}`) rather than vague warnings. The agent never has to guess what to fix next.

**Reward every step:** Every action class produces non-zero signal — schema repairs increase `column_recall`, data fixes increase `cell_f1`, spurious column drops decrease the penalty term.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | LLM endpoint (e.g. `https://router.huggingface.co/v1`) |
| `HF_TOKEN` | Yes | Hugging Face API token |
| `MODEL_NAME` | Yes | Model ID (e.g. `Qwen/Qwen2.5-72B-Instruct`) |
| `ENV_BASE_URL` | No | Environment server URL (default: `http://localhost:8000`) |
