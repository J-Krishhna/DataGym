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



# 🏋️ DataGym — Dataset Repair RL Environment

> **Meta × PyTorch × Hugging Face — OpenEnv AI Hackathon**

An **OpenEnv-compliant reinforcement learning environment** where an AI agent repairs corrupted datasets. The agent receives a dirty dataset and must apply a sequence of targeted fix actions to restore it to a clean ground-truth state — without over-correcting or destroying valid data.

**Why this matters:** Bad training data is one of the primary causes of LLM hallucination. Before any model can be trusted to curate its own training data, it must master data quality. DataGym is the environment where that mastery is tested.

---

## 📊 Agent Performance (Verified Results)

```
[START] task=easy env=datagym model=llama-3.3-70b-versatile
[STEP] step=1 action=fill_nulls(department)      reward=0.01 done=false error=null
[STEP] step=2 action=normalize_values(salary)    reward=0.00 done=false error=null
[STEP] step=3 action=cast_type(salary)           reward=0.00 done=false error=null
[STEP] step=4 action=fill_nulls(salary)          reward=0.02 done=false error=null
[STEP] step=5 action=deduplicate                 reward=0.04 done=false error=null
[STEP] step=6 action=submit                      reward=0.00 done=true  error=null
[END] success=true steps=6 rewards=0.01,0.00,0.00,0.02,0.04,0.00

[START] task=medium env=datagym model=llama-3.3-70b-versatile
[STEP] step=1 action=normalize_values(amount)         reward=0.00 done=false error=null
[STEP] step=2 action=normalize_values(amount)         reward=0.00 done=false error=null
[STEP] step=3 action=cast_type(transaction_date)      reward=0.23 done=false error=null
[STEP] step=4 action=normalize_values(category)       reward=0.17 done=false error=null
[STEP] step=5 action=cast_type(amount)                reward=0.04 done=false error=null
[STEP] step=6 action=fill_nulls(amount)               reward=0.02 done=false error=null
[STEP] step=7 action=normalize_values(amount)         reward=0.00 done=false error=null
[STEP] step=8 action=fill_nulls(transaction_date)     reward=0.00 done=false error=null
[STEP] step=9 action=fill_nulls(category)             reward=0.01 done=false error=null
[STEP] step=10 action=submit                          reward=0.00 done=true  error=null
[END] success=true steps=10 rewards=0.00,0.00,0.23,0.17,0.04,0.02,0.00,0.00,0.01,0.00

[START] task=hard env=datagym model=llama-3.3-70b-versatile
[STEP] step=1 action=drop_column(etl_timestamp)       reward=0.05 done=false error=null
[STEP] step=2 action=drop_column(_internal_row_uuid)  reward=0.05 done=false error=null
[STEP] step=3 action=split_column(full_name)          reward=0.20 done=false error=null
[STEP] step=4 action=arithmetic_transform(cost)       reward=0.13 done=false error=null
[STEP] step=5 action=parse_json_column(tags)          reward=0.08 done=false error=null
[STEP] step=6 action=parse_json_column(tags)          reward=0.08 done=false error=null
[STEP] step=7 action=drop_column(tags)                reward=0.05 done=false error=null
[STEP] step=8 action=submit                           reward=0.00 done=true  error=null
[END] success=true steps=8 rewards=0.05,0.05,0.20,0.13,0.08,0.08,0.05,0.00
```

| Task | Final F1 | Steps | All Actions Correct |
|---|---|---|---|
| easy | **0.968** | 6 | ✅ |
| medium | **0.967** | 10 | ✅ |
| hard | **0.900** | 8 | ✅ |

---

## 🧠 What the Agent Does — Step by Step

### Task 1 (easy) — Employee Dataset Repair

The dataset has 15% null departments, 10% salary values with `" USD"` suffixes, and 15 duplicate rows.

| Step | Action | Why |
|---|---|---|
| 1 | `fill_nulls(department, mode)` | Fills nulls with most common department |
| 2 | `normalize_values(salary, {" USD": ""})` | Strips the string suffix so salary can be cast |
| 3 | `cast_type(salary, int)` | Converts cleaned salary strings to integers |
| 4 | `fill_nulls(salary, mode)` | Fills NaNs created by cast coercion |
| 5 | `deduplicate` | Removes 15 injected duplicate rows |
| 6 | `submit` | Episode complete — F1 = 0.968 |

### Task 2 (medium) — Transaction Dataset Repair

The dataset has mixed date formats, currency symbols, inconsistent category casing, outliers, and 8% nulls across three columns.

| Step | Action | Why |
|---|---|---|
| 1-2 | `normalize_values(amount)` | Strips `$` prefix using `{"\$": ""}` |
| 3 | `cast_type(transaction_date, datetime)` | Resolves DD/MM/YYYY, MM-DD-YY, Unix timestamps in one pass — +0.23 reward |
| 4 | `normalize_values(category)` | Strips trailing spaces and fixes all casing variants — +0.17 reward |
| 5 | `cast_type(amount, float)` | Converts clean amount strings to floats |
| 6 | `fill_nulls(amount, mean)` | IQR-protected mean excludes outliers |
| 7-9 | `fill_nulls(...)` | Fills remaining nulls in date and category |
| 10 | `submit` | F1 = 0.967 |

### Task 3 (hard) — Schema Reconstruction

The dataset has structural corruption: merged columns, scaled values, JSONified tags, and spurious ETL columns.

| Step | Action | Why |
|---|---|---|
| 1 | `drop_column(etl_timestamp)` | Removes spurious ETL artifact — schema recall +1 col |
| 2 | `drop_column(_internal_row_uuid)` | Removes second ETL artifact |
| 3 | `split_column(full_name)` | Reconstructs `first_name` + `last_name` — +0.20 reward |
| 4 | `arithmetic_transform(cost, /1.3)` | Reverses currency scaling → `price_usd` — +0.13 reward |
| 5 | `parse_json_column(tags, t1→tag_1)` | Extracts `tag_1` from JSON blob |
| 6 | `parse_json_column(tags, t2→tag_2)` | Extracts `tag_2` from JSON blob |
| 7 | `drop_column(tags)` | Removes now-extracted JSON column |
| 8 | `submit` | F1 = 0.900 (theoretical ceiling for this task) |

---

## 🔁 Environment Loop

```
POST /reset  →  agent receives dirty dataset + issues list
     ↓
POST /step   →  agent applies one action
     ↓
observation: {dataset_preview, schema_info, issues_detected, f1_score, ...}
reward: ΔF1 (change in cell-level F1 vs hidden ground truth)
     ↓
repeat until done=True or max_steps reached
```

### Reward Design

- **Positive reward** — any action that improves F1 against ground truth
- **Zero reward** — valid action on already-clean data (no harm, no gain)
- **Negative reward** — destructive actions: `drop_nulls` on > 15% of rows (−0.05 to −0.60), unknown action (−0.05)
- **Partial progress** — every step that moves toward the goal gets signal, not just the final submit

---

## 🧪 Testing via Swagger UI

Open the Space and click the **API** tab to access the Swagger interface.

### Step 1 — Reset the environment

**Endpoint:** `POST /reset`

```json
{
  "task_id": "task1_easy",
  "seed": 42
}
```

Valid task IDs: `"task1_easy"` · `"task2_medium"` · `"task3_hard"`

**What you get back:** First observation with `issues_detected` listing every corruption found, `dataset_preview` showing the first 10 rows, and starting `f1_score`.

---

### Step 2 — Execute actions

**Endpoint:** `POST /step`

The `action` field must be a JSON object — **not a string**.

#### Fill nulls with mode
```json
{
  "action": {
    "action_type": "fill_nulls",
    "column": "department",
    "params": {"strategy": "mode"}
  }
}
```

#### Strip currency prefix from amount
```json
{
  "action": {
    "action_type": "normalize_values",
    "column": "amount",
    "params": {"mapping": {"\\$": ""}}
  }
}
```

#### Cast transaction_date to datetime (handles all formats automatically)
```json
{
  "action": {
    "action_type": "cast_type",
    "column": "transaction_date",
    "params": {"target_type": "datetime"}
  }
}
```

#### Fix category casing (all variants in one call)
```json
{
  "action": {
    "action_type": "normalize_values",
    "column": "category",
    "params": {
      "mapping": {
        "   ": "",
        "electronics": "Electronics",
        "ELECTRONICS": "Electronics",
        "clothing": "Clothing",
        "CLOTHING": "Clothing",
        "food": "Food",
        "FOOD": "Food"
      }
    }
  }
}
```

#### Split full_name into first and last (task3)
```json
{
  "action": {
    "action_type": "split_column",
    "column": "full_name",
    "params": {
      "delimiter": " ",
      "new_columns": ["first_name", "last_name"],
      "drop_original": true
    }
  }
}
```

#### Reverse price scaling (task3)
```json
{
  "action": {
    "action_type": "arithmetic_transform",
    "column": "cost",
    "params": {"operator": "/", "value": 1.3, "new_name": "price_usd"}
  }
}
```

#### Submit (end episode)
```json
{
  "action": {
    "action_type": "submit",
    "column": null,
    "params": {}
  }
}
```

---

### Step 3 — Check current state

**Endpoint:** `GET /state`

Returns `episode_id`, `step_count`, `task_id`, `current_f1`, `is_terminated`.

### Step 4 — Get schemas

**Endpoint:** `GET /schema`

Returns full JSON schemas for `DatagymAction`, `DatagymObservation`, and `State`.

---

## 🚀 Running the Inference Agent

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your_hf_token"
export ENV_BASE_URL="https://JKrishhhna-DataGym.hf.space"

python inference.py
```

Expected output format (panel specification):
```
[START] task=<name> env=datagym model=<model>
[STEP] step=<n> action=<action> reward=<0.00> done=<true|false> error=<null|msg>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...>
```

---

## 🏗️ Project Structure

```
/
├── inference.py              # Agent loop — emits mandatory stdout format
├── models.py                 # Pydantic models: Action, Observation, State
├── openenv.yaml              # OpenEnv metadata and deployment config
├── Dockerfile                # Multi-stage build using openenv-base
├── pyproject.toml            # Package config (uv)
└── server/
    ├── app.py                # FastAPI app via create_app()
    ├── DataGym_environment.py # Core RL environment with session store
    ├── generator.py          # Dataset generators + detect_issues()
    └── grader.py             # Cell-level F1 scorer
```

---

## ⚙️ Local Development

```bash
# Install
uv sync

# Run server
uv run server
# or
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Test reset
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_easy", "seed": 42}'

# Validate
openenv validate
```

---

## 🔧 Technical Design

**Session isolation:** Episode state lives in a class-level LRU store keyed by `episode_id`. Survives framework instance recreation between requests. Supports 10 concurrent sessions — judge runs never collide.

**Grader:** Cell-level F1 with datetime-aware comparison (timestamps compared date-only after normalisation), IQR-protected mean for null filling, and row-count-sensitive precision/recall (extra duplicate rows lower precision → dedup gives real reward signal).

**Issues-as-instructions:** `detect_issues()` runs after every action and emits specific, actionable fix instructions including exact replacement mappings. The agent never guesses — every corruption is described with the exact action needed to fix it.

**Task 3 scoring:** `0.45 × column_recall + 0.45 × cell_f1 − spurious_penalty`. Every action class produces non-zero reward — schema drops reduce penalty, column reconstructions increase recall, data fixes increase cell F1. Theoretical ceiling = 0.90.

---

## 🌍 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | LLM endpoint (e.g. `https://api.groq.com/openai/v1`) |
| `HF_TOKEN` | Yes | Hugging Face API token |
| `MODEL_NAME` | Yes | Model ID (e.g. `llama-3.3-70b-versatile`) |
| `ENV_BASE_URL` | No | Env server URL (default: `http://localhost:8000`) |

---

## 📋 Available Actions Reference

| Action | Required params | Tasks |
|---|---|---|
| `fill_nulls` | `column`, `strategy`: `mode`/`mean`/`value` | all |
| `cast_type` | `column`, `target_type`: `int`/`float`/`str`/`datetime` | all |
| `normalize_values` | `column`, `mapping`: `{pattern: replacement}` | all |
| `deduplicate` | none | all |
| `drop_nulls` | `column` | all (blocked if > 15% rows) |
| `rename_column` | `column`, `new_name` | all |
| `drop_column` | `column` | task3 |
| `split_column` | `column`, `delimiter`, `new_columns` | task3 |
| `parse_json_column` | `column`, `key`, `new_column` | task3 |
| `arithmetic_transform` | `column`, `operator`, `value`, `new_name` | task3 |
| `submit` | none | all |
