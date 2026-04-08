"""
DataGym RL Environment — OpenEnv compliant, session-safe implementation.

Session architecture
────────────────────
All mutable episode state lives in a class-level LRU store (_sessions) keyed
by a fixed __default__ key.  Instance methods are stateless shells that
delegate to _sessions, making the design robust against create_app recreating
instances between requests.

  close() is intentionally a no-op: openenv-core calls it after every HTTP
  request as a per-request teardown hook, not a server shutdown signal.

Endpoint contract (openenv-core 0.2.3):
  POST /reset  →  reset()      returns DatagymObservation
  POST /step   →  step()       returns StepResponse
  GET  /state  →  get_state()  returns DatagymState
"""

import json as json_lib
import math
import time
import threading
import logging
import traceback
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from openenv.core.env_server.types import StepResponse

sys.path.append(str(Path(__file__).parent.parent))
from models import DatagymAction, DatagymObservation, DatagymState
from .generator import load_task_data, detect_issues
from .grader import calculate_similarity

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("datagym.env")

# ── Task-aware configuration ──────────────────────────────────────────────────
TASK_MAX_STEPS: Dict[str, int] = {
    "task1_easy":   10,
    "task2_medium": 12,
    "task3_hard":   15,
}

# ── Session cap — must match max_concurrent_envs in app.py ───────────────────
_MAX_SESSIONS    = 10
_DEFAULT_SESSION = "__default__"


# ── Per-episode state container ───────────────────────────────────────────────

@dataclass
class _EpisodeState:
    """All mutable state for one RL episode. Stored at class level."""
    current_df:      pd.DataFrame
    ground_truth_df: pd.DataFrame
    task_id:         str                  = "task1_easy"
    episode_id:      Optional[str]        = None
    step_count:      int                  = 0
    max_steps:       int                  = 10
    history:         List[DatagymAction]  = field(default_factory=list)
    issues_detected: List[str]            = field(default_factory=list)
    current_metrics: Dict[str, float]     = field(
        default_factory=lambda: {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    )
    terminated:      bool                 = False
    session_key:     str                  = _DEFAULT_SESSION
    last_accessed:   float                = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_accessed = time.monotonic()


class DatagymEnvironment:
    """
    Dataset Repair RL environment.

    The agent receives a corrupted DataFrame and must apply a sequence of
    targeted fix actions to restore it to the hidden ground-truth state,
    maximising the cell-level F1 score.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    # ── Class-level LRU session store (survives instance recreation) ──────────
    _sessions: OrderedDict = OrderedDict()
    _lock: threading.RLock = threading.RLock()

    # ──────────────────────────────────────────────────────────────────────────
    # Session store helpers
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def _get_session(cls, sid: str) -> Optional[_EpisodeState]:
        with cls._lock:
            state = cls._sessions.get(sid)
            if state is not None:
                cls._sessions.move_to_end(sid)
                state.touch()
            return state

    @classmethod
    def _put_session(cls, sid: str, state: _EpisodeState) -> None:
        with cls._lock:
            if sid in cls._sessions:
                cls._sessions.move_to_end(sid)
            else:
                while len(cls._sessions) >= _MAX_SESSIONS:
                    evicted_id, evicted = cls._sessions.popitem(last=False)
                    log.info(
                        "[SESSION] Evicted %s (idle %.1fs)",
                        evicted_id, time.monotonic() - evicted.last_accessed,
                    )
            cls._sessions[sid] = state
            state.touch()

    @classmethod
    def _delete_session(cls, sid: str) -> None:
        with cls._lock:
            cls._sessions.pop(sid, None)

    # ──────────────────────────────────────────────────────────────────────────
    # Public OpenEnv interface
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        task_id:    str           = "task1_easy",
        seed:       int           = 42,
        episode_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> DatagymObservation:
        """
        POST /reset handler.

        Creates (or replaces) the session state, loads the task dataset,
        auto-detects initial issues, and returns the first observation.
        """
        # Generate a stable session key from episode_id when provided.
        # Falls back to __default__ for single-client use.
        # The key is embedded in observation.metadata so inference.py can
        # thread it back via action.metadata on every subsequent step.
        import uuid as _uuid
        sid = episode_id or _DEFAULT_SESSION
        log.info("[RESET] session=%s task=%s seed=%d", sid, task_id, seed)

        current_df, ground_truth_df = load_task_data(task_id, seed=seed)
        metrics   = calculate_similarity(current_df, ground_truth_df, task_id)
        issues    = detect_issues(current_df, task_id)
        max_steps = TASK_MAX_STEPS.get(task_id, 10)

        state = _EpisodeState(
            current_df=current_df,
            ground_truth_df=ground_truth_df,
            task_id=task_id,
            episode_id=episode_id,
            session_key=sid,
            max_steps=max_steps,
            issues_detected=issues,
            current_metrics=metrics,
        )
        self._put_session(sid, state)
        return self._build_observation(state, done=False, reward=None)

    def step(
        self,
        action:     DatagymAction,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> StepResponse:
        """
        POST /step handler.

        Executes the action, recomputes metrics, and returns a StepResponse.
        Auto-resets on cold-start / LRU eviction to prevent NoneType crashes.
        """
        # Prefer session key threaded back from the client via action.metadata.
        # Falls back to __default__ for backward compatibility.
        sid   = (action.metadata or {}).get("session_id", _DEFAULT_SESSION)
        state = self._get_session(sid)

        if state is None:
            log.warning("[STEP] No state found — cold-start reset to task1_easy")
            self.reset(session_id=sid)
            state = self._get_session(sid)

        state.step_count += 1
        prev_f1 = float(state.current_metrics.get("f1", 0.0))
        log.debug(
            "[STEP %d] session=%s action=%s col=%s prev_f1=%.4f",
            state.step_count, sid, action.action_type, action.column, prev_f1,
        )

        success, error_msg = self._execute_action(state, action)

        if success:
            state.history.append(action)
            state.current_metrics = calculate_similarity(
                state.current_df, state.ground_truth_df, state.task_id
            )
            step_reward = float(state.current_metrics.get("f1", 0.0)) - prev_f1
            # Refresh issues so the agent sees current state, not the initial snapshot
            state.issues_detected = detect_issues(state.current_df, state.task_id)
            log.debug(
                "[STEP %d] grader → f1=%.4f prec=%.4f rec=%.4f reward=%.4f",
                state.step_count,
                state.current_metrics.get("f1", 0.0),
                state.current_metrics.get("precision", 0.0),
                state.current_metrics.get("recall", 0.0),
                step_reward,
            )
        else:
            step_reward = -0.05
            # Append failure note but keep current issues fresh
            current_issues = detect_issues(state.current_df, state.task_id)
            state.issues_detected = current_issues + [
                f"[Step {state.step_count} FAILED] {error_msg}"
            ]
            log.warning("[STEP %d] FAILED: %s", state.step_count, error_msg)

        is_terminal = (
            action.action_type == "submit"
            or state.step_count >= state.max_steps
        )
        state.terminated = is_terminal
        self._put_session(sid, state)

        obs = self._build_observation(state, done=is_terminal, reward=step_reward)
        return StepResponse(
            observation=obs.model_dump(),
            reward=step_reward,
            done=is_terminal,
        )

    def get_state(self, session_id: Optional[str] = None, **kwargs) -> DatagymState:
        """GET /state handler (method form)."""
        return self._build_state()

    @property
    def state(self) -> DatagymState:
        """GET /state handler (property form — required by openenv-core)."""
        return self._build_state()

    def _build_state(self) -> DatagymState:
        """Shared implementation for both get_state() and state property."""
        s = self._get_session(_DEFAULT_SESSION)
        if s is None:
            return DatagymState(
                episode_id=None, step_count=0, task_id="task1_easy",
                current_f1=0.0, is_terminated=False,
            )
        return DatagymState(
            episode_id=s.episode_id,
            step_count=s.step_count,
            task_id=s.task_id,
            current_f1=float(s.current_metrics.get("f1", 0.0)),
            is_terminated=s.terminated,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Async wrappers
    # ──────────────────────────────────────────────────────────────────────────

    async def reset_async(self, task_id="task1_easy", seed=42,
                          episode_id=None, session_id=None, **kwargs):
        return self.reset(task_id=task_id, seed=seed,
                          episode_id=episode_id, session_id=session_id, **kwargs)

    async def step_async(self, action: DatagymAction,
                         session_id=None, **kwargs) -> StepResponse:
        return self.step(action, session_id=session_id, **kwargs)

    async def get_state_async(self, session_id=None, **kwargs) -> DatagymState:
        return self.get_state(session_id=session_id, **kwargs)

    def close(self, session_id=None, **kwargs) -> None:
        # openenv-core calls close() after EVERY request as a per-request teardown.
        # Deleting the session here wipes state between reset() and step().
        # LRU eviction in _put_session() handles memory — intentional no-op.
        log.debug("[CLOSE] called (no-op — state preserved in LRU store)")

    async def close_async(self, session_id=None, **kwargs) -> None:
        self.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize(obj):
        """
        Recursively convert numpy/pandas scalars → native Python types.

        Pydantic's JSON serializer crashes on pd.Timestamp, np.int64, pd.NA,
        etc.  This walk ensures the observation dict is fully JSON-safe before
        it reaches Pydantic.
        """
        if isinstance(obj, dict):
            return {k: DatagymEnvironment._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [DatagymEnvironment._sanitize(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return [DatagymEnvironment._sanitize(v) for v in obj.tolist()]
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if math.isnan(v) else v
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    @staticmethod
    def _build_observation(
        state:  _EpisodeState,
        done:   bool,
        reward: Optional[float],
    ) -> DatagymObservation:
        san     = DatagymEnvironment._sanitize
        preview = san(state.current_df.head(10).to_dict(orient="records"))
        schema  = san({
            "columns":       list(state.current_df.columns),
            "inferred_types": state.current_df.dtypes.astype(str).to_dict(),
            "null_counts":   state.current_df.isnull().sum().to_dict(),
            "shape":         list(state.current_df.shape),
        })
        return DatagymObservation(
            done=done,
            reward=reward,
            metadata={"episode_id": state.episode_id, "task_id": state.task_id, "session_id": state.session_key},
            dataset_preview=preview,
            schema_info=schema,
            issues_detected=list(state.issues_detected),
            actions_taken=list(state.history),
            step=state.step_count,
            max_steps=state.max_steps,
            f1_score=float(state.current_metrics.get("f1", 0.0)),
            precision=float(state.current_metrics.get("precision", 0.0)),
            recall=float(state.current_metrics.get("recall", 0.0)),
        )

    @staticmethod
    def _execute_action(
        state:  _EpisodeState,
        action: DatagymAction,
    ) -> Tuple[bool, str]:
        """
        Applies the pandas operation to state.current_df.

        Uses assignment (never inplace=True) so a failed op leaves
        state.current_df pointing to the pre-action DataFrame.

        Returns:
            (True, "")          on success
            (False, error_msg)  on any failure — never raises
        """
        col    = action.column
        params = action.params or {}
        df     = state.current_df

        column_agnostic = {"submit", "deduplicate"}
        if col and action.action_type not in column_agnostic:
            if col not in df.columns:
                return False, f"Column '{col}' does not exist in the dataset."

        try:
            # ── Basic cleaning ────────────────────────────────────────────────
            if action.action_type == "drop_nulls":
                if not col:
                    return False, "drop_nulls requires a column name."
                rows_to_drop = int(df[col].isna().sum())
                pct = rows_to_drop / len(df) if len(df) > 0 else 0
                if pct > 0.15:
                    return False, (
                        f"drop_nulls refused: would remove {rows_to_drop} rows "
                        f"({pct:.0%}). Use fill_nulls to preserve row count."
                    )
                state.current_df = df.dropna(subset=[col]).reset_index(drop=True)

            elif action.action_type == "fill_nulls":
                strategy = params.get("strategy")
                if strategy == "mean":
                    # Coerce to numeric first so mixed-type columns (e.g. amount
                    # with "$500" strings) don't crash .mean()
                    numeric_col = pd.to_numeric(df[col], errors="coerce")
                    if pd.isna(numeric_col).all():
                        return False, (
                            f"Cannot compute mean for '{col}': no parseable numeric values. "
                            f"Normalize the column first (e.g. strip currency symbols)."
                        )
                    # Exclude extreme outliers (> 3 IQR from Q3) from the mean
                    # so that fill values are not inflated by corruption artifacts.
                    q1, q3 = numeric_col.quantile(0.25), numeric_col.quantile(0.75)
                    iqr = q3 - q1
                    upper = q3 + 3 * iqr
                    clean = numeric_col[numeric_col <= upper]
                    val = clean.mean() if not clean.empty else numeric_col.mean()
                elif strategy == "mode":
                    val = df[col].mode()
                    if val.empty:
                        return False, f"Cannot compute mode: column '{col}' is all-null."
                    val = val[0]
                else:
                    val = params.get("value", "UNKNOWN")

                if pd.api.types.is_integer_dtype(df[col]) and pd.notna(val):
                    val = int(round(float(val)))

                tmp = df.copy()
                tmp[col] = df[col].fillna(val)
                state.current_df = tmp

            elif action.action_type == "cast_type":
                target   = params.get("target_type")
                type_map = {
                    "int":      "Int64",
                    "float":    "float64",
                    "str":      "string",
                    "datetime": "datetime64[ns]",
                }
                if target not in type_map:
                    return False, f"Unsupported target_type: '{target}'."

                tmp = df.copy()
                if target == "datetime":
                    # Handle all formats present in task2: DD/MM/YYYY, MM-DD-YY,
                    # Unix timestamp integers, and ISO 8601. A single pd.to_datetime
                    # call coerces every non-ISO value to NaT — use per-row parsing.
                    import re as _re

                    def _parse_date(val):
                        v = str(val).strip()
                        if v in ("nan", "None", "NaT", ""):
                            return pd.NaT
                        if _re.match(r"^[0-9]{9,10}$", v):  # Unix timestamp
                            try:
                                return pd.to_datetime(int(v), unit="s")
                            except Exception:
                                return pd.NaT
                        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m-%d-%y",
                                    "%m/%d/%Y", "%d-%m-%Y"):
                            try:
                                return pd.to_datetime(v, format=fmt)
                            except ValueError:
                                continue
                        try:
                            return pd.to_datetime(v)  # pandas inference fallback
                        except Exception:
                            return pd.NaT

                    tmp[col] = df[col].apply(_parse_date)
                else:
                    if target in ("int", "float"):
                        tmp[col] = pd.to_numeric(df[col], errors="coerce")
                    tmp[col] = tmp[col].astype(type_map[target])
                state.current_df = tmp

            elif action.action_type == "normalize_values":
                mapping = params.get("mapping")
                if not isinstance(mapping, dict):
                    return False, "normalize_values requires params.mapping to be a dict."
                tmp  = df.copy()
                orig = df[col].copy()

                # ── Numeric column: agent sends string keys like "999999.99" ──
                # pd.Series.replace with string keys never matches float values.
                # Coerce numeric-looking keys to the column dtype so the replace
                # works regardless of whether cast_type was called first.
                if pd.api.types.is_numeric_dtype(orig):
                    # Numeric column: coerce string keys to float so that
                    # {"999999.99": "260.41"} matches float values after cast_type.
                    numeric_mapping = {}
                    for k, v in mapping.items():
                        try:
                            fk = float(k)
                            fv = float(v)
                            numeric_mapping[fk] = fv
                        except (ValueError, TypeError):
                            numeric_mapping[k] = v
                    result = orig.replace(numeric_mapping)
                else:
                    # String column: apply each key→value pair SEQUENTIALLY so
                    # that chained transforms work in one action call.
                    # e.g. {"   ": "", "food": "Food"} correctly turns
                    # "food   " → "food" → "Food" in the same step.
                    import re as _re2
                    result = orig.copy()
                    for k, v in mapping.items():
                        prev = result.copy()
                        # Try regex first
                        try:
                            result = result.replace({k: v}, regex=True)
                        except Exception:
                            result = prev
                        # If unchanged, try re.escape (handles $, ., etc.)
                        if result.equals(prev):
                            try:
                                result = result.replace(
                                    {_re2.escape(k): v}, regex=True
                                )
                            except Exception:
                                result = prev
                        # Final fallback: literal
                        if result.equals(prev):
                            result = result.replace({k: v}, regex=False)

                tmp[col] = result
                state.current_df = tmp

            elif action.action_type == "rename_column":
                new_name = params.get("new_name")
                if not new_name:
                    return False, "rename_column requires params.new_name."
                state.current_df = df.rename(columns={col: new_name})

            elif action.action_type == "deduplicate":
                keys = params.get("key_columns", df.columns.tolist())
                state.current_df = df.drop_duplicates(subset=keys).reset_index(drop=True)

            # ── Schema repair (task3_hard) ─────────────────────────────────────
            elif action.action_type == "drop_column":
                if not col:
                    return False, "drop_column requires a column name."
                state.current_df = df.drop(columns=[col])

            elif action.action_type == "split_column":
                delimiter   = params.get("delimiter", " ")
                new_columns = params.get("new_columns")
                if not new_columns or len(new_columns) < 2:
                    return False, (
                        "split_column requires params.new_columns "
                        "(list of at least 2 names)."
                    )
                split_result = df[col].astype(str).str.split(delimiter, expand=True)
                tmp = df.copy()
                for i, new_col in enumerate(new_columns):
                    if i < split_result.shape[1]:
                        tmp[new_col] = split_result[i]
                if params.get("drop_original", True):
                    tmp = tmp.drop(columns=[col])
                state.current_df = tmp

            elif action.action_type == "parse_json_column":
                key     = params.get("key")
                new_col = params.get("new_column")
                if not key or not new_col:
                    return False, (
                        "parse_json_column requires params.key and params.new_column."
                    )

                def _extract(cell):
                    try:
                        return json_lib.loads(cell).get(key)
                    except Exception:
                        return None

                tmp         = df.copy()
                tmp[new_col] = df[col].apply(_extract)
                state.current_df = tmp

            elif action.action_type == "arithmetic_transform":
                operator = params.get("operator", "/")
                value    = params.get("value")
                new_name = params.get("new_name")
                if value is None:
                    return False, "arithmetic_transform requires params.value."

                numeric = pd.to_numeric(df[col], errors="coerce")
                ops     = {"/": numeric / value, "*": numeric * value,
                           "+": numeric + value, "-": numeric - value}
                if operator not in ops:
                    return False, f"Unsupported operator '{operator}'. Use / * + -."

                tmp     = df.copy()
                tmp[col] = ops[operator].round(2)
                if new_name:
                    tmp = tmp.rename(columns={col: new_name})
                state.current_df = tmp

            elif action.action_type == "submit":
                pass  # Terminal signal — no mutation

            else:
                return False, f"Unknown action_type: '{action.action_type}'."

            return True, ""

        except Exception as exc:
            traceback.print_exc()
            return False, f"Execution error: {str(exc).splitlines()[0]}"