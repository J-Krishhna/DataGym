"""
Pydantic models for the DataGym OpenEnv environment.

These models must stay in sync with the OpenEnv schema contract.
The Swagger-derived schemas impose `additionalProperties: false` on
DatagymAction and DatagymObservation, which is enforced via
`model_config = ConfigDict(extra="forbid")`.
"""
from pydantic import BaseModel, Field, field_validator, ConfigDict
import json
from typing import Optional, Dict, Any, List, Union
from openenv.core.env_server.types import Action, Observation


class DatagymAction(Action):
    """
    Action model for the DataGym environment.

    Schema contract (from /schema endpoint):
      - additionalProperties: false  →  extra="forbid"
      - Required fields: action_type
      - metadata is required by the OpenEnv Action base class
    """

    model_config = ConfigDict(extra="forbid")

    # Required by OpenEnv Action base — must be present even if empty
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the action",
    )

    # Domain fields
    action_type: str = Field(..., title="Action Type")
    column: Optional[str] = Field(default=None, title="Column")
    params: Optional[Dict[str, Any]] = Field(default=None, title="Params")

    @field_validator("params", mode="before")
    @classmethod
    def parse_params_if_string(cls, v):
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if not isinstance(parsed, dict):
                    raise ValueError(f"params must be a JSON object, got: {type(parsed).__name__}")
                return parsed
            except json.JSONDecodeError as e:
                raise ValueError(f"params is not valid JSON: {e}")
        return v


class DatagymObservation(Observation):
    """
    Observation model for the DataGym environment.

    Schema contract (from /schema endpoint):
      - additionalProperties: false  →  extra="forbid"
      - done, reward, metadata are required by the OpenEnv Observation base
      - dataset_preview, schema_info, issues_detected, actions_taken,
        step, max_steps are required domain fields
    """

    model_config = ConfigDict(extra="forbid")

    # ── OpenEnv Observation base fields ──────────────────────────────────────
    done: bool = Field(
        default=False,
        description="Whether the episode has terminated",
    )
    reward: Optional[Union[bool, int, float]] = Field(
        default=None,
        description="Reward signal from the last action",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the observation",
    )

    # ── Required domain fields ────────────────────────────────────────────────
    dataset_preview: List[Dict[str, Any]] = Field(..., title="Dataset Preview")
    schema_info: Dict[str, Any] = Field(..., title="Schema Info")
    issues_detected: List[str] = Field(..., title="Issues Detected")
    actions_taken: List[DatagymAction] = Field(..., title="Actions Taken")
    step: int = Field(..., title="Step")
    max_steps: int = Field(..., title="Max Steps")

    # ── Scoring metrics (optional, default 0) ────────────────────────────────
    f1_score: float = Field(default=0.0, title="F1 Score")
    precision: float = Field(default=0.0, title="Precision")
    recall: float = Field(default=0.0, title="Recall")


class DatagymState(BaseModel):
    """
    Internal environment state returned by GET /state.

    Schema contract:
      - additionalProperties: true  →  extra="allow"
      - episode_id and step_count are the base OpenEnv fields
      - Additional fields (task_id, current_f1, is_terminated) are allowed
        and surfaced to aid debugging / judging.
    """

    model_config = ConfigDict(extra="allow")

    episode_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the current episode",
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of steps taken in the current episode",
    )

    # Extra fields exposed for transparency
    task_id: str = Field(default="task1_easy")
    current_f1: float = Field(default=0.0)
    is_terminated: bool = Field(default=False)