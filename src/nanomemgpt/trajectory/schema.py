from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class FailureType(str, Enum):
    RETRIEVAL_MISS = "retrieval_miss"
    RETRIEVAL_HALLUCINATION = "retrieval_hallucination"
    CHAIN_FAILURE = "chain_failure"
    FINAL_ANSWER_FAILURE = "final_answer_failure"
    NONE = "none"


class FunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    request_heartbeat: bool = False


class TrajectoryStep(BaseModel):
    sample_id: str
    split: str = "train"
    step_index: int
    context: list[dict[str, Any]]
    teacher_action: FunctionCall | None = None
    function_output: dict[str, Any] | None = None
    target_text: str | None = None
    source: Literal["teacher", "oracle", "student"] = "teacher"


class FailureAnnotation(BaseModel):
    sample_id: str
    student_run_id: str
    teacher_run_id: str | None = None
    failure_type: FailureType
    rationale: str = ""
