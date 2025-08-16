from pydantic import BaseModel, Field
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class PromptFiles:
    system_path: Path
    user_path: Path

class ToDoBundle(BaseModel):
    do_this_now: str = Field(..., description="지금 당장 할 1~3개의 구체 행동")
    why_this_matters: str = Field(..., description="행동의 근거/개인화 설명")
    summary_and_habit_guide: str = Field(..., description="상태 요약 + 습관 가이드")
    short_term_plan: str = Field(..., description="3~7일 체크리스트/빈도/트리거 계획")
