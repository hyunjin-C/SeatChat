from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import os
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import importlib.util
from models import ToDoBundle, PromptFiles

BEHAVIOR_PATH = str(Path("./scripts/posture_behavior_analysis_all.py"))
PATTERN_PATH  = str(Path("./scripts/posture_pattern_all_repeat_time.py"))
RANKING_PATH  = str(Path("./scripts/posture_ranking_all.py"))

DO_SYS = str(Path("./prompts/DO_THIS_NOW_system_prompt.txt"))
DO_USER = str(Path("./prompts/DO_THIS_NOW_user_template.txt"))
WHY_SYS = str(Path("./prompts/WHY_THIS_MATTERS_system_prompt.txt"))
WHY_USER = str(Path("./prompts/WHY_THIS_MATTERS_user_template.txt"))
GUIDE_SYS = str(Path("./prompts/SUMMARY_HABIT_GUARD_system_prompt.txt"))
GUIDE_USER = str(Path("./prompts/SUMMARY_HABIT_GUARD_user_template.txt"))
PLAN_SYS = str(Path("./prompts/SHORT_TERM_PLAN_system_prompt.txt"))
PLAN_USER = str(Path("./prompts/SHORT_TERM_PLAN_user_template.txt"))
CHAT_SYS = str(Path("./prompts/ASK_ME_CHAT_system_prompt.txt"))
CHAT_USER = str(Path("./prompts/ASK_ME_CHAT_user_template.txt"))

PROMPT_FILES = {
    "do":    PromptFiles(DO_SYS,   DO_USER),
    "why":   PromptFiles(WHY_SYS,  WHY_USER),
    "guide": PromptFiles(GUIDE_SYS,GUIDE_USER),
    "plan":  PromptFiles(PLAN_SYS, PLAN_USER),
    "chat":  PromptFiles(CHAT_SYS, CHAT_USER),
}

PREFERENCES_JSON_PATH = Path("./data/user_preferences.json")

_DEFAULT_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def _load_module(name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _read_text(path: str | Path) -> str:
    path = str(path)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def _jsonify_if_needed(v):
    # 리스트/딕셔너리/불리언은 JSON 문자열로, 그 외는 문자열
    if isinstance(v, (list, dict, bool)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

# tone_preference: [] or ["neutral_informational"] or ["motivational_encouraging"] or ["authoritative_strict"]
# guidance_style: "prescriptive" | "facilitative" | "mixed" | "unspecified"
# live.current_label: UPRIGHT|UPRIGHT_LEFT|UPRIGHT_RIGHT|FORWARD|FORWARD_LEFT|FORWARD_RIGHT|BACK|BACK_LEFT|BACK_RIGHT
# personal.main_discomforts: ["neck stiffness","lower back pain"] or []
# prefs.allowed_stretches: ["chest_opener","neck_release","hip_hinge_extension"] or []
# profile.chat_interaction: "minimal_notifications" | "conversational" | "occasional_qna"
# profile.config_intentions: ["tone","notification_frequency","feedback_modality","goal_settings"] or []
# profile.feedback_modality: ["text_notification","visual_diagram","auditory_voice","tactile_vibration"] or []
# profile.alert_rule: "immediate" | "every_few_minutes" | "hourly_summary" | "only_when_extremely_bad" | "user_configurable" | "unspecified"
# profile.trust_preference: {"ai_only_ok":true|null,"human_checkins_desired":"none|occasional|regular"}
# profile.privacy_sensing: {"accepts_storage":"unsure","sensing":["pressure_chair"]}

def _build_common_inputs(metrics: Dict[str, Any], preference: Dict[str, Any]) -> Dict[str, Any]:
    """
    - metrics: dict from _summarize_once()
      • metrics['core']           : core metrics (Good posture time 등)
      • metrics['dominance_top']  : dominance top label/repetition
      • metrics['motif_summary']  : L2/L3 motif summary
      • metrics['live']    : live label/duration
      • metrics['batch_window_minutes'] : recent window (minutes)
    - preference: user preference/runtime settings
      • tone_preference, guidance_style, pain_flags, key_goal, improvement_timeframe
      • history_good_pct
      • params_* / wants_stretch / allowed_stretches
      • session_*  (elapsed/remaining/anchor_candidates)

    - returns: flat dotted keys dict
    """
    p = preference or {}
    m = metrics or {}

    core = (m.get("core") or {})
    dom  = (m.get("dominance_top") or {})
    live = (m.get("live") or {})
    motifs = (m.get("motif_summary") or {})
    batch_window_min = int(m.get("batch_window_minutes", 10))

    # 기본 선호/개인정보
    tone_pref = p.get("tone_preference", [])
    guidance  = p.get("guidance_style", "unspecified")
    pain_flags = p.get("pain_flags", [])
    key_goal   = p.get("key_goal") or p.get("goal") or None
    improvement_timeframe = p.get("improvement_timeframe", "4-6 weeks")

    # 코어/배치 지표
    good_minutes = core.get("Good posture time (min)", 0) or 0
    bad_minutes  = core.get("Bad posture time (min)", 0) or 0
    good_pct     = core.get("Good posture time (%)", 0) or 0
    bad_pct      = 100 - good_pct if isinstance(good_pct, (int, float)) else 0

    top_time_label   = dom.get("top_time_label", "FORWARD")
    top_time_minutes = dom.get("top_time_minutes", 0) or 0
    top_rep_label    = dom.get("top_repetition_label", "FORWARD")
    top_rep_count    = int(dom.get("top_repetition_count", 0) or 0)

    # 라이브(실시간) 라벨/지속
    current_label  = live.get("current_label") or top_time_label or "FORWARD"
    continuous_sec = int(live.get("continuous_seconds_current_label", 0) or 0)

    # 히스토리 기준선/대표 반복 시퀀스
    history_good_pct = p.get("history_good_pct", good_pct) or good_pct
    l3_top_seq = (motifs.get("L3") or {}).get("most_repeated_sequence")
    l2_top_seq = (motifs.get("L2") or {}).get("most_repeated_sequence")
    history_top_repeated_sequence = l3_top_seq or l2_top_seq or "N/A"

    # 최근 윈도우 해석(Guide/Plan 용)
    recent_window_min = batch_window_min
    recent_good_pct   = good_pct
    recent_top_label  = top_time_label if top_time_label not in ("UPRIGHT", None) else top_rep_label
    recent_top_count  = int(top_rep_count or 0)
    # 평균 episode 길이(초): 총 시간(분) → 초 / 횟수 (5초 단위 반올림)
    recent_top_avg_sec = int(round(((top_time_minutes or 0) * 60) / max(recent_top_count, 1) / 5) * 5)
    recent_continuous_bad_sec = int(round((continuous_sec or 0) / 5) * 5)
    recent_repetition_label = top_rep_label
    recent_most_repeated_sequence = history_top_repeated_sequence
    recent_most_repeated_count = int(
        (motifs.get("L3") or {}).get("most_repeated_count")
        or (motifs.get("L2") or {}).get("most_repeated_count")
        or 0
    )

    # 파라미터/런타임 기본값
    params_T_sec = int(p.get("params_T_sec", 90))
    params_N     = int(p.get("params_N", 3))
    params_M     = int(p.get("params_M", recent_window_min))
    params_delta_pct_threshold   = int(p.get("params_delta_pct_threshold", 5))
    params_suggest_stand_break   = bool(p.get("params_suggest_stand_break", True))
    params_suggest_stand_break_sec = int(p.get("params_suggest_stand_break_sec", 45))

    params_reset_interval_min = int(p.get("params_reset_interval_min", max(20, batch_window_min)))
    params_stand_sec          = int(p.get("params_stand_sec", 45))
    params_forward_bout_cap_sec = int(p.get("params_forward_bout_cap_sec", 60))
    params_wants_habit_break_clause = bool(p.get("params_wants_habit_break_clause", True))
    params_wants_anchor_stretch     = bool(p.get("params_wants_anchor_stretch", True))

    prefs_wants_stretch   = bool(p.get("wants_stretch", False))
    prefs_allowed_stretch = p.get("allowed_stretches",
                                  ["chest_opener","neck_release","hip_hinge_extension"])

    session_minutes_elapsed   = int(p.get("session_minutes_elapsed", 0))
    session_minutes_remaining = int(p.get("session_minutes_remaining", 60))
    session_anchor_candidates = p.get("session_anchor_candidates",
                                      ["start of next focus", "after meeting", "top of hour"])

    ci = {
        "tone_preference": tone_pref,
        "guidance_style": guidance,

        "live.current_label": current_label,
        "live.continuous_seconds_current_label": continuous_sec,

        "batch.window_minutes": batch_window_min,
        "batch.top_time_label": top_time_label,
        "batch.top_time_minutes": top_time_minutes,
        "batch.top_repetition_label": top_rep_label,
        "batch.top_repetition_count": top_rep_count,
        "batch.good_minutes": good_minutes,
        "batch.bad_minutes": bad_minutes,
        "batch.good_pct": good_pct,
        "batch.bad_pct": bad_pct,

        "personal.main_discomforts": pain_flags,
        "personal.key_goal": key_goal,
        "personal.improvement_timeframe": improvement_timeframe,

        "history.good_pct": history_good_pct,
        "history.top_repeated_sequence": history_top_repeated_sequence,

        "recent.window_min": recent_window_min,
        "recent.good_pct": recent_good_pct,
        "recent.top_label": recent_top_label or "FORWARD",
        "recent.top_count": recent_top_count,
        "recent.top_avg_sec": recent_top_avg_sec,
        "recent.continuous_bad_sec": recent_continuous_bad_sec,
        "recent.repetition_label": recent_repetition_label or "FORWARD",
        "recent.most_repeated_sequence": recent_most_repeated_sequence,
        "recent.most_repeated_count": recent_most_repeated_count,

        "params.T_sec": params_T_sec,
        "params.N": params_N,
        "params.M": params_M,
        "params.delta_pct_threshold": params_delta_pct_threshold,
        "params.suggest_stand_break": params_suggest_stand_break,
        "params.suggest_stand_break_sec": params_suggest_stand_break_sec,

        "params.reset_interval_min": params_reset_interval_min,
        "params.stand_sec": params_stand_sec,
        "params.forward_bout_cap_sec": params_forward_bout_cap_sec,
        "params.wants_habit_break_clause": params_wants_habit_break_clause,
        "params.wants_anchor_stretch": params_wants_anchor_stretch,

        "prefs.wants_stretch": prefs_wants_stretch,
        "prefs.allowed_stretches": prefs_allowed_stretch,

        "session.minutes_elapsed": session_minutes_elapsed,
        "session.minutes_remaining": session_minutes_remaining,
        "session.anchor_candidates": session_anchor_candidates,
    }
    return ci

def _pick(d: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    return {k: d[k] for k in keys if k in d}

def _fill_tokens(template_text: str, values: Dict[str, Any]) -> str:
    """
    템플릿의 <<FILL: key>> 토큰을 values[key]로 치환.
    - key는 'tone_preference' 같은 '정규화된 이름'을 권장.
    - 값은 자동으로 문자열화(리스트/딕셔너리는 JSON 직렬화).
    """
    out = template_text
    for k, v in values.items():
        token = f"<<FILL: {k}>>"
        out = out.replace(token, _jsonify_if_needed(v))
    return out

def _make_prompt_from_files(kind: str, user_values: Dict[str, Any]) -> ChatPromptTemplate:
    """
    kind: 'do' | 'why' | 'guide' | 'plan' | 'chat'
    user_values: 사용자 템플릿에 들어갈 key→value 맵
    """
    files = PROMPT_FILES[kind]
    sys_text   = _read_text(files.system_path)
    user_proto = _read_text(files.user_path)
    user_filled = _fill_tokens(user_proto, user_values)
    # system/user 한 쌍으로 고정
    return ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        ("human", "{user_prompt}")
    ]).partial(system_prompt=sys_text, user_prompt=user_filled)

class _Pipeline:
    """
    내부 상태를 보관하는 파이프라인 구현.
    - user_id별 preference 로딩/전환
    - 분석/요약/To-Do/채팅
    """
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.llm = _DEFAULT_LLM
        self.memory: Optional[ConversationSummaryBufferMemory] = None

        self.m_behavior = _load_module("posture_behavior_analysis_all", BEHAVIOR_PATH)
        self.m_pattern  = _load_module("posture_pattern_all_repeat_time", PATTERN_PATH)
        self.m_ranking  = _load_module("posture_ranking_all", RANKING_PATH)

        self._preferences_store: Dict[str, Dict[str, Any]] = {}
        self.preference: Dict[str, Any] = {}
        self._load_preferences_store(PREFERENCES_JSON_PATH)
        self._apply_user_preference(self.user_id)

        self.posture_summary_text: str = ""
        self.posture_summary_metrics: Dict[str, Any] = {}
        self.last_todo: Optional[ToDoBundle] = None

        self.annotation_path: Optional[str] = None
        self.annotation_supplier = None

    def set_llm(self, llm):
        self.llm = llm
        if self.memory is not None:
            self.memory.llm = llm

    def ensure_memory(self):
        if self.memory is None:
            if self.llm is None:
                raise RuntimeError("LLM is not initialized. Call configure_llm(...) first.")
            self.memory = ConversationSummaryBufferMemory(
                llm=self.llm, max_token_limit=2000, return_messages=True
            )

    def set_preferences(self, preference: Dict[str, Any]):
        self.preference = preference or {}

    def _load_preferences_store(self, path: Path):
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._preferences_store = data if isinstance(data, dict) else {}
            else:
                self._preferences_store = {}
        except Exception:
            self._preferences_store = {}

    def _apply_user_preference(self, user_id: str):
        self.user_id = user_id
        self.preference = self._preferences_store.get(user_id, {})

    def reload_preferences_from_json(self, json_path: str | Path = PREFERENCES_JSON_PATH):
        self._load_preferences_store(Path(json_path))
        self._apply_user_preference(self.user_id)

    def set_annotation_path(self, path: str):
        self.annotation_path = path

    def set_annotation_supplier(self, fn):
        """fn() -> (df, optional_participant_path)"""
        self.annotation_supplier = fn

    def _summarize_once(self, df: Any, participant_path: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        core = self.m_behavior.compute_summary_row(df)
        summary_df = pd.DataFrame([core])
        top_df, _ranks_df = self.m_ranking.build_dominance_tables(summary_df)

        if not top_df.empty:
            raw_dom = top_df.iloc[0].to_dict()
        else:
            raw_dom = {}

        dominance_top = {
            "top_time_label": raw_dom.get("top_time_label"),
            "top_time_minutes": raw_dom.get("top_time_minutes", raw_dom.get("top_time_total_min", 0)) or 0,
            "top_repetition_label": raw_dom.get("top_repetition_label"),
            "top_repetition_count": int(raw_dom.get("top_repetition_count", 0) or 0),
        }

        temp_path = None
        try:
            if participant_path and Path(participant_path).exists():
                row, _motifs = self.m_pattern.analyze_participant(Path(participant_path))
            else:
                temp_path = Path.cwd() / f"_tmp_{int(time.time()*1000)}.xlsx"
                with pd.ExcelWriter(temp_path) as w:
                    df.to_excel(w, index=False, sheet_name="annotations")
                row, _motifs = self.m_pattern.analyze_participant(temp_path)
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

        l3_seq = row.get("L3_most_time_sequence") or row.get("L3_most_repeated_sequence")
        l2_seq = row.get("L2_most_time_sequence") or row.get("L2_most_repeated_sequence")

        def _infer_live_from_df(df: pd.DataFrame, fallback_label: Optional[str]) -> Dict[str, int | str]:
            label_cols = ["label", "Label", "posture_label", "posture", "state", "State"]
            dur_cols   = ["duration_sec", "duration_s", "dur_sec", "seconds"]

            label_col = next((c for c in label_cols if c in df.columns), None)
            dur_col   = next((c for c in dur_cols if c in df.columns), None)

            start_cols = ["start", "start_time", "Start", "StartTime"]
            end_cols   = ["end", "end_time", "End", "EndTime"]
            start_col = next((c for c in start_cols if c in df.columns), None)
            end_col   = next((c for c in end_cols if c in df.columns), None)

            current_label = None
            continuous_sec = 0

            if label_col is None:
                current_label = fallback_label or "FORWARD"
                return {"current_label": current_label, "continuous_seconds_current_label": int(continuous_sec)}

            current_label = str(df[label_col].iloc[-1]) if len(df) else (fallback_label or "FORWARD")

            if len(df) == 0:
                return {"current_label": current_label, "continuous_seconds_current_label": int(continuous_sec)}

            i = len(df) - 1
            while i >= 0 and str(df[label_col].iloc[i]) == current_label:
                if dur_col and pd.notna(df[dur_col].iloc[i]):
                    try:
                        continuous_sec += float(df[dur_col].iloc[i] or 0)
                    except Exception:
                        pass
                elif start_col and end_col and pd.notna(df[start_col].iloc[i]) and pd.notna(df[end_col].iloc[i]):
                    try:
                        t0 = pd.to_datetime(df[start_col].iloc[i])
                        t1 = pd.to_datetime(df[end_col].iloc[i])
                        continuous_sec += max(0.0, (t1 - t0).total_seconds())
                    except Exception:
                        pass
                else:
                    continuous_sec += 0
                i -= 1

            return {
                "current_label": current_label or (fallback_label or "FORWARD"),
                "continuous_seconds_current_label": int(round(continuous_sec / 5) * 5),
            }

        live = _infer_live_from_df(df, dominance_top.get("top_time_label"))

        good_pct = core.get("Good posture time (%)")
        text = (
            f"Good posture percentage: {good_pct}% | "
            f"Dominance time: {dominance_top.get('top_time_label')} | "
            f"Dominance repetition: {dominance_top.get('top_repetition_label')} | "
            f"L3 motif: {l3_seq} | L2 motif: {l2_seq}"
        )

        metrics = {
            "core": core,
            "dominance_top": dominance_top,
            "motif_summary": {
                "L3": {
                    "most_time_sequence": row.get("L3_most_time_sequence"),
                    "most_time_total_min": row.get("L3_most_time_total_min"),
                    "most_repeated_sequence": row.get("L3_most_repeated_sequence"),
                    "most_repeated_count": row.get("L3_most_repeated_count"),
                },
                "L2": {
                    "most_time_sequence": row.get("L2_most_time_sequence"),
                    "most_time_total_min": row.get("L2_most_time_total_min"),
                    "most_repeated_sequence": row.get("L2_most_repeated_sequence"),
                    "most_repeated_count": row.get("L2_most_repeated_count"),
                },
            },
            "batch_window_minutes": 10,
            "live": live,
        }
        return text, metrics

    def _gen_note(self, metrics: Dict[str, Any], preference: Dict[str, Any]) -> ToDoBundle:
        if self.llm is None:
            raise RuntimeError("LLM is not initialized. Call configure_llm(...) first.")
        
        config = _build_common_inputs(metrics, preference)

        # ----- 1) DO THIS NOW -----
        do_values = _pick(config,
            "tone_preference",
            "guidance_style",
            "live.current_label",
            "live.continuous_seconds_current_label",
            "batch.window_minutes",
            "batch.top_time_label",
            "batch.top_time_minutes",
            "batch.top_repetition_label",
            "batch.top_repetition_count",
            "batch.good_minutes",
            "batch.bad_minutes",
            "batch.good_pct",
            "batch.bad_pct",
            "personal.main_discomforts",
            "personal.key_goal",
        )
        prompt_do = _make_prompt_from_files("do", do_values)
        do_chain = prompt_do | self.llm | StrOutputParser()
        do_this_now = do_chain.invoke({}).strip()
        if "\n" in do_this_now:
            do_this_now = do_this_now.splitlines()[0].strip()
        
        # ----- 2) WHY THIS MATTERS -----
        why_values = _pick(config,
            "tone_preference",
            "live.current_label",
            "batch.window_minutes",
            "batch.top_time_label",
            "batch.top_time_minutes",
            "batch.top_repetition_label",
            "batch.top_repetition_count",
            "batch.bad_pct",
            "personal.main_discomforts",
            "personal.key_goal",
            "personal.improvement_timeframe",
        )
        prompt_why = _make_prompt_from_files("why", why_values)
        why_chain = prompt_why | self.llm | StrOutputParser()
        why_this_matters = why_chain.invoke({}).strip()

        # ----- 3) SUMMARY & HABIT GUIDE -----
        guide_values = _pick(config,
            "tone_preference",
            "history.good_pct",
            "history.top_repeated_sequence",
            "recent.window_min",
            "recent.good_pct",
            "recent.top_label",
            "recent.top_count",
            "recent.top_avg_sec",
            "recent.continuous_bad_sec",
            "recent.repetition_label",
            "params.T_sec",
            "params.N",
            "params.M",
            "params.delta_pct_threshold",
            "params.suggest_stand_break",
            "params.suggest_stand_break_sec",
        )
        prompt_guide = _make_prompt_from_files("guide", guide_values)
        guide_chain = prompt_guide | self.llm | StrOutputParser()
        summary_and_habit_guide = guide_chain.invoke({}).strip()

        # ----- 4) SHORT TERM PLAN -----
        plan_values = _pick(config,
            "tone_preference",
            "previous_output.do_this_now",
            "previous_output.why_this_matters",
            "previous_output.summary_line",
            "previous_output.habit_guard_line",
            "personal.main_discomforts",
            "personal.key_goal",
            "recent.dominant_bad_label",
            "recent.repeated_pattern",
            "recent.good_pct",
            "recent.continuous_bad_sec",
            "recent.repetition_counts",
            "session.minutes_elapsed",
            "session.minutes_remaining",
            "session.anchor_candidates",
            "params.reset_interval_min",
            "params.stand_sec",
            "params.forward_bout_cap_sec",
            "params.wants_habit_break_clause",
            "params.wants_anchor_stretch",
            "prefs.wants_stretch",
            "prefs.allowed_stretches",
        )
        prompt_plan = _make_prompt_from_files("plan", plan_values)
        plan_chain = prompt_plan | self.llm | StrOutputParser()
        short_term_plan = plan_chain.invoke({}).strip()

        return ToDoBundle(
            do_this_now=do_this_now,
            why_this_matters=why_this_matters,
            summary_and_habit_guide=summary_and_habit_guide,
            short_term_plan=short_term_plan,
        )

    def run_analysis_and_update(self) -> Dict[str, Any]:
        """
        GUI 타이머/버튼에서 호출 → 분석/요약 수행 후 To-Do 생성, GUI 키로 반환
        - annotation_supplier()가 있으면 (df_or_path, participant_path) 사용
        - 아니면 annotation_path를 df/participant 동시 사용
        - _summarize_once 내부에서 df 보장(_ensure_df) 및 live/batch_window_minutes 포함 반환
        """
        if self.annotation_supplier:
            df_or_path, participant_path = self.annotation_supplier()
        elif self.annotation_path:
            df_or_path, participant_path = self.annotation_path, self.annotation_path
        else:
            raise RuntimeError("Annotation source not set. Use set_annotation_source().")

        # 요약/지표
        text, metrics = self._summarize_once(df_or_path, participant_path)
        self.posture_summary_text = text
        self.posture_summary_metrics = metrics

        # To-Do (파일 기반 프롬프트 + 공통 입력 사용)
        todo = self._gen_note(metrics, self.preference)
        self.last_todo = todo

        return {
            "Do this now": todo.do_this_now,
            "Why this matters": todo.why_this_matters,
            "Summary and Habit Guard": todo.summary_and_habit_guide,
            "Short Term Plan": todo.short_term_plan,
        }


    def get_profile_for_participant(self) -> dict:
        """
        preferences JSON에서 현재 user_id의 요약된 채팅 프로필을 구성해 반환.
        chat 템플릿의 profile.* 토큰을 채우는 데 사용.
        """
        p = self.preference or {}

        tone_pref = p.get("tone_preference", [])
        chat_interaction = p.get("chat_interaction", "conversational")
        guidance_style = p.get("guidance_style", "unspecified")

        goals = p.get("key_goal") or p.get("goal") or None
        discomforts = p.get("pain_flags") or p.get("discomforts") or []

        config_intentions = p.get("config_intentions", [])
        feedback_modalities = p.get("feedback_modalities", ["text_notification"])
        alert_rule = p.get("alert_rule", "unspecified")
        trust_preference = p.get("trust_preference", {"ai_only_ok": True, "human_checkins_desired": "occasional"})
        privacy_sensing = p.get("privacy_sensing", {})

        return {
            "tone_preference": tone_pref,
            "chat_interaction": chat_interaction,
            "guidance_style": guidance_style,
            "goals": goals,
            "discomforts": discomforts,
            "config_intentions": config_intentions,
            "feedback_modalities": feedback_modalities,
            "alert_rule": alert_rule,
            "trust_preference": trust_preference,
            "privacy_sensing": privacy_sensing,
        }


    def get_previous_outputs_bundle(self) -> dict:
        """
        최근 To-Do 결과(self.last_todo)를 채팅 컨텍스트 형태로 변환.
        chat 템플릿의 previous_outputs.* 토큰을 채우는 데 사용.
        """
        t = self.last_todo
        summary_line = ""
        habit_guard_line = None
        plan_line = ""
        stretch_line = None

        if t and t.summary_and_habit_guide:
            lines = [ln for ln in t.summary_and_habit_guide.split("\n") if ln.strip()]
            if len(lines) >= 1:
                summary_line = lines[0]
            if len(lines) >= 2:
                habit_guard_line = lines[1]

        if t and t.short_term_plan:
            plines = [ln for ln in t.short_term_plan.split("\n") if ln.strip()]
            if len(plines) >= 1:
                plan_line = plines[0]
            if len(plines) >= 2:
                stretch_line = plines[1]

        return {
            "do_this_now": (t.do_this_now if t else ""),
            "why_this_matters": (t.why_this_matters if t else ""),
            "summary_line": summary_line,
            "habit_guard_line": habit_guard_line,
            "session_plan": {
                "plan_line": plan_line,
                "stretch_line": stretch_line,
            },
        }


    def get_history_summary_for_chat(self) -> dict:
        """
        분석 지표(self.posture_summary_metrics)로 채팅 히스토리 컨텍스트 생성.
        chat 템플릿의 history.* 토큰을 채우는 데 사용.
        """
        m = self.posture_summary_metrics or {}
        core = m.get("core") or {}
        dom = m.get("dominance_top") or {}
        motifs = m.get("motif_summary") or {}

        baseline_good_pct = core.get("Good posture time (%)")

        dominant_labels = []
        if dom.get("top_time_label"):
            dominant_labels.append(dom["top_time_label"])
        if dom.get("top_repetition_label") and dom["top_repetition_label"] not in dominant_labels:
            dominant_labels.append(dom["top_repetition_label"])

        l3_top = (motifs.get("L3") or {}).get("most_repeated_sequence")
        l2_top = (motifs.get("L2") or {}).get("most_repeated_sequence")
        top_sequences = [s for s in (l3_top, l2_top) if s]

        return {
            "baseline_good_pct": baseline_good_pct,
            "dominant_labels": dominant_labels,
            "top_sequences": top_sequences,
        }


    def get_runtime_prefs_for_chat(self) -> dict:
        """
        런타임 옵션(스트레치 허용 등). chat 템플릿의 prefs.* 토큰을 채우는 데 사용.
        """
        p = self.preference or {}
        wants_stretch = bool(p.get("wants_stretch", False))
        allowed = p.get("allowed_stretches", ["chest_opener", "neck_release", "hip_hinge_extension"])
        return {
            "wants_stretch": wants_stretch,
            "allowed_stretches": allowed,
        }


    def chat_once(self, user_message: str) -> AIMessage:
        """
        GUI 채팅 입력 → 파일 기반(system/user) 프롬프트 + 요약 메모리로 LLM 호출
        """
        if self.llm is None:
            raise RuntimeError("LLM is not initialized. Call configure_llm(...) first.")
        self.ensure_memory()

        profile = self.get_profile_for_participant()
        prev_outputs = self.get_previous_outputs_bundle()
        history = self.get_history_summary_for_chat()
        prefs = self.get_runtime_prefs_for_chat()

        def _flatten(prefix: str, obj, out: Dict[str, Any]):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _flatten(f"{prefix}.{k}" if prefix else k, v, out)
            else:
                out[prefix] = obj
            return out

        user_values: Dict[str, Any] = {"question": user_message}
        _flatten("profile", profile, user_values)
        _flatten("previous_outputs", prev_outputs, user_values)
        _flatten("history", history, user_values)
        _flatten("prefs", prefs, user_values)

        prompt = _make_prompt_from_files("chat", user_values)

        prior = self.memory.load_memory_variables({}).get("history", [])
        messages = []
        if isinstance(prior, list):
            messages.extend(prior)
        elif prior:
            messages.append(SystemMessage(content=str(prior)))
        messages.extend(prompt.format_messages())

        resp = self.llm.invoke(messages)
        ai_text = getattr(resp, "content", str(resp)).strip()

        lines = [ln.strip() for ln in ai_text.splitlines() if ln.strip()]
        final = "\n".join(lines)

        self.memory.save_context(inputs={"human": user_message}, outputs={"ai": final})

        return AIMessage(content=final)

_PIPE = _Pipeline()

# ================= GUI-facing wrapper functions ================= # To Hyun-jin : 여기 쓰면 됩니다.
def configure_llm(llm=None, *, openai_model: Optional[str] = None, temperature: float = 0.3):
    """
    LLM 설정
    - llm: LangChain ChatModel 객체 직접 주입
    - openai_model: "gpt-4o-mini" 등 모델명으로 생성 (temperature 반영)
    """
    global _PIPE
    if llm is not None:
        _PIPE.set_llm(llm)
    elif openai_model is not None:
        _PIPE.set_llm(ChatOpenAI(model=openai_model, temperature=temperature))
    elif _PIPE.llm is None:
        raise RuntimeError("LLM is not configured. Provide llm or openai_model.")

def load_preferences_json(json_path: str):
    """
    Preferences JSON 파일 경로를 바꿔서 로드.
    최상위 키는 user_id, 값은 해당 사용자의 preference dict.
    """
    global _PIPE
    _PIPE.reload_preferences_from_json(json_path)

def select_user(user_id: str):
    """
    현재 세션의 타겟 유저 전환.
    (GUI에서 드롭다운/탭 변경 시 호출)
    """
    global _PIPE
    _PIPE._apply_user_preference(user_id)

def set_user_preference(pref: Dict[str, Any]):
    """
    런타임 임시 오버라이드. JSON에 저장되지는 않음.
    """
    global _PIPE
    _PIPE.set_preferences(pref)

def set_annotation_source(path_or_supplier):
    """
    - str 경로를 주면 그 파일(xlsx/csv)을 사용
    - 콜러블을 주면 매 호출 시 (df, optional_participant_path)를 반환해야 함
      * df: pandas.DataFrame 또는 xlsx/csv 경로 문자열
      * optional_participant_path: 패턴 분석용 xlsx 경로(없으면 None)
    """
    global _PIPE
    if isinstance(path_or_supplier, str):
        _PIPE.set_annotation_path(path_or_supplier)
    elif callable(path_or_supplier):
        _PIPE.set_annotation_supplier(path_or_supplier)
    else:
        raise ValueError("set_annotation_source expects a str path or a callable supplier().")

def get_analysis_data() -> Dict[str, Any]:
    """
    GUI LLMWorker.run_analysis()에서 호출.
    반환 딕셔너리 키는 GUI의 update_analysis_posture()가 기대하는 이름으로 맞춤.
    """
    try:
        return _PIPE.run_analysis_and_update()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

def get_chat_response(user_message: str, _chat_history_from_gui=None) -> AIMessage | Dict[str, Any]:
    """
    GUI LLMWorker.run_chat()에서 호출.
    반환은 AIMessage (실패 시 {"error": "..."} 딕셔너리)
    """
    try:
        return _PIPE.chat_once(user_message)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

### CODE SNIPPET FOR HYUNJIN ###
# from posture_pipeline import (
#     configure_llm, load_preferences_json, select_user, set_annotation_source,
#     get_analysis_data, get_chat_response,
#     select_user, set_user_preference,
#     set_annotation_source,
#     configure_llm
# )


# def app_boot():
#     # 1) LLM 설정
#     configure_llm(openai_model="gpt-4o-mini", temperature=0.3)

#     # 2) 환경설정 JSON 로드
#     load_preferences_json("./data/user_preferences.json")

#     # 3) 초기 사용자 선택
#     select_user("default_user")

#     # 4) 데이터 소스 지정 (파일 경로 or supplier 콜러블)
#     #   - 파일 경로로 지정할 때:
#     set_annotation_source("./data/sample_annotations.xlsx")

#     #   - 또는 콜러블로 지정할 때(선택):
#     # def annotation_supplier():
#     #     # (DataFrame 또는 경로, optional_participant_path) 튜플
#     #     return "./data/sample_annotations.xlsx", "./data/sample_annotations.xlsx"
#     # set_annotation_source(annotation_supplier)

# def on_click_analyze():
#     result = get_analysis_data()
#     if "error" in result:
#         gui.show_error(result["error"])
#         return

#     # GUI 반영 (원하는 위젯 id/함수에 맞게 연결)
#     gui.set_text("do_now_label",       result["Do this now"])
#     gui.set_text("why_label",          result["Why this matters"])
#     gui.set_text("summary_label",      result["Summary and Habit Guard"])
#     gui.set_text("plan_label",         result["Short Term Plan"])

# def on_click_send_chat():
#     user_msg = gui.get_text("chat_input")
#     resp = get_chat_response(user_msg)

#     if isinstance(resp, dict) and "error" in resp:
#         gui.show_error(resp["error"])
#         return

#     # resp는 AIMessage
#     gui.append_chat(role="user", text=user_msg)
#     gui.append_chat(role="assistant", text=resp.content)
#     gui.clear("chat_input")

# def on_user_changed(new_user_id: str):
#     select_user(new_user_id)


# def on_preferences_changed(prefs: dict):
#     """
#     prefs 예시:
#     {
#       "tone_preference": ["motivational_encouraging"],
#       "guidance_style": "facilitative",
#       "wants_stretch": True,
#       "allowed_stretches": ["chest_opener","neck_release"],
#       "params_T_sec": 90,
#       ...
#     }
#     """
#     set_user_preference(prefs)


# def on_pick_annotation_file(path: str):
#     # xlsx/csv 파일 경로
#     set_annotation_source(path)
#     # 필요시 즉시 재분석
#     # on_click_analyze()

# def on_model_changed(model_name: str, temperature: float):
#     configure_llm(openai_model=model_name, temperature=temperature)
#############################################################################################