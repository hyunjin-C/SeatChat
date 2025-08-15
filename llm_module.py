from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import importlib.util
from models import ToDoBundle

BEHAVIOR_PATH = str(Path("./scripts/posture_behavior_analysis_all.py"))
PATTERN_PATH  = str(Path("./scripts/posture_pattern_all_repeat_time.py"))
RANKING_PATH  = str(Path("./scripts/posture_ranking_all.py"))

PREFERENCES_JSON_PATH = Path("./data/user_preferences.json")

_DEFAULT_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def _load_module(name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

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

        # 분석 모듈 로드
        self.m_behavior = _load_module("posture_behavior_analysis_all", BEHAVIOR_PATH)
        self.m_pattern  = _load_module("posture_pattern_all_repeat_time", PATTERN_PATH)
        self.m_ranking  = _load_module("posture_ranking_all", RANKING_PATH)

        # Preferences
        self._preferences_store: Dict[str, Dict[str, Any]] = {}
        self.preference: Dict[str, Any] = {}
        self._load_preferences_store(PREFERENCES_JSON_PATH)
        self._apply_user_preference(self.user_id)

        # 최신 결과 보관
        self.posture_summary_text: str = ""
        self.posture_summary_metrics: Dict[str, Any] = {}
        self.last_todo: Optional[ToDoBundle] = None

        # 데이터 소스: 경로 또는 공급자 콜러블
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
        """fn() -> (df_or_path, optional_participant_path)"""
        self.annotation_supplier = fn

    def _ensure_df(self, data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        p = Path(str(data))
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")
        if p.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(p)
        elif p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        else:
            raise ValueError("Unsupported input type. Use xlsx/csv/DataFrame.")

    def _summarize_once(self, df_or_path: Any, participant_path: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        df = self._ensure_df(df_or_path)

        core = self.m_behavior.compute_summary_row(df)

        summary_df = pd.DataFrame([core])
        top_df, _ranks_df = self.m_ranking.build_dominance_tables(summary_df)

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
                try: temp_path.unlink()
                except Exception: pass

        good_pct = core.get("Good posture time (%)")
        top_time_label = top_df.iloc[0]["top_time_label"] if not top_df.empty else None
        top_rep_label  = top_df.iloc[0]["top_repetition_label"] if not top_df.empty else None
        l3_seq = row.get("L3_most_time_sequence") or row.get("L3_most_repeated_sequence")
        l2_seq = row.get("L2_most_time_sequence") or row.get("L2_most_repeated_sequence")

        text = (
            f"좋은 자세 유지 비율: {good_pct}% | "
            f"시간 지배 자세: {top_time_label} | 반복 지배 자세: {top_rep_label} | "
            f"L3 대표 패턴: {l3_seq} | L2 대표 패턴: {l2_seq}"
        )
        metrics = {
            "core": core,
            "dominance_top": top_df.to_dict(orient="records")[0] if not top_df.empty else {},
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
                }
            }
        }
        return text, metrics

    def _gen_note(self, summary_text: str, metrics: Dict[str, Any], preference: Dict[str, Any]) -> ToDoBundle:
        if self.llm is None:
            raise RuntimeError("LLM is not initialized. Call configure_llm(...) first.")

        # ---------- 공통 입력 정리 ----------
        core = (metrics or {}).get("core", {})
        dom  = (metrics or {}).get("dominance_top", {})
        live = (metrics or {}).get("live", {})  # 있을 경우 사용

        tone_pref  = preference.get("tone_preference", []) if isinstance(preference, dict) else []
        guidance   = preference.get("guidance_style", "unspecified") if isinstance(preference, dict) else "unspecified"
        pain_flags = preference.get("pain_flags", []) if isinstance(preference, dict) else []
        key_goal   = preference.get("key_goal") if isinstance(preference, dict) else None
        improvement_timeframe = preference.get("improvement_timeframe", "4-6 weeks") if isinstance(preference, dict) else "4-6 weeks"

        # 최근/지배 라벨 값 추정
        current_label      = (live.get("current_label")
                              or dom.get("top_time_label")
                              or "FORWARD")
        continuous_sec     = (live.get("continuous_seconds_current_label") or 0)
        batch_window_min   = metrics.get("batch_window_minutes", 10) if isinstance(metrics, dict) else 10
        top_time_label     = dom.get("top_time_label", "FORWARD")
        top_time_minutes   = dom.get("top_time_minutes", 0)
        top_rep_label      = dom.get("top_repetition_label", "FORWARD")
        top_rep_count      = dom.get("top_repetition_count", 0)
        good_minutes       = core.get("Good posture time (min)", 0)
        bad_minutes        = core.get("Bad posture time (min)", 0)
        good_pct           = core.get("Good posture time (%)", 0)
        bad_pct            = 100 - good_pct if isinstance(good_pct, (int, float)) else 0


        # ---------- 1) Do This Now ----------
        sys_do = """You are a posture coach. Return ONE line only. Keep it short and clear.

        Rules
        - Output ONE line, max ~20 words. No emojis. No extra text.
        - End with a duration in parentheses: (20s), (30s), or (45s).
        - Structure: {action_verb} + {body_region} + {how} + {duration}.
        Example format: “Reset now: sit back; ears over shoulders; relax jaw (30s).”
        - Tone:
        • If tone_preference contains "authoritative_strict" → strict.
        • Else if it contains "motivational_encouraging" → motivational.
        • Else → neutral/informational (default).
        • If guidance_style == "facilitative", you may use a soft imperative or a
            question-like nudge (still ONE line, with duration).
        - Target the current problem using the LIVE posture label.
        Prefer left/right specific cues when relevant.
        - Choose duration by severity (use the longest rule that applies):
        1) If continuous_seconds_current_label ≥ 120 OR top_repetition_count ≥ 3
            (in last 20 min) → (45s)
        2) Else if bad_pct ≥ 80 in current batch window → (30s)
        3) Else → (20s)
        - If current_label == "UPRIGHT": give a short maintenance line, e.g.,
        “Hold neutral; relax shoulders; easy breaths (20s).”
        - Safety: no diagnoses or medical claims. Plain language only.

        Label set (9)
        - Good: UPRIGHT
        - Non-neutral: UPRIGHT_LEFT, UPRIGHT_RIGHT, FORWARD, FORWARD_LEFT, FORWARD_RIGHT,
                    BACK, BACK_LEFT, BACK_RIGHT

        Directional micro‑resets (map label → how)
        - FORWARD:        sit back; ears over shoulders; lengthen neck
        - FORWARD_LEFT:   sit back, shift slightly right; even hips
        - FORWARD_RIGHT:  sit back, shift slightly left; even hips
        - BACK (recline): come forward; ribs over hips; hinge at hips
        - BACK_LEFT:      come forward and center; level shoulders
        - BACK_RIGHT:     come forward and center; level shoulders
        - UPRIGHT_LEFT:   re‑center from left tilt; drop right shoulder; even weight
        - UPRIGHT_RIGHT:  re‑center from right tilt; drop left shoulder; even weight
        - UPRIGHT:        hold neutral; relax shoulders; easy breaths

        Fallbacks
        - If tone_preference is empty → use neutral tone.
        - If label is unknown → treat as FORWARD.
        - If batch stats are missing → use (20s).
        Return ONE line only.
        """

        human_do = """tone_preference: {tone_preference}
        guidance_style: {guidance_style}
        live.current_label: {current_label}
        live.continuous_seconds_current_label: {continuous_seconds_current_label}
        batch_window_minutes: {batch_window_minutes}
        batch.top_time_label: {top_time_label}
        batch.top_time_minutes: {top_time_minutes}
        batch.top_repetition_label: {top_repetition_label}
        batch.top_repetition_count: {top_repetition_count}
        batch.good_minutes: {good_minutes}
        batch.bad_minutes: {bad_minutes}
        batch.good_pct: {good_pct}
        batch.bad_pct: {bad_pct}
        personal.main_discomforts: {main_discomforts}
        personal.key_goal: {key_goal}
        Return only ONE line as instructed.
        """

        prompt_do = ChatPromptTemplate.from_messages([
            ("system", sys_do),
            ("human", human_do),
        ])

        inputs_common = {
            "tone_preference": json.dumps(tone_pref, ensure_ascii=False),
            "guidance_style": guidance,
            "current_label": current_label,
            "continuous_seconds_current_label": continuous_sec,
            "batch_window_minutes": batch_window_min,
            "top_time_label": top_time_label,
            "top_time_minutes": top_time_minutes,
            "top_repetition_label": top_rep_label,
            "top_repetition_count": top_rep_count,
            "good_minutes": good_minutes,
            "bad_minutes": bad_minutes,
            "good_pct": good_pct,
            "bad_pct": bad_pct,
            "main_discomforts": json.dumps(pain_flags, ensure_ascii=False),
            "key_goal": key_goal or "null",
            "improvement_timeframe": improvement_timeframe,
        }

        do_chain = prompt_do | self.llm | StrOutputParser()
        do_this_now = do_chain.invoke(inputs_common).strip()
        if "\n" in do_this_now:
            do_this_now = do_this_now.splitlines()[0].strip()

        # ---------- 2) Why This Matters ----------
        sys_why = """You are a posture coach. Return ONE sentence only. Keep it friendly and clear.

        Rules
        - Output exactly ONE sentence, ~12-22 words. No emojis. No extra text.
        - Content shape: {personal_goal_or_discomfort} + {short benefit} + (optional {timeframe})
        AND include a brief reference to the latest posture issue
        (live current label and/or dominant bad label from the last batch).
        - Be positive and non-threatening. Use phrases like “can ease,” “helps,” “supports.”
        Do NOT use fear or harm language (“damage,” “injury,” “risk of …”).
        - Tone:
        • If tone_preference contains "motivational_encouraging" → encouraging wording.
        • Else → neutral/informational tone.
        • Ignore “authoritative_strict” here (no threats); keep it factual or encouraging.
        - Use label-aware phrasing when possible (examples below). Keep it short.
        - If personal info is missing, use a neutral benefit (“stay comfortable and focused”).
        - If timeframe is provided, include it; otherwise omit it.
        - If current_label == "UPRIGHT", reference the most recent bad label from the batch.

        Label-aware micro-phrases (pick one that fits the evidence)
        - FORWARD / FORWARD_LEFT / FORWARD_RIGHT → “forward lean can tense the neck/shoulders”
        - BACK / BACK_LEFT / BACK_RIGHT → “recline/slouch can load the lower back”
        - UPRIGHT_LEFT / UPRIGHT_RIGHT → “tilt can load one side more than the other”

        Evidence phrasing
        - Use compact parentheticals: e.g., “(recent FORWARD 32×/10 min)” or “(BACK 19.5 min)”.

        Return ONE sentence only.
        """

        human_why = """tone_preference: {tone_preference}
        live.current_label: {current_label}
        batch_window_minutes: {batch_window_minutes}
        batch.top_time_label: {top_time_label}
        batch.top_time_minutes: {top_time_minutes}
        batch.top_repetition_label: {top_repetition_label}
        batch.top_repetition_count: {top_repetition_count}
        batch.good_minutes: {good_minutes}
        batch.bad_minutes: {bad_minutes}
        batch.good_pct: {good_pct}
        batch.bad_pct: {bad_pct}
        personal.main_discomforts: {main_discomforts}
        personal.key_goal: {key_goal}
        personal.improvement_timeframe: {improvement_timeframe}
        Return only ONE line as instructed.
        """
        prompt_why = ChatPromptTemplate.from_messages([
            ("system", sys_why),
            ("human", human_why),
        ])
        why_chain = prompt_why | self.llm | StrOutputParser()
        why_this_matters = why_chain.invoke({
            "tone_preference": json.dumps(tone_pref, ensure_ascii=False),
            "current_label": current_label,
            "batch_window_minutes": batch_window_min,
            "top_time_label": top_time_label,
            "top_time_minutes": top_time_minutes,
            "top_repetition_label": top_rep_label,
            "top_repetition_count": top_rep_count,
            "good_minutes": good_minutes,
            "bad_minutes": bad_minutes,
            "good_pct": good_pct,
            "bad_pct": bad_pct,
            "main_discomforts": json.dumps(pain_flags, ensure_ascii=False),
            "key_goal": key_goal or "null",
            "improvement_timeframe": improvement_timeframe,
        }).strip()

        # ---------- 3) Summary & Habit Guide ----------
        sys_guide = """You are a posture coach. Output one or two lines only, as described.

        General rules
        - Maximum: 2 lines. If no pattern trigger fires, return only Line 1.
        - Be clear, brief, and neutral/motivational (no fear language, no emojis).
        - Numbers: round percentages to whole numbers; minutes to 1 decimal; seconds to nearest 5s.
        - Labels are from this set: UPRIGHT (good), UPRIGHT_LEFT, UPRIGHT_RIGHT, FORWARD, FORWARD_LEFT, FORWARD_RIGHT, BACK, BACK_LEFT, BACK_RIGHT.

        ------------------------------------
        LINE 1 — Summary (always present)
        Content format:
        Last {recent.window_min} min: neutral {recent.good_pct}% (vs baseline {delta.good_pct});
        top: {label1} ×{count1} (avg {avg_s1}s){, label2 ×{count2} (avg {avg_s2}s)}.

        Rules
        - Use the recent window minutes and recent percentage of good posture.
        - Compare with personal baseline (percentage of good posture from history) and show a brief delta:
        • If recent.good_pct >= history.good_pct + 3 → write "vs baseline ↑+{diff}%"
        • If recent.good_pct <= history.good_pct - 3 → write "vs baseline ↓−{diff}%"
        • Else → write "vs baseline ≈"
        - Choose up to two non-neutral labels from the recent batch:
        • Primary: the label with the longest time or highest count (prefer the one with longer time).
        • Secondary: the next most prominent label if it adds information.
        - Show each label with count and average episode duration in seconds.

        ------------------------------------
        LINE 2 — Habit Guard (only if triggered)
        Purpose: Break repeating bad patterns by referencing the user's historical patterns AND
        the current window. Keep it firm but supportive.

        Trigger (fire if ANY is true)
        - recent.continuous_bad_sec >= params.T_sec
        - OR recent.repetition[label] >= params.N within params.M minutes
        - OR recent.most_repeated_sequence == history.top_repeated_sequence (label sequence match)
        - OR (recent.good_pct ≤ history.good_pct − params.delta_pct_threshold)

        Content format (choose tone: neutral/motivational; strict only if tone_preference contains "authoritative_strict"):
        Pattern repeating: {pattern_text}. {micro_break} then {micro_reset}.

        Where
        - {pattern_text} compactly names what’s repeating, e.g.:
        • "forward ×{n}/{M}min"  OR  "BACK 3× in a row" OR "FORWARD→BACK repeating"
        - {micro_break} = "Stand {stand_sec}s" if params.suggest_stand_break is true (default 45s), else "Pause {pause_sec}s"
        - {micro_reset} is a direction-aware cue for the most problematic current label:
            FORWARD: sit back; ears over shoulders
            FORWARD_LEFT: sit back, shift slightly right; even hips
            FORWARD_RIGHT: sit back, shift slightly left; even hips
            BACK: come forward; ribs over hips
            BACK_LEFT: come forward and center; level shoulders
            BACK_RIGHT: come forward and center; level shoulders
            UPRIGHT_LEFT: re-center from left tilt; drop right shoulder
            UPRIGHT_RIGHT: re-center from right tilt; drop left shoulder
        - Keep Line 2 ≤ 18–22 words.

        If no trigger fires, omit Line 2.

        ------------------------------------
        TONE
        - If tone_preference contains "motivational_encouraging": use encouraging verbs ("let’s break it", "you’re building a streak").
        - If tone_preference contains "authoritative_strict": use firm imperatives ("Stop.", "Stand 45s, then reset.").
        - Otherwise: neutral/informational tone.

        ------------------------------------
        OUTPUT
        - Return either:
        • ONE line (Summary) or
        • TWO lines (Summary + Habit Guard), separated by a newline.
        - No extra commentary.
        """

        human_guide = """tone_preference: {tone_preference}
        history.good_pct: {history_good_pct}
        history.top_repeated_sequence: {history_top_repeated_sequence}
        recent.window_min: {recent_window_min}
        recent.good_pct: {recent_good_pct}
        recent.top_label: {recent_top_label}
        recent.top_count: {recent_top_count}
        recent.top_avg_sec: {recent_top_avg_sec}
        recent.continuous_bad_sec: {recent_continuous_bad_sec}
        recent.repetition[label]: {recent_repetition_label}
        recent.most_repeated_sequence: {recent_most_repeated_sequence}
        recent.most_repeated_count: {recent_most_repeated_count}
        params.T_sec: {params_T_sec}
        params.N: {params_N}
        params.M: {params_M}
        params.delta_pct_threshold: {params_delta_pct_threshold}
        params.suggest_stand_break: {params_suggest_stand_break}
        params.suggest_stand_break_sec: {params_suggest_stand_break_sec}

        # The model will format:
        # Line 1: summary + delta vs baseline + top labels with counts/avg seconds.
        # Line 2: only if triggered; compact pattern text + micro-break + direction-aware micro-reset.
        """
        history_good_pct = preference.get("history_good_pct", good_pct) or good_pct
        history_top_repeated_sequence = (metrics.get("motif_summary", {})
                                              .get("L3", {})
                                              .get("most_repeated_sequence")) or \
                                        (metrics.get("motif_summary", {})
                                                .get("L2", {})
                                                .get("most_repeated_sequence")) or "N/A"

        recent_window_min = batch_window_min
        recent_good_pct = good_pct
        recent_top_label = top_time_label if top_time_label not in ("UPRIGHT", None) else top_rep_label
        recent_top_count = int(top_rep_count or 0)
        recent_top_avg_sec = int(round(((top_time_minutes or 0) * 60) / max(recent_top_count, 1) / 5) * 5)
        recent_continuous_bad_sec = int(round((continuous_sec or 0) / 5) * 5)
        recent_repetition_label = top_rep_label
        recent_most_repeated_sequence = history_top_repeated_sequence
        recent_most_repeated_count = int((metrics.get("motif_summary", {})
                                                .get("L3", {})
                                                .get("most_repeated_count"))
                                         or (metrics.get("motif_summary", {})
                                                     .get("L2", {})
                                                     .get("most_repeated_count"))
                                         or 0)

        # 파라미터 기본값 (필요시 preference로 오버라이드 가능)
        params_T_sec = int(preference.get("params_T_sec", 90))
        params_N = int(preference.get("params_N", 3))
        params_M = int(preference.get("params_M", recent_window_min))
        params_delta_pct_threshold = int(preference.get("params_delta_pct_threshold", 5))
        params_suggest_stand_break = bool(preference.get("params_suggest_stand_break", True))
        params_suggest_stand_break_sec = int(preference.get("params_suggest_stand_break_sec", 45))

        prompt_guide = ChatPromptTemplate.from_messages([
            ("system", sys_guide),
            ("human", human_guide),
        ])
        guide_chain = prompt_guide | self.llm | StrOutputParser()
        summary_and_habit_guide = guide_chain.invoke({
            "tone_preference": json.dumps(tone_pref, ensure_ascii=False),
            "history_good_pct": history_good_pct,
            "history_top_repeated_sequence": history_top_repeated_sequence,
            "recent_window_min": recent_window_min,
            "recent_good_pct": recent_good_pct,
            "recent_top_label": recent_top_label or "FORWARD",
            "recent_top_count": recent_top_count,
            "recent_top_avg_sec": recent_top_avg_sec,
            "recent_continuous_bad_sec": recent_continuous_bad_sec,
            "recent_repetition_label": recent_repetition_label or "FORWARD",
            "recent_most_repeated_sequence": recent_most_repeated_sequence,
            "recent_most_repeated_count": recent_most_repeated_count,
            "params_T_sec": params_T_sec,
            "params_N": params_N,
            "params_M": params_M,
            "params_delta_pct_threshold": params_delta_pct_threshold,
            "params_suggest_stand_break": str(params_suggest_stand_break),
            "params_suggest_stand_break_sec": params_suggest_stand_break_sec,
        }).strip()

        # ---------- 4) Short Term Plan ----------
        sys_plan = """You are a posture coach. Return one or two lines only.
        Style & Safety
        - Clear, plain language. No fear terms or medical claims.
        - Short and scannable. Prefer “Plan:” / “Stretch:” prefixes.
        - If stretches are included, append: “General wellness advice; stop if you feel pain.”

        Output format
        - Always print **Line 1 (Plan)**.
        - Print **Line 2 (Stretch)** only if `prefs.wants_stretch` is true AND a stretch is allowed.
        - Do **not** invent times: only use anchors provided in input.
        - Keep each line ≤ ~22 words.

        Line 1 — Plan
        - Compose from these components in order:
        1) micro-goals (pick 1–2) based on recent problems and personal goals, e.g.:
            • FORWARD-heavy → “keep forward-lean bouts <60s”
            • BACK-heavy → “limit recline bouts <60s”
            • UPRIGHT tilt → “re-center tilt during each reset”
        2) simple rule (choose one):
            • timed resets: “30-sec reset every {interval} min”
            • one stand break: “add a {stand_sec}-sec stand break at {anchor}”
        3) time anchors: pick 1–2 from `session.anchor_candidates` that fit remaining time.
        - If a Habit Guard trigger is active, include a brief “break the {pattern} loop” clause.
        - Respect tone:
        • motivational_encouraging → “Let’s…”, “You’re building a streak”
        • authoritative_strict → firm imperative without threats
        • otherwise neutral/informational

        Line 2 — Stretch (optional)
        - Include at most one stretch set, only if `prefs.wants_stretch` is true.
        - Choose a stretch that matches discomfort/goal:
        • neck/shoulders → “2‑min chest opener” or “1‑min neck release”
        • lower back/hips → “2‑min hip hinge + gentle extension”
        - Place at one anchor; include duration; append the safety note.

        Micro-reset mapping (for brief mentions when needed)
        - FORWARD: sit back; ears over shoulders
        - FORWARD_LEFT: sit back, shift slightly right; even hips
        - FORWARD_RIGHT: sit back, shift slightly left; even hips
        - BACK: come forward; ribs over hips
        - UPRIGHT_LEFT: re-center from left tilt
        - UPRIGHT_RIGHT: re-center from right tilt

        Return exactly the lines (1 or 2), separated by a newline. No extra commentary.
    """
        human_plan = """tone_preference: {tone_preference}
        previous_output: {previous_output}
        personal.main_discomforts: {main_discomforts}
        personal.key_goal: {key_goal}
        recent.dominant_bad_label: {recent_dominant_bad_label}
        recent.repeated_pattern: {recent_repeated_pattern}
        recent.good_pct: {recent_good_pct}
        recent.continuous_bad_sec: {recent_continuous_bad_sec}
        recent.repetition_counts: {recent_repetition_counts}
        session.minutes_elapsed: {session_minutes_elapsed}
        session.minutes_remaining: {session_minutes_remaining}
        session.anchor_candidates: {session_anchor_candidates}
        params.reset_interval_min: {params_reset_interval_min}
        params.stand_sec: {params_stand_sec}
        params.forward_bout_cap_sec: {params_forward_bout_cap_sec}
        params.wants_habit_break_clause: {params_wants_habit_break_clause}
        params.wants_anchor_stretch: {params_wants_anchor_stretch}
        prefs.wants_stretch: {prefs_wants_stretch}
        prefs.allowed_stretches: {prefs_allowed_stretches}

        # The model will produce:
        # Line 1: session plan with 1–2 micro-goals + a simple rule + anchors.
        # Line 2: stretch cue (only if prefs.wants_stretch is true and aligned with discomfort/goal).
        # Keep output to 1–2 lines, easy to read.
        """
        previous_output = f"Do: {do_this_now} | Why: {why_this_matters} | Guide: {summary_and_habit_guide}"
        recent_dominant_bad_label = recent_top_label or "FORWARD"
        recent_repeated_pattern = recent_most_repeated_sequence
        recent_repetition_counts = {"label": recent_repetition_label or "FORWARD", "count": recent_top_count}

        # 세션/앵커 기본값 (필요시 preference에서 오버라이드)
        session_minutes_elapsed = int(preference.get("session_minutes_elapsed", 0))
        session_minutes_remaining = int(preference.get("session_minutes_remaining", 60))
        session_anchor_candidates = preference.get("session_anchor_candidates",
                                                   ["start of next focus", "after meeting", "top of hour"])

        params_reset_interval_min = int(preference.get("params_reset_interval_min", max(20, batch_window_min)))
        params_stand_sec = int(preference.get("params_stand_sec", 45))
        params_forward_bout_cap_sec = int(preference.get("params_forward_bout_cap_sec", 60))
        params_wants_habit_break_clause = bool(preference.get("params_wants_habit_break_clause", True))
        params_wants_anchor_stretch = bool(preference.get("params_wants_anchor_stretch", True))

        prefs_wants_stretch = bool(preference.get("wants_stretch", False))
        prefs_allowed_stretches = preference.get("allowed_stretches",
                                                 ["chest opener 2m", "neck release 1m", "hip hinge + gentle extension 2m"])

        prompt_plan = ChatPromptTemplate.from_messages([
            ("system", sys_plan),
            ("human", human_plan),
        ])
        plan_chain = prompt_plan | self.llm | StrOutputParser()
        short_term_plan = plan_chain.invoke({
            "tone_preference": json.dumps(tone_pref, ensure_ascii=False),
            "previous_output": previous_output,
            "main_discomforts": json.dumps(pain_flags, ensure_ascii=False),
            "key_goal": key_goal or "없음",
            "recent_dominant_bad_label": recent_dominant_bad_label,
            "recent_repeated_pattern": recent_repeated_pattern,
            "recent_good_pct": recent_good_pct,
            "recent_continuous_bad_sec": recent_continuous_bad_sec,
            "recent_repetition_counts": json.dumps(recent_repetition_counts, ensure_ascii=False),
            "session_minutes_elapsed": session_minutes_elapsed,
            "session_minutes_remaining": session_minutes_remaining,
            "session_anchor_candidates": json.dumps(session_anchor_candidates, ensure_ascii=False),
            "params_reset_interval_min": params_reset_interval_min,
            "params_stand_sec": params_stand_sec,
            "params_forward_bout_cap_sec": params_forward_bout_cap_sec,
            "params_wants_habit_break_clause": str(params_wants_habit_break_clause),
            "params_wants_anchor_stretch": str(params_wants_anchor_stretch),
            "prefs_wants_stretch": str(prefs_wants_stretch),
            "prefs_allowed_stretches": json.dumps(prefs_allowed_stretches, ensure_ascii=False),
        }).strip()

        return ToDoBundle(
            do_this_now=do_this_now,
            why_this_matters=why_this_matters,
            summary_and_habit_guide=summary_and_habit_guide,
            short_term_plan=short_term_plan,
        )

    # ----------------- 공개 API ----------------- 
    def run_analysis_and_update(self) -> Dict[str, Any]:
        """
        GUI 타이머/버튼에서 호출 → 분석/요약 수행 후 To-Do 생성, GUI 키로 반환
        """
        # 데이터 소스 확보
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

        # To-Do
        todo = self._gen_note(text, metrics, self.preference)
        self.last_todo = todo

        return {
            "Do this now": todo.do_this_now,
            "Why this matters": todo.why_this_matters,
            "Summary and Habit Guard": todo.summary_and_habit_guide,
            "Short Term Plan": todo.short_term_plan,
        }

    def chat_once(self, user_message: str) -> AIMessage:
        """
        GUI의 채팅 입력을 받아 LLM 답변(AIMessage) 반환
        """
        if self.llm is None:
            raise RuntimeError("LLM is not initialized. Call configure_llm(...) first.")
        self.ensure_memory()

        sys = SystemMessage(content=(
            "당신은 개인 자세 코치입니다. 간결하고 실행가능한 한국어 답변을 선호합니다.\n\n"
            f"[최신 자세 요약]\n{self.posture_summary_text}\n\n"
            f"[최신 To-Do(JSON)]\n{json.dumps(self.last_todo.dict() if self.last_todo else {}, ensure_ascii=False)}\n\n"
            f"[고정 Preference(JSON)]\n{json.dumps(self.preference, ensure_ascii=False)}\n"
            "과도한 의학적 주장/진단은 피하고, 실행 단계/트리거/환경 단서를 명시하세요."
        ))
        human = HumanMessage(content=user_message)

        hist = self.memory.load_memory_variables({}).get("history", [])
        messages = [sys]
        if isinstance(hist, list):
            messages.extend(hist)
        elif hist:
            messages.append(SystemMessage(content=str(hist)))
        messages.append(human)

        resp = self.llm.invoke(messages)
        ai_text = getattr(resp, "content", str(resp))
        self.memory.save_context({"input": user_message}, {"output": ai_text})
        return AIMessage(content=ai_text)

_PIPE = _Pipeline()

# ================= 외부 공개 함수 (GUI에서 호출) ================= # To Hyun-jin : 여기 쓰면 됩니다.

def configure_llm(llm=None, *, openai_model: Optional[str]=None, temperature: float=0.3):
    """
    LLM 설정 (선택)
    - llm: LangChain ChatModel 객체 직접 주입
    - openai_model: "gpt-4o-mini" 등 모델명으로 생성
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
    - 콜러블을 주면 매 호출 시 (df_or_path, optional_participant_path)를 반환해야 함
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
    반환은 AIMessage
    """
    try:
        return _PIPE.chat_once(user_message)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
