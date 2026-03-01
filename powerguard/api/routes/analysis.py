"""
api/routes/analysis.py
======================
POST /v1/analyze — the main analysis endpoint.

Flow:
  1. Validate multipart input (video file + JSON body)
  2. Fetch user data via DataProvider
  3. Run CV analyzer (bench/deadlift/squat) in threadpool
  4. Summarize biomechanics for LLM
  5. Compute physics-based stress, fatigue, recovery indices
  6. Run LangGraph agent pipeline
  7. Build and return AnalysisResponse
"""

import json
import os
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.concurrency import run_in_threadpool

from agents.graph import run_pipeline
from config.settings import get_settings
from data.mock_provider import MockDataProvider
from engines.fatigue_engine import compute_all_fatigue_metrics
from engines.recovery_engine import compute_recovery_score
from engines.stress_engine import compute_stress_indices, summarize_biomechanics
from observability.logger import get_logger, trace_id_var, generate_trace_id
from schemas.agent_state import AgentState
from schemas.request import AnalysisRequest, LiftType
from schemas.response import AnalysisResponse, AgentTrace

logger = get_logger("analysis_route")
router = APIRouter(prefix="/v1", tags=["Analysis"])

# DataProvider — swap MockDataProvider → SupabaseDataProvider when DB tables are ready
_data_provider = MockDataProvider()

# Allowed video extensions
_ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def _validate_video_file(file: UploadFile) -> None:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported video format '{ext}'. Allowed: {_ALLOWED_EXTENSIONS}",
        )
    if file.size and file.size > 500 * 1024 * 1024:  # 500 MB limit
        raise HTTPException(status_code=413, detail="Video file exceeds 500 MB limit.")


def _run_cv_analyzer(lift_type: str, video_path: str) -> dict:
    """
    Call the appropriate lift analyzer from biomechanics-sbd.
    Runs synchronously in a threadpool (OpenCV/MediaPipe are CPU-bound).
    Adds biomechanics-sbd to sys.path temporarily.
    """
    settings  = get_settings()
    bio_path  = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", settings.biomechanics_module_path)
    )

    # Temporarily add biomechanics-sbd directory so we can import its modules
    sys.path.insert(0, bio_path)
    try:
        if lift_type == LiftType.bench_press:
            from bench_press_analyzer import analyze
        elif lift_type == LiftType.deadlift:
            from deadlift_analyzer import analyze
        elif lift_type == LiftType.squat:
            from squat_analyzer import analyze
        else:
            raise ValueError(f"Unknown lift_type: {lift_type}")

        result = analyze(video_path)
        return result
    finally:
        if bio_path in sys.path:
            sys.path.remove(bio_path)


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Run full injury-risk analysis on a lift video",
    response_model_exclude_none=True,
)
async def analyze_lift(
    video:     UploadFile = File(..., description="Lift video (.mp4/.mov/.avi)"),
    user_id:   str        = Form(...),
    lift_type: LiftType   = Form(...),
    weight_kg: float      = Form(...),
    rpe:       float      = Form(...),
) -> AnalysisResponse:

    trace_id = generate_trace_id()
    token    = trace_id_var.set(trace_id)
    t_total  = time.perf_counter()

    logger.info(
        "analyze.start",
        trace_id=trace_id,
        user=user_id,
        lift=lift_type,
        weight_kg=weight_kg,
        rpe=rpe,
    )

    try:
        # ── 1. Validate input ────────────────────────────────────────────────
        req = AnalysisRequest(
            user_id=user_id, lift_type=lift_type, weight_kg=weight_kg, rpe=rpe
        )
        _validate_video_file(video)

        # ── 2. Fetch user data ───────────────────────────────────────────────
        profile, workout_logs, sleep_logs, nutrition_logs = await _fetch_user_data(req.user_id, req.lift_type.value)

        # ── 3. Save video to temp file, run CV analyzer ──────────────────────
        video_bytes = await video.read()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            logger.info("cv_analyzer.start", trace_id=trace_id, path=tmp_path)
            bio_json = await run_in_threadpool(_run_cv_analyzer, req.lift_type.value, tmp_path)
            logger.info("cv_analyzer.complete", trace_id=trace_id, reps=bio_json.get("overall_summary", {}).get("total_reps"))
        finally:
            os.unlink(tmp_path)

        # ── 4. Compute physics indices ───────────────────────────────────────
        bio_summary    = summarize_biomechanics(req.lift_type.value, bio_json)
        stress_indices = compute_stress_indices(req.lift_type.value, req.weight_kg, bio_summary)

        fatigue_metrics = compute_all_fatigue_metrics(
            workout_logs, req.lift_type.value, req.rpe, req.weight_kg
        )
        recovery_metrics = compute_recovery_score(sleep_logs, nutrition_logs, profile.bodyweight_kg)

        # ── 5. Build initial agent state ─────────────────────────────────────
        initial_state: AgentState = {
            "trace_id":       trace_id,
            "user_id":        req.user_id,
            "lift_type":      req.lift_type.value,
            "weight_kg":      req.weight_kg,
            "rpe":            req.rpe,
            "user_profile":   profile.model_dump(mode="json"),
            "workout_logs":   [w.model_dump(mode="json") for w in workout_logs],
            "sleep_logs":     [s.model_dump(mode="json") for s in sleep_logs],
            "nutrition_logs": [n.model_dump(mode="json") for n in nutrition_logs],
            "biomechanics_data": bio_json,
            "biomech_summary":   bio_summary,
            "stress_indices":    stress_indices.model_dump(),
            "fatigue_score":     fatigue_metrics["combined_fatigue_score"],
            "progression_rate":  fatigue_metrics["weekly_progression_rate"],
            "recovery_score":    recovery_metrics["recovery_score"],
            "biomech_output":    None,
            "fatigue_output":    None,
            "risk_output":       None,
            "agent_traces":      [],
            "errors":            [],
        }

        # ── 6. Run LangGraph agent pipeline ──────────────────────────────────
        final_state = await run_pipeline(initial_state)

        if not final_state.get("risk_output"):
            raise HTTPException(status_code=500, detail="Agent pipeline did not produce a risk output.")

        # ── 7. Build response ────────────────────────────────────────────────
        total_ms = round((time.perf_counter() - t_total) * 1000, 1)

        response = AnalysisResponse(
            trace_id=trace_id,
            user_id=req.user_id,
            lift_type=req.lift_type.value,
            weight_kg=req.weight_kg,
            stress_indices=stress_indices,
            fatigue_score=fatigue_metrics["combined_fatigue_score"],
            recovery_score=recovery_metrics["recovery_score"],
            progression_rate_pct=round(fatigue_metrics["weekly_progression_rate"] * 100, 1),
            risk_result=final_state["risk_output"],
            agent_traces=[AgentTrace(**t) for t in final_state.get("agent_traces", [])],
            processing_time_ms=total_ms,
        )

        logger.info(
            "analyze.complete",
            trace_id=trace_id,
            risk_level=final_state["risk_output"].get("risk_level"),
            total_ms=total_ms,
        )
        return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("analyze.error", trace_id=trace_id, error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")
    finally:
        trace_id_var.reset(token)


async def _fetch_user_data(user_id: str, exercise: str):
    """Fetch all user data in parallel (single call in mock; parallel in Supabase version)."""
    import asyncio
    profile, workout_logs, sleep_logs, nutrition_logs = await asyncio.gather(
        _data_provider.get_user_profile(user_id),
        _data_provider.get_workout_logs(user_id, days=28),
        _data_provider.get_sleep_logs(user_id, days=7),
        _data_provider.get_nutrition_logs(user_id, days=7),
    )
    return profile, workout_logs, sleep_logs, nutrition_logs
