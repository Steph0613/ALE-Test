"""
Remote Evaluator Service for ALE-Bench GRPO Training.

Runs on the development host (10.214.54.87) where Docker is available.
Exposes a FastAPI HTTP API for the training container to call.

Endpoints:
  GET  /health       - Health check
  POST /warmup       - Pre-download & build tools for a problem
  POST /evaluate     - Compile, run, and judge code on given seeds

Usage:
  uvicorn evaluator_server:app --host 0.0.0.0 --port 8000

Requires:
  - ale_bench installed (pip install -e . from repo root)
  - Docker daemon running
  - Rust tool Docker image pulled (rust:1.79.0-buster)
  - Language Docker images pulled (scripts/docker_pull_all.sh)
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# ALE-Bench imports
import ale_bench.constants
from ale_bench.code_language import CodeLanguage, JudgeVersion
from ale_bench.data import (
    Problem,
    ProblemType,
    RankPerformanceMap,
    ScoreType,
    Seeds,
    Standings,
    build_rust_tools,
    load_problem,
)
from ale_bench.result import CaseResult, JudgeResult
from ale_bench.tool_wrappers.case_runner import run_cases
from ale_bench.tool_wrappers.input_generation import generate_inputs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CACHE_BASE = Path(os.environ.get("ALE_EVAL_CACHE", "/data/ale_bench_cache"))
MAX_CONCURRENT_EVALS = int(os.environ.get("ALE_EVAL_MAX_CONCURRENT", "4"))
API_KEY = os.environ.get("ALE_EVAL_API_KEY", "")
EVAL_WALL_TIMEOUT = int(os.environ.get("ALE_EVAL_WALL_TIMEOUT", "600"))  # seconds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ale_evaluator")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
app = FastAPI(title="ALE-Bench Remote Evaluator", version="0.1.0")
eval_semaphore: asyncio.Semaphore  # initialized on startup

# Cache: problem_id -> { problem, seeds, standings, rpm, tool_dir, data_root }
problem_cache: dict[str, dict[str, Any]] = {}
problem_cache_lock = asyncio.Lock()


@app.on_event("startup")
async def startup() -> None:
    global eval_semaphore
    eval_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EVALS)
    CACHE_BASE.mkdir(parents=True, exist_ok=True)
    logger.info(f"Evaluator started. max_concurrent={MAX_CONCURRENT_EVALS}, cache={CACHE_BASE}")


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
def check_auth(authorization: Optional[str]) -> None:
    if API_KEY and (not authorization or authorization != f"Bearer {API_KEY}"):
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class WarmupRequest(BaseModel):
    problem_id: str
    lite_version: bool = True


class EvaluateRequest(BaseModel):
    problem_id: str
    code_language: str = "cpp17"
    judge_version: str = "202301"
    code: str
    seeds: list[int]
    time_limit: Optional[float] = None
    memory_limit: Optional[int] = None
    skip_local_visualization: bool = True
    return_details: bool = False
    lite_version: bool = True


class CaseResultOut(BaseModel):
    judge_result: str
    absolute_score: int
    execution_time: float
    memory_usage: int
    message: Optional[str] = None


class AggOut(BaseModel):
    mean_abs_score: float
    sum_abs_score: int
    fail_counts: dict[str, int]


class FeedbackOut(BaseModel):
    fail_type: str
    score: float
    mean_time: Optional[float] = None
    mean_mem: Optional[float] = None
    short_message: Optional[str] = None


class TimingsOut(BaseModel):
    compile_s: float
    run_s_total: float


class EvaluateResponse(BaseModel):
    ok: bool
    problem_id: str
    problem_type: str
    score_type: str
    raw_case_results: list[CaseResultOut]
    agg: AggOut
    feedback: FeedbackOut
    timings: TimingsOut
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Problem loading and caching
# ---------------------------------------------------------------------------
async def ensure_problem_loaded(problem_id: str, lite_version: bool) -> dict[str, Any]:
    """Load a problem, caching results. Build rust tools if needed."""
    cache_key = f"{problem_id}_{'lite' if lite_version else 'full'}"
    if cache_key in problem_cache:
        return problem_cache[cache_key]

    async with problem_cache_lock:
        # Double-check after lock
        if cache_key in problem_cache:
            return problem_cache[cache_key]

        logger.info(f"Loading problem {problem_id} (lite={lite_version})...")
        t0 = time.time()

        # load_problem extracts to a temp dir; we want a persistent cache dir
        problem, seeds, standings, rpm, data_root = await asyncio.to_thread(
            load_problem, problem_id, lite_version
        )

        # Persistent tool dir
        tool_cache_dir = CACHE_BASE / "problems" / problem_id / "tools"
        if not (tool_cache_dir / "target" / "release").is_dir():
            # Copy tools from data_root to persistent cache
            if tool_cache_dir.exists():
                shutil.rmtree(tool_cache_dir)
            shutil.copytree(data_root / "tools", tool_cache_dir)
            logger.info(f"Building rust tools for {problem_id}...")
            await asyncio.to_thread(build_rust_tools, tool_cache_dir)
            logger.info(f"Rust tools built for {problem_id}")
        else:
            logger.info(f"Rust tools already cached for {problem_id}")

        # The tool_dir for run_cases needs the 'tools' dir at tool_cache_dir.parent
        # because run_cases references tool_dir / "tools" / "target" / "release" / "tester"
        # So we set tool_dir to the parent of our tool_cache_dir
        tool_dir = tool_cache_dir.parent

        entry = {
            "problem": problem,
            "seeds": seeds,
            "standings": standings,
            "rpm": rpm,
            "data_root": data_root,
            "tool_dir": tool_dir,
        }
        problem_cache[cache_key] = entry
        logger.info(f"Problem {problem_id} loaded in {time.time() - t0:.1f}s")
        return entry


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "max_concurrent": str(MAX_CONCURRENT_EVALS)}


@app.post("/warmup")
async def warmup(
    req: WarmupRequest,
    authorization: Optional[str] = Header(None),
) -> dict[str, Any]:
    check_auth(authorization)
    try:
        entry = await ensure_problem_loaded(req.problem_id, req.lite_version)
        p: Problem = entry["problem"]
        return {
            "ok": True,
            "problem_id": req.problem_id,
            "problem_type": p.metadata.problem_type.value,
            "score_type": p.metadata.score_type.value,
            "time_limit": p.constraints.time_limit,
            "memory_limit": p.constraints.memory_limit,
            "num_public_seeds": len(entry["seeds"].public),
            "num_private_seeds": len(entry["seeds"].private),
        }
    except Exception as e:
        logger.error(f"Warmup failed for {req.problem_id}: {traceback.format_exc()}")
        return {"ok": False, "error": str(e)}


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    req: EvaluateRequest,
    authorization: Optional[str] = Header(None),
) -> EvaluateResponse:
    check_auth(authorization)
    overall_start = time.time()

    # Load problem
    try:
        entry = await ensure_problem_loaded(req.problem_id, req.lite_version)
    except Exception as e:
        return EvaluateResponse(
            ok=False,
            problem_id=req.problem_id,
            problem_type="unknown",
            score_type="unknown",
            raw_case_results=[],
            agg=AggOut(mean_abs_score=0, sum_abs_score=0, fail_counts={}),
            feedback=FeedbackOut(fail_type="INTERNAL_ERROR", score=0),
            timings=TimingsOut(compile_s=0, run_s_total=0),
            error=f"Failed to load problem: {e}",
        )

    problem: Problem = entry["problem"]
    seeds_obj: Seeds = entry["seeds"]
    tool_dir: Path = entry["tool_dir"]

    # Resolve parameters
    code_lang = CodeLanguage(req.code_language.replace("python3", "python"))
    judge_ver = JudgeVersion(req.judge_version)
    time_limit = req.time_limit or problem.constraints.time_limit
    memory_limit = req.memory_limit or problem.constraints.memory_limit
    problem_type = problem.metadata.problem_type
    score_type = problem.metadata.score_type

    # Generate inputs for requested seeds
    try:
        async with eval_semaphore:
            gen_kwargs: dict[str, Any] = {}
            t_gen_start = time.time()
            inputs = await asyncio.to_thread(
                generate_inputs, req.seeds, gen_kwargs, tool_dir
            )
            t_gen_end = time.time()

            if len(inputs) != len(req.seeds):
                return EvaluateResponse(
                    ok=False,
                    problem_id=req.problem_id,
                    problem_type=problem_type.value,
                    score_type=score_type.value,
                    raw_case_results=[],
                    agg=AggOut(mean_abs_score=0, sum_abs_score=0, fail_counts={}),
                    feedback=FeedbackOut(fail_type="INTERNAL_ERROR", score=0),
                    timings=TimingsOut(compile_s=0, run_s_total=0),
                    error=f"Input generation mismatch: expected {len(req.seeds)}, got {len(inputs)}",
                )

            # Run cases
            t_run_start = time.time()
            case_results: list[CaseResult] = await asyncio.to_thread(
                run_cases,
                inputs,
                req.code,
                code_lang,
                judge_ver,
                time_limit,
                memory_limit,
                req.problem_id,
                problem_type,
                tool_dir,
                req.return_details,
                req.skip_local_visualization,
                1,  # num_workers: run sequentially per evaluation to limit Docker overhead
            )
            t_run_end = time.time()

    except asyncio.TimeoutError:
        return EvaluateResponse(
            ok=False,
            problem_id=req.problem_id,
            problem_type=problem_type.value,
            score_type=score_type.value,
            raw_case_results=[],
            agg=AggOut(mean_abs_score=0, sum_abs_score=0, fail_counts={}),
            feedback=FeedbackOut(fail_type="TIME_LIMIT_EXCEEDED", score=0),
            timings=TimingsOut(compile_s=0, run_s_total=0),
            error="Evaluation timed out",
        )
    except Exception as e:
        logger.error(f"Evaluate failed: {traceback.format_exc()}")
        return EvaluateResponse(
            ok=False,
            problem_id=req.problem_id,
            problem_type=problem_type.value,
            score_type=score_type.value,
            raw_case_results=[],
            agg=AggOut(mean_abs_score=0, sum_abs_score=0, fail_counts={}),
            feedback=FeedbackOut(fail_type="INTERNAL_ERROR", score=0),
            timings=TimingsOut(compile_s=0, run_s_total=0),
            error=str(e),
        )

    # Build response
    raw_results = []
    fail_counts: dict[str, int] = {}
    total_score = 0
    accepted_count = 0
    total_time = 0.0
    total_mem = 0

    for cr in case_results:
        raw_results.append(CaseResultOut(
            judge_result=cr.judge_result.value,
            absolute_score=cr.absolute_score,
            execution_time=cr.execution_time,
            memory_usage=cr.memory_usage,
            message=(cr.message[:500] if cr.message else None),
        ))
        jr_name = cr.judge_result.value
        fail_counts[jr_name] = fail_counts.get(jr_name, 0) + 1
        if cr.judge_result == JudgeResult.ACCEPTED:
            total_score += cr.absolute_score
            accepted_count += 1
            total_time += cr.execution_time
            total_mem += cr.memory_usage

    n = len(case_results)
    mean_score = total_score / n if n > 0 else 0.0
    mean_time = total_time / accepted_count if accepted_count > 0 else None
    mean_mem = total_mem / accepted_count if accepted_count > 0 else None

    # Determine overall fail_type
    # Priority: COMPILE_ERROR > RE > TLE > MLE > WA > OK
    fail_type = "OK"
    for ft in ["COMPILATION_ERROR", "RUNTIME_ERROR", "TIME_LIMIT_EXCEEDED",
               "MEMORY_LIMIT_EXCEEDED", "WRONG_ANSWER", "INTERNAL_ERROR"]:
        if fail_counts.get(ft, 0) > 0:
            # Map to simpler names
            type_map = {
                "COMPILATION_ERROR": "COMPILE_ERROR",
                "RUNTIME_ERROR": "RE",
                "TIME_LIMIT_EXCEEDED": "TLE",
                "MEMORY_LIMIT_EXCEEDED": "MLE",
                "WRONG_ANSWER": "WA",
                "INTERNAL_ERROR": "INTERNAL_ERROR",
            }
            fail_type = type_map.get(ft, ft)
            break

    # Short message from first failing case
    short_msg = None
    for cr in case_results:
        if cr.judge_result != JudgeResult.ACCEPTED and cr.message:
            short_msg = cr.message[:500]
            break

    compile_s = t_gen_end - t_gen_start  # gen time as proxy for compile
    run_s_total = t_run_end - t_run_start

    return EvaluateResponse(
        ok=True,
        problem_id=req.problem_id,
        problem_type=problem_type.value,
        score_type=score_type.value,
        raw_case_results=raw_results,
        agg=AggOut(
            mean_abs_score=mean_score,
            sum_abs_score=total_score,
            fail_counts=fail_counts,
        ),
        feedback=FeedbackOut(
            fail_type=fail_type,
            score=mean_score,
            mean_time=mean_time,
            mean_mem=mean_mem,
            short_message=short_msg,
        ),
        timings=TimingsOut(
            compile_s=compile_s,
            run_s_total=run_s_total,
        ),
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("ALE_EVAL_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
