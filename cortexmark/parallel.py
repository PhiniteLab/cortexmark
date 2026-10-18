"""Parallel processing wrapper for tree-level pipeline operations.

Provides:
  - parallel_map: generic parallel execution with thread/process pools
  - parallel_tree: wrap any file-level function for parallel tree processing
  - Configurable workers, batch deadline, and pool type

Uses concurrent.futures for safe, stdlib-only parallelism.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from cortexmark.common import load_config, resolve_configured_path, setup_logging

T = TypeVar("T")

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_WORKERS = min(os.cpu_count() or 1, 8)
DEFAULT_POOL = "thread"  # "thread" or "process"


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""

    workers: int = DEFAULT_WORKERS
    pool_type: str = DEFAULT_POOL  # "thread" or "process"
    timeout: float | None = None  # overall batch wall-clock deadline in seconds


@dataclass
class TaskResult:
    """Result of a single parallel task."""

    input_path: str
    success: bool
    result: Any = None
    error: str = ""
    elapsed: float = 0.0


@dataclass
class ParallelReport:
    """Summary of a parallel batch run."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    total_elapsed: float = 0.0
    results: list[TaskResult] = field(default_factory=list)


# ── Core parallel engine ─────────────────────────────────────────────────────


def _get_pool(cfg: ParallelConfig) -> ProcessPoolExecutor | ThreadPoolExecutor:
    """Create the appropriate executor pool."""
    if cfg.pool_type == "process":
        return ProcessPoolExecutor(max_workers=cfg.workers)
    return ThreadPoolExecutor(max_workers=cfg.workers)


def _timed_call(func: Callable[[Path], T], path: Path) -> tuple[bool, T | None, str, float]:
    """Execute one task and measure time spent inside the worker."""
    start = time.monotonic()
    try:
        result = func(path)
        return True, result, "", round(time.monotonic() - start, 3)
    except Exception as exc:
        return False, None, str(exc), round(time.monotonic() - start, 3)


def parallel_map(
    func: Callable[[Path], T],
    paths: list[Path],
    *,
    config: ParallelConfig | None = None,
) -> ParallelReport:
    """Execute *func* on each path in parallel, returning a report.

    Parameters
    ----------
    func:
        A callable taking a single ``Path`` and returning any result.
    paths:
        File paths to process.
    config:
        Parallel configuration (workers, pool type, timeout). ``timeout`` is
        an overall wall-clock deadline for the batch. If it expires, completed
        results are retained, unfinished tasks are reported as timed out, and
        queued work is cancelled.

    Returns
    -------
    ParallelReport with per-file results and timing.
    """
    if not paths:
        return ParallelReport()

    cfg = config or ParallelConfig()
    start = time.monotonic()
    task_results: list[TaskResult] = []

    # Single-threaded fast-path for 1 worker or just 1 item
    if (cfg.workers <= 1 or len(paths) == 1) and cfg.timeout is None:
        for p in paths:
            t0 = time.monotonic()
            try:
                result = func(p)
                task_results.append(
                    TaskResult(
                        input_path=str(p),
                        success=True,
                        result=result,
                        elapsed=round(time.monotonic() - t0, 3),
                    )
                )
            except Exception as exc:
                task_results.append(
                    TaskResult(
                        input_path=str(p),
                        success=False,
                        error=str(exc),
                        elapsed=round(time.monotonic() - t0, 3),
                    )
                )
    else:
        pool = _get_pool(cfg)
        try:
            futures: dict[Future[tuple[bool, T | None, str, float]], tuple[int, Path]] = {}
            for index, p in enumerate(paths):
                future = pool.submit(_timed_call, func, p)
                futures[future] = (index, p)

            pending = dict(futures)
            results_by_index: dict[int, TaskResult] = {}
            try:
                for future in as_completed(futures, timeout=cfg.timeout):
                    index, p = pending.pop(future)
                    try:
                        success, result, error, elapsed = future.result()
                        results_by_index[index] = TaskResult(
                            input_path=str(p),
                            success=success,
                            result=result,
                            error=error,
                            elapsed=elapsed,
                        )
                    except Exception as exc:
                        results_by_index[index] = TaskResult(
                            input_path=str(p),
                            success=False,
                            error=str(exc),
                        )
            except TimeoutError:
                for future, (index, p) in pending.items():
                    future.cancel()
                    results_by_index[index] = TaskResult(
                        input_path=str(p),
                        success=False,
                        error="task did not complete within batch timeout",
                    )

            task_results = [results_by_index[index] for index in range(len(paths))]
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    succeeded = sum(1 for r in task_results if r.success)
    total_elapsed = round(time.monotonic() - start, 3)

    return ParallelReport(
        total=len(task_results),
        succeeded=succeeded,
        failed=len(task_results) - succeeded,
        total_elapsed=total_elapsed,
        results=task_results,
    )


def collect_md_files(input_root: Path) -> list[Path]:
    """Collect all Markdown files under a directory."""
    return sorted(p for p in input_root.rglob("*.md") if p.is_file())


def parallel_tree(
    func: Callable[[Path], T],
    input_root: Path,
    *,
    config: ParallelConfig | None = None,
    glob: str = "*.md",
) -> ParallelReport:
    """Run *func* in parallel on all files matching *glob* under *input_root*."""
    files = sorted(p for p in input_root.rglob(glob) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No {glob} files found under: {input_root}")
    return parallel_map(func, files, config=config)


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a pipeline operation in parallel across files.")
    parser.add_argument("--input", type=Path, help="Input directory")
    parser.add_argument(
        "--operation",
        choices=["ocr_quality", "figures", "qa"],
        default="ocr_quality",
        help="Operation to run in parallel",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--pool",
        choices=["thread", "process"],
        default=DEFAULT_POOL,
        help="Pool type (default: thread)",
    )
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("parallel", cfg)

    input_path = (args.input or resolve_configured_path(cfg, "output_raw_md", "outputs/raw_md")).resolve()

    pcfg = ParallelConfig(workers=args.workers, pool_type=args.pool)

    # Select operation
    if args.operation == "ocr_quality":
        from cortexmark.ocr_quality import assess_file

        func: Callable[[Path], Any] = assess_file
    elif args.operation == "figures":
        from cortexmark.figures import extract_from_file

        func = extract_from_file
    elif args.operation == "qa":
        from cortexmark.qa_pipeline import qa_file

        func = qa_file
    else:
        log.error("unknown operation: %s", args.operation)
        return 1

    try:
        report = parallel_tree(func, input_path, config=pcfg)
        log.info(
            "parallel %s: %d/%d succeeded in %.2fs (workers=%d, pool=%s)",
            args.operation,
            report.succeeded,
            report.total,
            report.total_elapsed,
            pcfg.workers,
            pcfg.pool_type,
        )
        if report.failed > 0:
            for r in report.results:
                if not r.success:
                    log.warning("FAILED %s: %s", r.input_path, r.error)

    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
