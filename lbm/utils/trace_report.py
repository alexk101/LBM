"""Parse JAX/Perfetto traces and produce a simple text report.

Use this to compare runs with different collision schemes or distributions:
run a traced session, then pass the trace path (or jax-trace log dir) to
report_trace() or summarize_trace().

Requires the optional dependency: pip install perfetto (or uv sync --group profiling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    from perfetto.trace_processor import TraceProcessor
except ImportError:
    TraceProcessor = None  # type: ignore[misc, assignment]

# Time in trace is nanoseconds
NS_TO_MS = 1e-6


def _find_trace_in_dir(log_dir: Path) -> Path | None:
    """Return path to perfetto_trace.json.gz in the latest run under plugins/profile."""
    profile_dir = log_dir / "plugins" / "profile"
    if not profile_dir.is_dir():
        return None
    runs = sorted(profile_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in runs:
        if not run_dir.is_dir():
            continue
        trace_path = run_dir / "perfetto_trace.json.gz"
        if trace_path.is_file():
            return trace_path
    return None


def resolve_trace_path(path: str | Path) -> Path:
    """Resolve to a single trace file.

    If path is a directory (e.g. jax-trace), finds the latest perfetto_trace.json.gz
    under plugins/profile/<timestamp>/.
    """
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        found = _find_trace_in_dir(p)
        if found is not None:
            return found
        raise FileNotFoundError(
            f"No perfetto_trace.json.gz found under {p}/plugins/profile/"
        )
    raise FileNotFoundError(f"Trace path does not exist: {p}")


@dataclass
class TraceSummary:
    """Aggregated timing from a Perfetto trace."""

    trace_path: str
    total_host_ns: int = 0
    total_gpu_compute_ns: int = 0
    total_gpu_memcpy_ns: int = 0
    by_category: dict[str, int] = field(default_factory=dict)  # category -> ns
    by_name: dict[str, int] = field(default_factory=dict)  # slice name -> total ns
    categories_used: list[str] = field(default_factory=list)


def _categorize(name: str | None) -> str:
    if not name:
        return "other"
    if name.startswith("Memcpy"):
        return "gpu_memcpy"
    if "PjitFunction(step)" in name or name == "jit__step" or "step" in name and "lbm" in name.lower():
        return "lbm_step"
    if "lbm.py" in name and "run" in name:
        return "lbm_run"
    if "lbm.py" in name and "plot" in name:
        return "lbm_plot"
    if "lbm.py" in name and "log" in name:
        return "lbm_log"
    if any(
        x in name
        for x in (
            "Compile",
            "compile",
            "PJRT_Client_Compile",
            "CompileToBackendResult",
            "CompileSingleModule",
            "Compiling IR",
            "OptimizeHloModule",
            "CompileModuleToLlvmIr",
            "PTX->CUBIN",
            "backend_compile",
            "cache_miss",
            "_cached_compilation",
            "lower_jaxpr",
            "lower_sharding",
            "autotune",
            "post-fusion-simplification",
            "post-fusion optimization",
            "pre-fusion",
            "rename_fusions",
            "multi_output_fusion",
        )
    ):
        return "jax_compile"
    if any(
        x in name
        for x in (
            "maxwell_",
            "sgemm",
            "input_reduce_fusion",
            "loop_gather_fusion",
            "loop_divide_fusion",
            "loop_reduce_fusion",
            "wrapped_broadcast",
            "fusion-wrapper",
        )
    ) or (name.startswith("fusion") and "emitter" not in name and "optimization" not in name):
        return "gpu_compute"
    if "lbm" in name.lower() or "step" in name or "plot" in name or "log" in name:
        return "lbm_other"
    return "other"


def summarize_trace(path: str | Path) -> TraceSummary:
    """Load a Perfetto trace and return an aggregated TraceSummary.

    Args:
        path: Path to perfetto_trace.json.gz or to the trace log directory
              (e.g. ./jax-trace).

    Returns:
        TraceSummary with totals and per-category / per-name breakdowns.

    Raises:
        FileNotFoundError: If no trace file is found.
        RuntimeError: If the perfetto package is not installed.
    """
    if TraceProcessor is None:
        raise RuntimeError(
            "Trace parsing requires the 'perfetto' package. "
            "Install with: uv sync --group profiling"
        )
    trace_path = resolve_trace_path(path)
    summary = TraceSummary(trace_path=str(trace_path))

    with TraceProcessor(trace=str(trace_path)) as tp:
        rows = tp.query(
            """
            SELECT name, SUM(dur) as total_dur
            FROM slice
            WHERE dur > 0 AND name IS NOT NULL
            GROUP BY name
            """
        )
        for row in rows:
            name = row.name
            total_ns = row.total_dur
            summary.by_name[name] = total_ns
            cat = _categorize(name)
            summary.by_category[cat] = summary.by_category.get(cat, 0) + total_ns

    summary.total_host_ns = (
        summary.by_category.get("lbm_run", 0)
        + summary.by_category.get("lbm_plot", 0)
        + summary.by_category.get("lbm_log", 0)
        + summary.by_category.get("lbm_step", 0)
        + summary.by_category.get("lbm_other", 0)
        + summary.by_category.get("jax_compile", 0)
        + summary.by_category.get("other", 0)
    )
    summary.total_gpu_compute_ns = summary.by_category.get("gpu_compute", 0)
    summary.total_gpu_memcpy_ns = summary.by_category.get("gpu_memcpy", 0)
    summary.categories_used = sorted(summary.by_category.keys())

    return summary


def format_report(summary: TraceSummary) -> str:
    """Format a TraceSummary as a human-readable text report."""
    lines = [
        f"Trace: {summary.trace_path}",
        "",
        "--- Totals ---",
        f"  Host (total):     {summary.total_host_ns * NS_TO_MS:>12.2f} ms",
        f"  GPU compute:      {summary.total_gpu_compute_ns * NS_TO_MS:>12.2f} ms",
        f"  GPU memcpy:       {summary.total_gpu_memcpy_ns * NS_TO_MS:>12.2f} ms",
        "",
    ]
    total_ns = summary.total_host_ns + summary.total_gpu_compute_ns + summary.total_gpu_memcpy_ns
    if total_ns > 0:
        lines.append("--- By category ---")
        for cat in summary.categories_used:
            ns = summary.by_category[cat]
            pct = 100.0 * ns / total_ns
            lines.append(f"  {cat:<20} {ns * NS_TO_MS:>10.2f} ms  ({pct:>5.1f}%)")
        lines.append("")
    lines.append("--- Top GPU compute kernels (by duration) ---")
    gpu_names = [
        (name, ns)
        for name, ns in summary.by_name.items()
        if _categorize(name) == "gpu_compute"
    ]
    gpu_names.sort(key=lambda x: -x[1])
    for name, ns in gpu_names[:20]:
        lines.append(f"  {ns * NS_TO_MS:>10.2f} ms  {name}")
    lines.append("")
    lines.append("--- Top GPU memcpy ---")
    memcpy_names = [
        (name, ns)
        for name, ns in summary.by_name.items()
        if _categorize(name) == "gpu_memcpy"
    ]
    memcpy_names.sort(key=lambda x: -x[1])
    for name, ns in memcpy_names[:10]:
        lines.append(f"  {ns * NS_TO_MS:>10.2f} ms  {name}")
    lines.append("")
    lines.append("--- LBM / JAX (host) ---")
    for cat in ("lbm_run", "lbm_step", "lbm_plot", "lbm_log", "jax_compile"):
        ns = summary.by_category.get(cat, 0)
        if ns > 0:
            lines.append(f"  {cat:<20} {ns * NS_TO_MS:>10.2f} ms")
    return "\n".join(lines)


def report_trace(path: str | Path, print_report: bool = True) -> TraceSummary:
    """Load a trace, summarize it, and optionally print the report.

    Args:
        path: Path to perfetto_trace.json.gz or trace log directory.
        print_report: If True, print the formatted report to stdout.

    Returns:
        TraceSummary for programmatic use or comparison.
    """
    summary = summarize_trace(path)
    if print_report:
        print(format_report(summary))
    return summary


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "jax-trace"
    report_trace(path)
