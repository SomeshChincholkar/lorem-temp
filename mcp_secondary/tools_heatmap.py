"""
mcp_secondary/tools_heatmap.py

generate_risk_heatmap_tool. Reads already-generated JSON reports from
data/reports/ (produced by the Primary server's
clinical_insight_reporter_tool, step 3.8) and renders a risk-score
heatmap across patients using matplotlib.

This tool does NOT call the primary server over MCP -- it reads the
same data/reports/ directory directly off disk, since both servers
share the same filesystem in this project layout. If your deployment
ever splits them onto different machines, this would need to become
an actual MCP client call to the primary server instead.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import Context  # noqa: E402

REPORTS_DIR = Path("data/reports")


def _load_risk_scores(patient_ids: List[str]) -> Dict[str, dict]:
    """
    Returns {patient_id: {"risk_score": int|None, "error": str|None}}
    -- missing/unreadable reports get risk_score=None + an error note
    instead of crashing the whole tool over one bad patient_id.
    """
    results = {}
    for pid in patient_ids:
        report_path = REPORTS_DIR / f"{pid}_report.json"
        if not report_path.exists():
            results[pid] = {"risk_score": None, "error": f"No report found at {report_path}"}
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            results[pid] = {"risk_score": report.get("risk_score"), "error": None}
        except (json.JSONDecodeError, OSError) as e:
            results[pid] = {"risk_score": None, "error": f"Failed to read report: {e}"}
    return results


def plot_heatmap(scores: Dict[str, int]):
    """
    Build a simple 1-row heatmap: one column per patient, colored by
    risk score. `scores` must already have None/missing entries
    filtered out before calling this.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless -- no GUI backend needed on a server
    import matplotlib.pyplot as plt
    import numpy as np

    patient_ids = list(scores.keys())
    values = np.array([[scores[pid] for pid in patient_ids]])

    fig, ax = plt.subplots(figsize=(max(4, len(patient_ids) * 1.2), 2.5))
    im = ax.imshow(values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(patient_ids)))
    ax.set_xticklabels(patient_ids, rotation=45, ha="right")
    ax.set_yticks([])
    ax.set_title("Patient Risk Score Heatmap")

    for i, pid in enumerate(patient_ids):
        ax.text(i, 0, str(scores[pid]), ha="center", va="center", color="black", fontweight="bold")

    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.3, label="Risk Score")
    fig.tight_layout()
    return fig


async def generate_risk_heatmap_tool(ctx: Context, patient_ids: List[str]) -> Dict:
    """
    Args:
        ctx: MCP Context (unused -- reads local files + renders, no I/O
            beyond disk).
        patient_ids: list of patient IDs, e.g. ["P1001", "P1002"]

    Returns:
        {heatmap_path: str, scores: {patient_id: risk_score|None},
         skipped: {patient_id: error_reason}}  (skipped only present if
         any patient_ids had no readable report)
    """
    all_results = _load_risk_scores(patient_ids)

    valid_scores = {pid: r["risk_score"] for pid, r in all_results.items() if r["risk_score"] is not None}
    skipped = {pid: r["error"] for pid, r in all_results.items() if r["risk_score"] is None}

    if not valid_scores:
        raise ValueError(
            f"No readable reports found for any of {patient_ids}. "
            f"Run clinical_insight_reporter_tool for these patients first. "
            f"Details: {skipped}"
        )

    fig = plot_heatmap(valid_scores)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    heatmap_path = REPORTS_DIR / f"heatmap_{uuid4().hex[:6]}.png"
    fig.savefig(heatmap_path)

    result = {"heatmap_path": str(heatmap_path), "scores": valid_scores}
    if skipped:
        result["skipped"] = skipped
    return result