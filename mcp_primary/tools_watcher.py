"""
tools_watcher.py

The Watcher primitive. Scans the authorized Data/incoming/ subfolders
(bills, doctor_reports, lab_reports) for files that haven't been
processed yet, and returns them as a worklist for the Discharge
Monitor Agent.

Note: this module defines the tool as a plain function (no @mcp.tool()
decorator here). Registration happens centrally in server.py via
mcp.tool()(clinical_watcher_tool), per the wiring plan in section 3.9.
This keeps the module importable/testable without needing a live
FastMCP app instance.
"""

import json
from pathlib import Path
from typing import Dict, List

from mcp.server.fastmcp import Context

from roots import resolve_authorized_root, safe_join

# Where we persist which files have already been emitted, so repeated
# scans don't re-report the same document.
PROCESSED_STATE_FILE = Path("data/processed.json")

DOC_SUBFOLDERS = ["bills", "doctor_reports", "lab_reports"]


def _load_processed() -> set:
    """Load the set of already-processed filenames from the state file."""
    if not PROCESSED_STATE_FILE.exists():
        return set()
    try:
        with open(PROCESSED_STATE_FILE, "r") as f:
            return set(json.load(f))
    except (json.JSONDecodeError, OSError):
        # Corrupt or unreadable state file -> fail safe to "nothing processed"
        return set()


def already_processed(f: Path) -> bool:
    """True if this file has already been emitted by a previous scan."""
    processed = _load_processed()
    return f.name in processed


def mark_processed(f: Path) -> None:
    """
    Record a file as processed. Not called by the watcher itself (the
    watcher only *reports* unprocessed files) -- call this downstream,
    once an agent has actually consumed/handled the file, so it stops
    showing up in future scans.

    Exposed over MCP as mark_documents_processed_tool below, because the
    Orchestrator that knows a case finished is in a different process
    from the server that owns this state file.
    """
    PROCESSED_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    processed = _load_processed()
    processed.add(f.name)
    with open(PROCESSED_STATE_FILE, "w") as fp:
        json.dump(sorted(processed), fp, indent=2)


async def mark_documents_processed_tool(
    ctx: Context, patient_id: str, subfolder: str = ""
) -> Dict:
    """
    Mark a patient's documents as processed so later scans skip them.

    Called by the Host Orchestrator once a discharge case has been
    validated. Without it the Watcher re-reports the same paperwork on
    every scan and the Orchestrator re-runs cases that are already done.

    Marking is deliberately scoped to one patient rather than "everything
    the last scan returned": a scan can return several patients, and only
    the ones that actually completed should be retired.

    Args:
        ctx: MCP Context (used to resolve the authorized root).
        patient_id: e.g. "P1019".
        subfolder: optional, restrict to one of "bills", "doctor_reports",
            "lab_reports". Empty marks all three.

    Returns:
        {"patient_id": str, "marked": [filename, ...], "count": int}
    """
    root = await resolve_authorized_root(ctx)

    if subfolder:
        if subfolder not in DOC_SUBFOLDERS:
            raise ValueError(
                f"Invalid subfolder '{subfolder}'. Must be one of {DOC_SUBFOLDERS}"
            )
        subfolders_to_scan = [subfolder]
    else:
        subfolders_to_scan = DOC_SUBFOLDERS

    marked: List[str] = []
    for sub in subfolders_to_scan:
        folder = safe_join(root, sub)
        if not folder.exists() or not folder.is_dir():
            continue
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.name.startswith(f"{patient_id}_"):
                mark_processed(f)
                marked.append(f.name)

    return {"patient_id": patient_id, "marked": marked, "count": len(marked)}


def reset_processed_state() -> None:
    """
    Clear the processed ledger. Used by tests and by anyone wanting to
    re-run the whole corpus from scratch.
    """
    if PROCESSED_STATE_FILE.exists():
        PROCESSED_STATE_FILE.unlink()


async def clinical_watcher_tool(ctx: Context, subfolder: str = "") -> List[Dict]:
    """
    Scan Data/incoming/{bills,doctor_reports,lab_reports} (relative to
    the client-authorized root) for new/unprocessed files.

    Args:
        ctx: MCP Context (used to resolve the authorized root).
        subfolder: optional. One of "bills", "doctor_reports",
            "lab_reports". If empty, all three are scanned.

    Returns:
        list[dict] of {patient_id, filename, doc_type, path}
    """
    root = await resolve_authorized_root(ctx)

    if subfolder:
        if subfolder not in DOC_SUBFOLDERS:
            raise ValueError(
                f"Invalid subfolder '{subfolder}'. Must be one of {DOC_SUBFOLDERS}"
            )
        subfolders_to_scan = [subfolder]
    else:
        subfolders_to_scan = DOC_SUBFOLDERS

    found: List[Dict] = []
    for sub in subfolders_to_scan:
        folder = safe_join(root, sub)
        if not folder.exists() or not folder.is_dir():
            continue

        for f in sorted(folder.iterdir()):
            if not f.is_file():
                continue
            if f.name.startswith("."):
                continue
            if already_processed(f):
                continue

            # patient_id is always the first underscore-delimited token
            # e.g. P1019_bill.json -> P1019, P1019_JohnDoe.txt -> P1019
            pid = f.name.split("_")[0]

            found.append({
                "patient_id": pid,
                "filename": f.name,
                "doc_type": sub,
                "path": str(f),
            })

    return found