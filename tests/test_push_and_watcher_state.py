"""
tests/test_push_and_watcher_state.py

Covers A2A push notifications (spec §9) and the Watcher's processed-file
ledger.

The ledger matters because without it the Watcher re-reports the same
paperwork on every scan, so the Orchestrator keeps re-running cases that
are already done. `mark_processed` existed but had no callers until the
mark_documents_processed_tool was added.

Run:  python -m pytest tests/test_push_and_watcher_state.py -q
"""

import json
from pathlib import Path

import pytest

from agents.common import push_notifications
from mcp_primary import tools_watcher


# ---------------------------------------------------------------------
# Watcher processed-file ledger
# ---------------------------------------------------------------------
@pytest.fixture
def temp_ledger(tmp_path, monkeypatch):
    ledger = tmp_path / "processed.json"
    monkeypatch.setattr(tools_watcher, "PROCESSED_STATE_FILE", ledger)
    return ledger


def test_unseen_file_is_not_processed(temp_ledger):
    assert tools_watcher.already_processed(Path("P1019_bill.json")) is False


def test_marking_makes_a_file_processed(temp_ledger):
    tools_watcher.mark_processed(Path("P1019_bill.json"))
    assert tools_watcher.already_processed(Path("P1019_bill.json")) is True


def test_marking_is_scoped_to_that_file(temp_ledger):
    """Retiring one patient's paperwork must not retire another's."""
    tools_watcher.mark_processed(Path("P1019_bill.json"))
    assert tools_watcher.already_processed(Path("P1015_bill.json")) is False


def test_ledger_persists_across_reads(temp_ledger):
    tools_watcher.mark_processed(Path("P1019_bill.json"))
    tools_watcher.mark_processed(Path("P1015_labs.txt"))

    stored = json.loads(temp_ledger.read_text(encoding="utf-8"))
    assert sorted(stored) == ["P1015_labs.txt", "P1019_bill.json"]


def test_marking_twice_is_idempotent(temp_ledger):
    tools_watcher.mark_processed(Path("P1019_bill.json"))
    tools_watcher.mark_processed(Path("P1019_bill.json"))

    assert json.loads(temp_ledger.read_text(encoding="utf-8")) == ["P1019_bill.json"]


def test_corrupt_ledger_fails_safe_to_nothing_processed(temp_ledger):
    """
    A corrupt ledger must mean "re-process", not "skip everything".
    Skipping would silently drop live cases.
    """
    temp_ledger.write_text("{not json", encoding="utf-8")
    assert tools_watcher.already_processed(Path("P1019_bill.json")) is False


def test_reset_clears_the_ledger(temp_ledger):
    tools_watcher.mark_processed(Path("P1019_bill.json"))
    tools_watcher.reset_processed_state()

    assert tools_watcher.already_processed(Path("P1019_bill.json")) is False


# ---------------------------------------------------------------------
# Push notifications
# ---------------------------------------------------------------------
def test_disabled_without_a_configured_url(monkeypatch):
    monkeypatch.setattr(push_notifications, "PUSH_NOTIFICATION_URL", "")
    assert push_notifications.is_enabled() is False


def test_enabled_with_a_url(monkeypatch):
    monkeypatch.setattr(push_notifications, "PUSH_NOTIFICATION_URL", "http://x/hook")
    assert push_notifications.is_enabled() is True


@pytest.mark.asyncio
async def test_send_reports_failure_instead_of_raising(monkeypatch):
    """
    A dead webhook must not fail a discharge that already completed.
    """
    monkeypatch.setattr(push_notifications, "PUSH_NOTIFICATION_URL", "")
    result = await push_notifications.send_push_notification("discharge.reviewed", "P1019")

    assert result["sent"] is False
    assert result["error"]


@pytest.mark.asyncio
async def test_unreachable_host_is_reported_not_raised(monkeypatch):
    # Port 1 is reserved and refuses instantly, so this stays fast.
    result = await push_notifications.send_push_notification(
        "discharge.reviewed", "P1019", url="http://127.0.0.1:1/hook"
    )
    assert result["sent"] is False
    assert result["error"]


def test_payload_carries_the_case_identity():
    payload = push_notifications.build_notification(
        "discharge.reviewed", "P1016", trace_id="t-1", risk_level="high"
    )

    assert payload["event"] == "discharge.reviewed"
    assert payload["patient_id"] == "P1016"
    assert payload["trace_id"] == "t-1"
    assert payload["risk_level"] == "high"
    assert payload["sent_at"]


def test_signature_round_trips():
    body = b'{"event":"discharge.reviewed"}'
    signature = push_notifications.sign_payload(body, secret="s3cret")

    assert push_notifications.verify_signature(body, signature, secret="s3cret") is True


def test_tampered_body_fails_verification():
    body = b'{"risk_level":"high"}'
    signature = push_notifications.sign_payload(body, secret="s3cret")
    tampered = b'{"risk_level":"low"}'

    assert push_notifications.verify_signature(tampered, signature, secret="s3cret") is False


def test_wrong_secret_fails_verification():
    body = b'{"event":"x"}'
    signature = push_notifications.sign_payload(body, secret="s3cret")

    assert push_notifications.verify_signature(body, signature, secret="other") is False


def test_missing_signature_fails_verification():
    assert push_notifications.verify_signature(b"{}", "", secret="s3cret") is False
    assert push_notifications.verify_signature(b"{}", None, secret="s3cret") is False
