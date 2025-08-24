from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

from flask import Blueprint, jsonify, request, send_file
from firebase_admin import firestore, storage

from .auth import AuthError, verify_firebase_id_token


api_bp = Blueprint("api", __name__)


# -----------------------------
# Helpers
# -----------------------------


def _parse_iso8601(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    value = dt_str.strip()
    if not value:
        return None
    # Normalize Zulu suffix to +00:00 for fromisoformat
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _get_time_range() -> Tuple[Optional[datetime], Optional[datetime]]:
    from_param = request.args.get("from")
    to_param = request.args.get("to")
    start = _parse_iso8601(from_param)
    end = _parse_iso8601(to_param)
    return start, end


def _get_user_timezone_name(user_doc: Dict[str, Any]) -> str:
    tz = (user_doc or {}).get("timezone") or "UTC"
    return tz


def _ensure_owner(uid: str, user_id: str) -> Optional[Tuple[dict, int]]:
    if uid != user_id:
        return {"error": "unauthorized"}, 403
    return None


def _user_ref(db, user_id: str):
    return db.collection("users").document(user_id)


def _as_datetime(ts: Any) -> Optional[datetime]:
    # Firestore returns google.cloud.firestore_v1._helpers.Timestamp
    try:
        if isinstance(ts, datetime):
            return ts
        # Attempt to call to_datetime if available
        to_dt = getattr(ts, "to_datetime", None)
        if callable(to_dt):
            return to_dt()
    except Exception:
        pass
    return None


def _format_local(dt: Optional[datetime], tz_name: Optional[str]) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        if tz_name:
            dt = dt.astimezone(ZoneInfo(tz_name))
    except Exception:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%b %d, %Y %I:%M %p")  # e.g., Aug 24, 2025 03:12 PM


# -----------------------------
# Auth
# -----------------------------


@api_bp.post("/api/auth/verify")
def verify():
    try:
        uid, claims = verify_firebase_id_token(request)
        return jsonify({"uid": uid, "claims": claims}), 200
    except AuthError as exc:
        return jsonify({"error": str(exc)}), 401


# -----------------------------
# Medications & logs
# -----------------------------


@api_bp.post("/api/user/<user_id>/medEvent")
def create_med_event(user_id: str):
    try:
        uid, _claims = verify_firebase_id_token(request)
    except AuthError as exc:
        return jsonify({"error": str(exc)}), 401

    unauthorized = _ensure_owner(uid, user_id)
    if unauthorized:
        err, code = unauthorized
        return jsonify(err), code

    payload = request.get_json(silent=True) or {}
    med_id = (payload.get("medId") or "").strip()
    taken = bool(payload.get("taken", True))
    scheduled_time = payload.get("scheduledTime")  # allow None or ISO string
    notes = payload.get("notes")
    timezone_name = (payload.get("timezone") or "UTC").strip()
    device_id = (payload.get("deviceId") or "").strip()

    db = firestore.client()
    user = _user_ref(db, user_id)
    event_ref = user.collection("medEvents").document()
    now_iso = datetime.now(timezone.utc).isoformat()

    data = {
        "medId": med_id,
        "taken": taken,
        "scheduledTime": scheduled_time,  # keep as provided (string or None)
        "timeRecorded": firestore.SERVER_TIMESTAMP,
        "timeRecordedIso": now_iso,
        "timezone": timezone_name,
        "reportedBy": device_id,
        "notes": notes,
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }
    event_ref.set(data)

    # Build JSON-safe response (exclude Firestore sentinel fields)
    resp = {
        "id": event_ref.id,
        "medId": med_id,
        "taken": taken,
        "scheduledTime": scheduled_time,
        "timeRecordedIso": now_iso,
        "timezone": timezone_name,
        "reportedBy": device_id,
        "notes": notes,
    }
    return jsonify(resp), 201


@api_bp.post("/api/user/<user_id>/symptomLog")
def create_symptom_log(user_id: str):
    try:
        uid, _claims = verify_firebase_id_token(request)
    except AuthError as exc:
        return jsonify({"error": str(exc)}), 401

    unauthorized = _ensure_owner(uid, user_id)
    if unauthorized:
        err, code = unauthorized
        return jsonify(err), code

    payload = request.get_json(silent=True) or {}

    def _clamp(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            x = float(val)
            if math.isnan(x):
                return None
            return max(0.0, min(10.0, x))
        except Exception:
            return None

    fields = {
        "motor": _clamp(payload.get("motor")),
        "voice": _clamp(payload.get("voice")),
        "cognition": _clamp(payload.get("cognition")),
        "mood": _clamp(payload.get("mood")),
        "fatigue": _clamp(payload.get("fatigue")),
        "onOff": payload.get("onOff") or "Unknown",
        "medEventRef": payload.get("medEventRef"),
        "notes": payload.get("notes"),
        "timezone": (payload.get("timezone") or "UTC").strip(),
        "deviceId": (payload.get("deviceId") or "").strip(),
    }

    db = firestore.client()
    user = _user_ref(db, user_id)
    log_ref = user.collection("symptomLogs").document()
    now_iso = datetime.now(timezone.utc).isoformat()
    data = {
        **fields,
        "timeRecorded": firestore.SERVER_TIMESTAMP,
        "timeRecordedIso": now_iso,
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }
    log_ref.set(data)
    resp = {
        "id": log_ref.id,
        "motor": fields["motor"],
        "voice": fields["voice"],
        "cognition": fields["cognition"],
        "mood": fields["mood"],
        "fatigue": fields["fatigue"],
        "onOff": fields["onOff"],
        "medEventRef": fields["medEventRef"],
        "notes": fields["notes"],
        "timezone": fields["timezone"],
        "deviceId": fields["deviceId"],
        "timeRecordedIso": now_iso,
    }
    return jsonify(resp), 201


@api_bp.get("/api/user/<user_id>/timeline")
def get_timeline(user_id: str):
    try:
        uid, _claims = verify_firebase_id_token(request)
    except AuthError as exc:
        return jsonify({"error": str(exc)}), 401

    unauthorized = _ensure_owner(uid, user_id)
    if unauthorized:
        err, code = unauthorized
        return jsonify(err), code

    start, end = _get_time_range()
    db = firestore.client()
    user = _user_ref(db, user_id)

    med_q = user.collection("medEvents").order_by("timeRecorded")
    sym_q = user.collection("symptomLogs").order_by("timeRecorded")
    if start is not None:
        med_q = med_q.where("timeRecorded", ">=", start)
        sym_q = sym_q.where("timeRecorded", ">=", start)
    if end is not None:
        med_q = med_q.where("timeRecorded", "<=", end)
        sym_q = sym_q.where("timeRecorded", "<=", end)

    med_events = [dict(id=doc.id, **doc.to_dict()) for doc in med_q.stream()]
    symptom_logs = [dict(id=doc.id, **doc.to_dict()) for doc in sym_q.stream()]

    # Also include current medications to resolve names on client
    meds = [dict(id=doc.id, **doc.to_dict()) for doc in user.collection("medications").stream()]
    return jsonify({"medEvents": med_events, "symptomLogs": symptom_logs, "medications": meds}), 200


# -----------------------------
# Exports
# -----------------------------


def _collect_meds_map(user_ref) -> Dict[str, Dict[str, Any]]:
    meds: Dict[str, Dict[str, Any]] = {}
    for doc in user_ref.collection("medications").stream():
        meds[doc.id] = doc.to_dict() or {}
    return meds


@api_bp.get("/api/user/<user_id>/export/csv")
def export_csv(user_id: str):
    try:
        uid, _claims = verify_firebase_id_token(request)
    except AuthError as exc:
        return jsonify({"error": str(exc)}), 401

    unauthorized = _ensure_owner(uid, user_id)
    if unauthorized:
        err, code = unauthorized
        return jsonify(err), code

    start, end = _get_time_range()
    db = firestore.client()
    user = _user_ref(db, user_id)

    med_q = user.collection("medEvents").order_by("timeRecorded")
    sym_q = user.collection("symptomLogs").order_by("timeRecorded")
    if start is not None:
        med_q = med_q.where("timeRecorded", ">=", start)
        sym_q = sym_q.where("timeRecorded", ">=", start)
    if end is not None:
        med_q = med_q.where("timeRecorded", "<=", end)
        sym_q = sym_q.where("timeRecorded", "<=", end)

    meds_map = _collect_meds_map(user)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "eventType",
        "time",
        "timeZone",
        "medId",
        "medName",
        "scheduled",
        "taken",
        "motor",
        "voice",
        "cognition",
        "mood",
        "fatigue",
        "onOff",
        "notes",
        "audioFile",
        "deviceId",
    ])

    for e in med_q.stream():
        d = e.to_dict() or {}
        med_id = d.get("medId") or ""
        med_name = (meds_map.get(med_id) or {}).get("name", "")
        ts_dt = _as_datetime(d.get("timeRecorded")) or _parse_iso8601(d.get("timeRecordedIso"))
        ts_str = _format_local(ts_dt, d.get("timezone"))
        writer.writerow([
            "MED_EVENT",
            ts_str,
            d.get("timezone", ""),
            med_id,
            med_name,
            d.get("scheduledTime", ""),
            str(d.get("taken", "")),
            "",
            "",
            "",
            "",
            "",
            "",
            d.get("notes", ""),
            "",
            d.get("reportedBy", d.get("deviceId", "")),
        ])

    for s in sym_q.stream():
        d = s.to_dict() or {}
        ts_dt = _as_datetime(d.get("timeRecorded")) or _parse_iso8601(d.get("timeRecordedIso"))
        ts_str = _format_local(ts_dt, d.get("timezone"))
        writer.writerow([
            "SYMPTOM_LOG",
            ts_str,
            d.get("timezone", ""),
            "",
            "",
            "",
            "",
            d.get("motor", ""),
            d.get("voice", ""),
            d.get("cognition", ""),
            d.get("mood", ""),
            d.get("fatigue", ""),
            d.get("onOff", ""),
            d.get("notes", ""),
            d.get("audioFilePath", ""),
            d.get("deviceId", ""),
        ])

    output.seek(0)

    # Optional: save to Storage and return signed URL
    save = (request.args.get("save") or "").lower() in ("1", "true", "yes")
    if save:
        bucket = storage.bucket()  # requires storageBucket configured at init
        filename = f"exports/{user_id}/export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
        blob = bucket.blob(filename)
        blob.upload_from_string(output.getvalue(), content_type="text/csv")
        url = blob.generate_signed_url(expiration=timedelta(hours=1), method="GET")
        # Record export metadata
        _user_ref(db, user_id).collection("exports").add(
            {
                "type": "csv",
                "path": filename,
                "url": url,
                "createdAt": firestore.SERVER_TIMESTAMP,
            }
        )
        return jsonify({"url": url, "path": filename}), 200

    return send_file(
        io.BytesIO(output.read().encode()),
        as_attachment=True,
        download_name="export.csv",
        mimetype="text/csv",
    )


# Minimal PDF export using ReportLab (no heavy charting for MVP)
try:  # optional import to keep app usable without reportlab at dev time
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:  # pragma: no cover - optional dependency
    letter = None
    canvas = None


@api_bp.get("/api/user/<user_id>/export/pdf")
def export_pdf(user_id: str):
    try:
        uid, _claims = verify_firebase_id_token(request)
    except AuthError as exc:
        return jsonify({"error": str(exc)}), 401

    unauthorized = _ensure_owner(uid, user_id)
    if unauthorized:
        err, code = unauthorized
        return jsonify(err), code

    if canvas is None or letter is None:
        return jsonify({"error": "reportlab_not_installed"}), 500

    start, end = _get_time_range()
    include_audio = (request.args.get("includeAudio") or "").lower() in ("1", "true", "yes")

    db = firestore.client()
    user = _user_ref(db, user_id)
    user_doc = user.get()
    profile = user_doc.to_dict() if user_doc.exists else {}

    # Fetch quick aggregates for summary
    med_q = user.collection("medEvents").order_by("timeRecorded")
    if start is not None:
        med_q = med_q.where("timeRecorded", ">=", start)
    if end is not None:
        med_q = med_q.where("timeRecorded", "<=", end)
    meds = [d.to_dict() for d in med_q.stream()]
    scheduled = sum(1 for m in meds if m.get("taken") is not None)
    taken = sum(1 for m in meds if m.get("taken") is True)
    adherence_pct = round((taken / scheduled) * 100.0, 1) if scheduled else 0.0

    # Build PDF in-memory
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # Cover
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(72, height - 72, "VocalScan Report")
    pdf.setFont("Helvetica", 10)
    rng = f"{start.isoformat() if start else ''} — {end.isoformat() if end else ''}"
    pdf.drawString(72, height - 92, f"Date range: {rng if rng.strip(' — ') else 'All time'}")
    pdf.drawString(72, height - 108, f"Generated at: {datetime.now(timezone.utc).isoformat()}")
    pdf.drawString(72, height - 124, "Disclaimer: This is a patient-record export. Not a diagnosis.")
    name = (profile or {}).get("displayName") or "(anonymous)"
    pdf.drawString(72, height - 140, f"Patient: {name}")
    pdf.showPage()

    # Summary
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(72, height - 72, "Summary")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(72, height - 96, f"Adherence: {adherence_pct}% ({taken}/{scheduled})")
    pdf.drawString(72, height - 112, f"Audio included: {'Yes' if include_audio else 'No'}")
    pdf.showPage()

    # Finish
    pdf.save()
    buf.seek(0)

    save = (request.args.get("save") or "").lower() in ("1", "true", "yes")
    if save:
        bucket = storage.bucket()
        filename = f"exports/{user_id}/report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf"
        blob = bucket.blob(filename)
        blob.upload_from_string(buf.getvalue(), content_type="application/pdf")
        url = blob.generate_signed_url(expiration=timedelta(hours=1), method="GET")
        _user_ref(db, user_id).collection("exports").add(
            {
                "type": "pdf",
                "path": filename,
                "url": url,
                "createdAt": firestore.SERVER_TIMESTAMP,
            }
        )
        return jsonify({"url": url, "path": filename}), 200

    return send_file(buf, as_attachment=True, download_name="report.pdf", mimetype="application/pdf")


@api_bp.post("/api/user/<user_id>/share")
def create_share_link(user_id: str):
    try:
        uid, _claims = verify_firebase_id_token(request)
    except AuthError as exc:
        return jsonify({"error": str(exc)}), 401

    unauthorized = _ensure_owner(uid, user_id)
    if unauthorized:
        err, code = unauthorized
        return jsonify(err), code

    payload = request.get_json(silent=True) or {}
    storage_path = (payload.get("storagePath") or "").strip()
    if not storage_path:
        return jsonify({"error": "storagePath required"}), 400

    expires_minutes = int(payload.get("expiresMinutes") or 60)
    bucket = storage.bucket()
    blob = bucket.blob(storage_path)
    url = blob.generate_signed_url(expiration=timedelta(minutes=expires_minutes), method="GET")

    db = firestore.client()
    _user_ref(db, user_id).collection("exports").add(
        {
            "type": "share",
            "path": storage_path,
            "url": url,
            "createdAt": firestore.SERVER_TIMESTAMP,
        }
    )
    return jsonify({"url": url, "path": storage_path, "expiresMinutes": expires_minutes}), 200


# -----------------------------
# Insights: wear-off heuristic
# -----------------------------


@dataclass
class WearOffResult:
    schedule_time: str
    occurrences: int
    total_days: int
    mean_peak_minutes_before: Optional[float]
    message: str


def _iter_days(start: datetime, end: datetime) -> Iterable[datetime]:
    cur = datetime(start.year, start.month, start.day, tzinfo=start.tzinfo)
    while cur <= end:
        yield cur
        cur = cur + timedelta(days=1)


def _time_of_day_to_dt(day: datetime, hhmm: str) -> Optional[datetime]:
    try:
        hh, mm = hhmm.split(":")
        return day.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
    except Exception:
        return None


def _score_value(entry: Dict[str, Any]) -> Optional[float]:
    nums: List[float] = []
    for key in ("motor", "voice", "cognition"):
        val = entry.get(key)
        if isinstance(val, (int, float)):
            nums.append(float(val))
    if not nums:
        return None
    # Higher means worse per 0-10 scale
    return sum(nums) / len(nums)


@api_bp.get("/api/user/<user_id>/insights/wearoff")
def wear_off_insights(user_id: str):
    try:
        uid, _claims = verify_firebase_id_token(request)
    except AuthError as exc:
        return jsonify({"error": str(exc)}), 401

    unauthorized = _ensure_owner(uid, user_id)
    if unauthorized:
        err, code = unauthorized
        return jsonify(err), code

    start, end = _get_time_range()
    if start is None or end is None:
        # Default to last 7 days in UTC
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=7)

    threshold = float(request.args.get("threshold", 1.0))
    min_days = int(request.args.get("minDays", 3))

    db = firestore.client()
    user = _user_ref(db, user_id)
    user_doc = user.get()
    user_profile = user_doc.to_dict() if user_doc.exists else {}
    tz_name = _get_user_timezone_name(user_profile)

    # Fetch meds and schedules
    meds = [dict(id=doc.id, **doc.to_dict()) for doc in user.collection("medications").stream()]
    schedules: List[str] = []
    for m in meds:
        for t in (m.get("schedule") or []):
            if isinstance(t, str):
                schedules.append(t)

    # Fetch logs in range
    sym_q = user.collection("symptomLogs").order_by("timeRecorded")
    sym_q = sym_q.where("timeRecorded", ">=", start).where("timeRecorded", "<=", end)
    logs = [dict(id=doc.id, **doc.to_dict()) for doc in sym_q.stream()]

    # Group logs by day (in UTC) and compute baseline medians
    logs_by_day: Dict[str, List[Dict[str, Any]]] = {}
    for entry in logs:
        ts = _as_datetime(entry.get("timeRecorded"))
        if ts is None:
            # try iso field
            ts = _parse_iso8601(entry.get("timeRecordedIso"))
        if ts is None:
            continue
        day_key = ts.date().isoformat()
        logs_by_day.setdefault(day_key, []).append({**entry, "_ts": ts})

    results: List[WearOffResult] = []
    for schedule_time in schedules:
        deltas_before: List[int] = []  # minutes before dose where peak observed
        satisfied_days = 0
        total_days = 0
        for day_key, day_logs in logs_by_day.items():
            total_days += 1
            # Day baseline = median of score values
            values = [v for v in (_score_value(e) for e in day_logs) if v is not None]
            if not values or len(values) < 3:
                continue
            day_baseline = median(values)

            # Determine schedule datetime for this day
            day_dt = datetime.fromisoformat(day_key + "T00:00:00+00:00")
            sched_dt = _time_of_day_to_dt(day_dt, schedule_time)
            if sched_dt is None:
                continue

            # Windows
            pre_start = sched_dt - timedelta(minutes=120)
            pre_end = sched_dt - timedelta(minutes=10)
            post_start = sched_dt + timedelta(minutes=10)
            post_end = sched_dt + timedelta(minutes=90)

            def _norm_scores(start_dt: datetime, end_dt: datetime) -> List[float]:
                vals: List[float] = []
                for e in day_logs:
                    ts = e.get("_ts")
                    if ts is None:
                        continue
                    if start_dt <= ts <= end_dt:
                        v = _score_value(e)
                        if v is not None:
                            vals.append(v - day_baseline)
                return vals

            pre_vals = _norm_scores(pre_start, pre_end)
            post_vals = _norm_scores(post_start, post_end)
            if not pre_vals or not post_vals:
                continue
            pre_avg = sum(pre_vals) / len(pre_vals)
            post_avg = sum(post_vals) / len(post_vals)
            if (pre_avg - post_avg) >= threshold:
                satisfied_days += 1
                # Peak minute before dose (max pre value time)
                peak_val = -1e9
                peak_min_before = None
                for e in day_logs:
                    ts = e.get("_ts")
                    if ts is None or not (pre_start <= ts <= pre_end):
                        continue
                    v = _score_value(e)
                    if v is None:
                        continue
                    v_norm = v - day_baseline
                    if v_norm > peak_val:
                        peak_val = v_norm
                        peak_min_before = int((sched_dt - ts).total_seconds() // 60)
                if peak_min_before is not None and peak_min_before >= 0:
                    deltas_before.append(peak_min_before)

        mean_peak = (sum(deltas_before) / len(deltas_before)) if deltas_before else None
        if satisfied_days >= min_days:
            msg = (
                f"Wearing-off likely begins ~{round(mean_peak) if mean_peak is not None else 'N/A'} minutes "
                f"before your {schedule_time} dose (observed {satisfied_days} of {total_days} days)."
            )
            results.append(
                WearOffResult(
                    schedule_time=schedule_time,
                    occurrences=satisfied_days,
                    total_days=total_days,
                    mean_peak_minutes_before=mean_peak,
                    message=msg,
                )
            )

    return jsonify({"results": [r.__dict__ for r in results]}), 200


