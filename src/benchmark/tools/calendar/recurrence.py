"""Bounded daily, weekly and first-weekday monthly recurrence."""

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
import re


class RecurrenceError(ValueError):
    pass


def local_time(value, zone, all_day):
    if all_day:
        return datetime.combine(date.fromisoformat(value), time.min, zone)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(zone)


def rule_date(value, zone, all_day):
    # RFC compact dates or datetimes, plus native ISO exception timestamps.
    if re.fullmatch(r"\d{8}", value):
        if not all_day:
            raise RecurrenceError("Timed recurrence requires datetime values")
        return datetime.strptime(value, "%Y%m%d").replace(tzinfo=zone)
    if re.fullmatch(r"\d{8}T\d{6}Z?", value):
        if all_day:
            raise RecurrenceError("All-day recurrence requires date values")
        utc = value.endswith("Z")
        return (
            datetime.strptime(value.rstrip("Z"), "%Y%m%dT%H%M%S")
            .replace(tzinfo=timezone.utc if utc else zone)
            .astimezone(zone)
        )
    return local_time(value, zone, all_day)


def starts(event, rules, upper, *, budget=None):
    """Yield the recurrence set in local wall time, excluding EXDATEs."""
    budget = [20000] if budget is None else budget
    zone = ZoneInfo(event["timezone"])
    all_day = bool(event["all_day"])
    first = local_time(event["start_time"], zone, all_day)
    adds, excludes, recurrence = set(), set(), None
    for line in rules:
        head, sep, body = line.partition(":")
        if not sep:
            raise RecurrenceError("Invalid recurrence line")
        pieces = head.split(";")
        kind = pieces.pop(0)
        parameters = dict(p.split("=", 1) for p in pieces)
        if (
            len(parameters) != len(pieces)
            or set(parameters) - {"TZID", "VALUE"}
            or parameters.get("VALUE", "DATE" if all_day else "DATE-TIME")
            != ("DATE" if all_day else "DATE-TIME")
        ):
            raise RecurrenceError("Unsupported recurrence date parameters")
        value_zone = ZoneInfo(parameters.get("TZID", event["timezone"]))
        if kind in ("RDATE", "EXDATE"):
            target = adds if kind == "RDATE" else excludes
            target.update(
                rule_date(v, value_zone, all_day).astimezone(timezone.utc)
                for v in body.split(",")
            )
        elif kind == "RRULE" and recurrence is None and not parameters:
            pairs = [p.split("=", 1) for p in body.split(";")]
            recurrence = dict(pairs)
            if len(recurrence) != len(pairs) or set(recurrence) - {
                "FREQ",
                "INTERVAL",
                "COUNT",
                "UNTIL",
                "BYDAY",
                "BYSETPOS",
                "WKST",
            }:
                raise RecurrenceError("Unsupported RRULE fields")
        else:
            raise RecurrenceError("Only one RRULE plus RDATE and EXDATE is supported")
    generated = {first.astimezone(timezone.utc)}
    if recurrence is not None:
        freq = recurrence.get("FREQ")
        if (
            freq not in ("DAILY", "WEEKLY", "MONTHLY")
            or recurrence.get("WKST", "MO") != "MO"
        ):
            raise RecurrenceError("Unsupported recurrence frequency or week start")
        interval = int(recurrence.get("INTERVAL", "1"))
        count = int(recurrence.get("COUNT", "2147483647"))
        if (
            interval < 1
            or count < 1
            or ("COUNT" in recurrence and "UNTIL" in recurrence)
        ):
            raise RecurrenceError("Invalid recurrence interval or bound")
        names = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
        days = recurrence.get(
            "BYDAY", names[first.weekday()] if freq == "WEEKLY" else ",".join(names)
        ).split(",")
        if not set(days) <= set(names):
            raise RecurrenceError("Ordinal BYDAY values are unsupported")
        weekdays = {names.index(d) for d in days}
        if freq == "MONTHLY":
            if recurrence.get("BYSETPOS") != "1" or len(weekdays) != 1:
                raise RecurrenceError(
                    "Monthly recurrence requires one weekday and BYSETPOS=1"
                )
        elif "BYSETPOS" in recurrence:
            raise RecurrenceError("BYSETPOS is only supported for monthly recurrence")
        until = (
            rule_date(recurrence["UNTIL"], zone, all_day).astimezone(timezone.utc)
            if "UNTIL" in recurrence
            else None
        )
        generated = set()
        current, seen = first, 0
        week_zero = first.date() - timedelta(days=first.weekday())
        while current.timestamp() < upper and seen < count:
            budget[0] -= 1
            if budget[0] < 0:
                raise RecurrenceError("Recurrence scan limit exceeded")
            instant = current.astimezone(timezone.utc)
            if until is not None and instant > until:
                break
            distance = (
                current.date() - (first.date() if freq == "DAILY" else week_zero)
            ).days
            if freq == "MONTHLY":
                months = (current.year - first.year) * 12 + current.month - first.month
                eligible = months % interval == 0 and current.day <= 7
            else:
                eligible = (
                    distance % interval == 0
                    if freq == "DAILY"
                    else (distance // 7) % interval == 0
                )
            # RFC invalid local times are skipped and do not consume COUNT.
            valid = instant.astimezone(zone).replace(tzinfo=None) == current.replace(
                tzinfo=None
            )
            if eligible and current.weekday() in weekdays and valid:
                generated.add(instant)
                seen += 1
            current += timedelta(days=1)
    return sorted((generated | adds) - excludes)


def expand(rows, child_rules, lower, upper, *, cap=5000):
    """Return occurrences with stored IDs and exact series/override identities."""
    overrides = {}
    parents = {
        (r["calendar_id"], r["id"]): r for r in rows if not r["recurring_event_id"]
    }
    for row in rows:
        if row["recurring_event_id"]:
            parent = parents.get((row["calendar_id"], row["recurring_event_id"]), row)
            zone = ZoneInfo(parent["timezone"])
            original = local_time(
                row["original_start_time"], zone, bool(row["all_day"])
            ).timestamp()
            key = (row["calendar_id"], row["recurring_event_id"], original)
            if key in overrides:
                raise RecurrenceError("Ambiguous stored recurrence override")
            overrides[key] = row
    result = []
    budget = [20000]
    for row in rows:
        rules = child_rules(row)
        if row["recurring_event_id"] and rules:
            raise RecurrenceError("Recurrence rules on an override are unsupported")
        if not rules:
            zone = ZoneInfo(row["timezone"])
            first = local_time(
                row["start_time"], zone, bool(row["all_day"])
            ).timestamp()
            last = local_time(row["end_time"], zone, bool(row["all_day"])).timestamp()
            if last > lower and first < upper:
                result.append(dict(row, start_utc=first, end_utc=last))
        else:
            zone = ZoneInfo(row["timezone"])
            first = local_time(row["start_time"], zone, bool(row["all_day"]))
            last = local_time(row["end_time"], zone, bool(row["all_day"]))
            duration = (
                (last.replace(tzinfo=None) - first.replace(tzinfo=None))
                if row["all_day"]
                else last.astimezone(timezone.utc) - first.astimezone(timezone.utc)
            )
            for instant in starts(row, rules, upper, budget=budget):
                start = instant.astimezone(zone)
                end = (
                    start + duration
                    if row["all_day"]
                    else (instant + duration).astimezone(zone)
                )
                if end.timestamp() <= lower or start.timestamp() >= upper:
                    continue
                key = (row["calendar_id"], row["id"], start.timestamp())
                if key in overrides:
                    continue  # Stored moved/cancelled rows replace generated slots.
                value = dict(
                    row,
                    start_utc=start.timestamp(),
                    end_utc=end.timestamp(),
                    start_time=(
                        start.date().isoformat()
                        if row["all_day"]
                        else start.isoformat()
                    ),
                    end_time=(
                        end.date().isoformat() if row["all_day"] else end.isoformat()
                    ),
                    recurring_event_id=row["id"],
                    original_start_time=(
                        start.date().isoformat()
                        if row["all_day"]
                        else start.isoformat()
                    ),
                    occurrence=True,
                )
                result.append(value)
                if len(result) > cap:
                    raise RecurrenceError("Expanded occurrence limit exceeded")
        if len(result) > cap:
            raise RecurrenceError("Expanded occurrence limit exceeded")
    return sorted(
        result,
        key=lambda r: (r["start_utc"], r["end_utc"], r["calendar_id"].lower(), r["id"]),
    )
