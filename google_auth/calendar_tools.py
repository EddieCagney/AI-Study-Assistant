from datetime import datetime, timedelta

from strands import tool

from google_auth.auth import get_calendar_service


def _format_time_12h(time_str: str) -> str:
    """Convert a 24-hour HH:MM string to H:MM AM/PM for display."""
    try:
        hour_str, minute_str = time_str.split(":")
        hour = int(hour_str)
        minute = int(minute_str)
        suffix = "AM" if hour < 12 else "PM"
        hour_12 = hour % 12 or 12
        return f"{hour_12}:{minute:02d} {suffix}"
    except Exception:
        return time_str


@tool
def get_calendar_events(days_ahead: int = 7) -> str:
    """Get Google Calendar events for the upcoming days to check availability.

    Returns a list of calendar events with their times so you can determine
    when the user is free to schedule meetings.

    Args:
        days_ahead: Number of days ahead to check (default 7)
    """
    service = get_calendar_service()

    now = datetime.utcnow()
    time_min = now.isoformat() + "Z"
    time_max = (now + timedelta(days=days_ahead)).isoformat() + "Z"

    events_result = service.events().list(
        calendarId="primary",
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy="startTime",
    ).execute()

    events = events_result.get("items", [])
    if not events:
        return f"No calendar events found in the next {days_ahead} days. The user appears to be fully available."

    events_by_date = {}

    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date"))
        end = event["end"].get("dateTime", event["end"].get("date"))
        summary = event.get("summary", "(No title)")
        status = event.get("status", "confirmed")

        if "T" in start:
            date_key = start.split("T")[0]
            start_time = start.split("T")[1][:5]
            end_time = end.split("T")[1][:5] if "T" in end else "all day"
            time_str = f"{_format_time_12h(start_time)} - {_format_time_12h(end_time)}"
        else:
            date_key = start
            time_str = "All day"

        if date_key not in events_by_date:
            events_by_date[date_key] = []

        events_by_date[date_key].append(f"  {time_str}: {summary} (status: {status})")

    output_lines = [f"Calendar events for the next {days_ahead} days:\n"]
    for date in sorted(events_by_date.keys()):
        output_lines.append(f"{date}:")
        output_lines.extend(events_by_date[date])
        output_lines.append("")

    return "\n".join(output_lines)


@tool
def create_calendar_event(
    summary: str,
    date: str,
    start_time: str,
    end_time: str,
    description: str = "",
) -> str:
    """Create a Google Calendar event for a study session.

    Args:
        summary: Title of the event (e.g. "Study: Transformer Architecture")
        date: ISO date string YYYY-MM-DD
        start_time: Start time HH:MM (24-hour)
        end_time: End time HH:MM (24-hour)
        description: Optional description / notes for the event
    """
    service = get_calendar_service()

    event = {
        "summary": summary,
        "description": description,
        "start": {
            "dateTime": f"{date}T{start_time}:00",
            "timeZone": "America/Los_Angeles",
        },
        "end": {
            "dateTime": f"{date}T{end_time}:00",
            "timeZone": "America/Los_Angeles",
        },
        "colorId": "2",  # sage green — visually distinct study blocks
    }

    created = service.events().insert(calendarId="primary", body=event).execute()
    return f"Created: {created.get('summary')} on {date} {start_time}–{end_time} (id: {created.get('id')})"
