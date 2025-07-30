import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime

def visualize_schedule(start_times, time_slots, sessions_df, room_assignments=None, instructor_assignments=None):
    fig, ax = plt.subplots(figsize=(24, 16)) 
    sessions_df = sessions_df.set_index("Session ID")

    slot_to_time = {i: ts for i, ts in enumerate(time_slots)}
    session_rects = []

    # Build metadata per session
    for sid, start_idx in start_times.items():
        session = sessions_df.loc[sid]
        duration_minutes = session["Duration (mins)"]
        duration_slots = duration_minutes // 30
        start_time = slot_to_time[start_idx]
        end_time = start_time + pd.Timedelta(minutes=duration_minutes)

        instructors = (
            ", ".join(instructor_assignments[instructor_assignments["Session ID"] == sid]["Instructor ID"].unique())
            if instructor_assignments is not None else "N/A"
        )

        session_rects.append({
            "sid": sid,
            "start_time": start_time,
            "end_time": end_time,
            "duration_slots": duration_slots,
            "session_name": session["Session Name"],
            "room": room_assignments.get(sid, "Virtual") if room_assignments else "Virtual",
            "instructors": instructors
        })

    y_labels = pd.date_range("2000-01-01 09:00", "2000-01-01 12:00", freq="30min").time
    y_positions = {t: i for i, t in enumerate(y_labels)}

    # Plot each session
    for rect in session_rects:
        day = rect["start_time"].date()
        time = rect["start_time"].time()
        start_y = y_positions[time]
        bar_y = start_y - 0.4
        bar_height = rect["duration_slots"]

        ax.broken_barh(
            [(float(mdates.date2num(day)), 0.8)],
            (bar_y, bar_height),
            facecolors="tab:blue",
            edgecolors="black"
        )

        label = f"{rect['session_name']}\nRoom: {rect['room']}\nInstructor(s): {rect['instructors']}"
        ax.text(
            float(mdates.date2num(day)) + 0.05,
            bar_y + bar_height / 2,
            label,
            va="center",
            ha="left",
            fontsize=10,
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8)
        )

    all_dates = sorted({ts.date() for ts in time_slots})
    ax.set_xticks([float(mdates.date2num(d)) for d in all_dates])
    ax.set_xticklabels([d.strftime("%m-%d") for d in all_dates], rotation=90)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels([t.strftime("%H:%M") for t in y_labels])
    ax.set_ylabel("Time of Day")
    ax.set_xlabel("Date")
    ax.set_title("Training Session Schedule")
    ax.grid(True, axis="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.show()
