import plotly.graph_objects as go
import pandas as pd
from datetime import timedelta, datetime

def visualize_schedule(
    start_times,
    time_slots,
    sessions_df,
    room_assignments=None,
    instructor_assignments=None,
    group_training=None,
    groups=None,
):
    sessions_df = sessions_df.set_index("Session ID")
    slot_to_dt = {i: ts for i, ts in enumerate(time_slots)}

    def time_to_minutes(t):
        return t.hour * 60 + t.minute

    all_dates = pd.date_range(time_slots[0].date(), time_slots[-1].date(), freq="D")
    date_strs = [d.strftime("%Y-%m-%d") for d in all_dates]

    bars = []

    for sid, start_idx in start_times.items():
        session = sessions_df.loc[sid]
        start_dt = slot_to_dt[start_idx]
        duration_mins = int(session["Duration (mins)"])
        end_dt = start_dt + timedelta(minutes=duration_mins)

        date_str = start_dt.strftime("%Y-%m-%d")

        room = room_assignments.get(sid, "Virtual") if room_assignments else "Virtual"

        if instructor_assignments is not None:
            instructors = ", ".join(
                instructor_assignments[instructor_assignments["Session ID"] == sid]["Instructor ID"].unique()
            )
        else:
            instructors = "N/A"

        group_label = ""
        if group_training is not None and groups is not None:
            gids = group_training[group_training["Session ID"] == sid]["Group ID"].unique()
            if "Group Name" in groups.columns:
                group_names = groups[groups["Group ID"].isin(gids)]["Group Name"].tolist()
                group_label = ", ".join(group_names)
            else:
                group_label = ", ".join(map(str, gids))

        bars.append(dict(
            x=date_str,
            y=duration_mins,
            base=time_to_minutes(start_dt.time()),
            text=f"<b>{session['Session Name']}</b><br>Room: {room}<br>Instructors: {instructors}<br>Groups: {group_label}",
            hoverinfo="text",
        ))

    fig = go.Figure()
    for bar in bars:
        fig.add_trace(go.Bar(
            x=[bar["x"]],
            y=[bar["y"]],
            base=[bar["base"]],
            width=0.8,
            text=[bar["text"]],
            textposition="inside",
            marker_color="lightblue",
            orientation='v',
            hoverinfo="text",
            name=""
        ))

    visible_range = 7

    fig.update_layout(
        title="Training Schedule",
        height=600,
        width=1000,  
        dragmode='pan',  
        barmode='overlay',
        bargap=0.2,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            title="Date",
            type="category",
            categoryorder="array",
            categoryarray=date_strs,
            tickangle=45,
            tickmode='array',
            tickvals=date_strs,
            range=[0, min(visible_range, len(date_strs))], 
            showgrid=True,
            gridcolor="black",
            linecolor="black",
            ticks="outside",
            tickfont=dict(color="black"),
            fixedrange=False
        ),
        yaxis=dict(
            title="Time of Day",
            tickvals=[i for i in range(9 * 60, 17 * 60 + 1, 30)],
            ticktext=[
                (datetime.strptime("09:00", "%H:%M") + timedelta(minutes=i)).strftime("%I:%M %p")
                for i in range(0, (17 - 9) * 60 + 1, 30)
            ],
            range=[9 * 60, 17 * 60],
            autorange=False,
            showgrid=True,
            gridcolor="black",
            linecolor="black",
            ticks="outside",
            tickfont=dict(size=10, color="black"),
        ),
        showlegend=False
    )

    fig.write_html("schedule_vis.html", auto_open=True)
