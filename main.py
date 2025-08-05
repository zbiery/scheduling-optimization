# import pandas as pd
# from datetime import datetime, timedelta

# from src.data_loader import load_data
# from src.model import build_milp_model
# from src.solver import solve
# from src.visualizer import visualize_schedule

# from ortools.linear_solver import pywraplp

# def generate_time_slots(start_date, num_days=30):
#     time_slots = []
#     current = start_date
#     while len(time_slots) < num_days * 6:
#         if current.weekday() < 5:  # Monday to Friday
#             for i in range(6):  # 6 half-hour slots: 9:00 to 12:00
#                 time_slots.append(current.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(minutes=30 * i))
#         current += timedelta(days=1)
#     return time_slots

# # Explain why a session is scheduled where it is
# def explain_schedule_decision(sid, scheduled_starts, data, room_assignments):
#     reasons = []

#     sessions_df = data["sessions"].set_index("Session ID")
#     session_orders = sessions_df["Order"].to_dict()
#     group_sessions = data["group_training"].groupby("Group ID")["Session ID"].apply(list)

#     for group, sessions in group_sessions.items():
#         if sid in sessions:
#             sessions_sorted = sorted(sessions, key=lambda s: session_orders[s])
#             idx = sessions_sorted.index(sid)
#             if idx > 0:
#                 sid_prev = sessions_sorted[idx - 1]
#                 prev_end = scheduled_starts[sid_prev] + sessions_df.loc[sid_prev]["Duration (mins)"] // 30
#                 if scheduled_starts[sid] >= prev_end:
#                     reasons.append(f"Must follow {sid_prev} due to group precedence (Group {group})")

#     instructor_sessions = data["instructor_assignments"].groupby("Instructor ID")["Session ID"].apply(list)
#     for instructor, sessions in instructor_sessions.items():
#         if sid in sessions:
#             for other_sid in sessions:
#                 if other_sid != sid and other_sid in scheduled_starts:
#                     si = scheduled_starts[sid]
#                     sj = scheduled_starts[other_sid]
#                     di = sessions_df.loc[sid]["Duration (mins)"] // 30
#                     dj = sessions_df.loc[other_sid]["Duration (mins)"] // 30
#                     if si < sj + dj and sj < si + di:
#                         reasons.append(f"Avoids conflict with {other_sid} (same instructor: {instructor})")

#     if sid in room_assignments:
#         rid = room_assignments[sid]
#         cap = data["rooms"].set_index("Room ID").loc[rid]["Capacity"]
#         groups = data["group_training"][data["group_training"]["Session ID"] == sid]["Group ID"].tolist()
#         group_sizes = data["groups"].set_index("Group ID").loc[groups]["Group Size"].sum()
#         if group_sizes > cap:
#             reasons.append(f"Not assigned room {rid}: capacity too small for group size {group_sizes}")
#         else:
#             reasons.append(f"Assigned to room {rid} (capacity {cap}) for group size {group_sizes}")
#     else:
#         reasons.append("Virtual session (no room needed)")

#     return reasons

# if __name__ == "__main__":
#     excel_path = "data/Onboarding_Data.xlsx"
#     data = load_data(excel_path)

#     time_slots = generate_time_slots(datetime(2025, 8, 1))

#     solver, start_time_vars, room_vars = build_milp_model(data, time_slots)
#     solver, status = solve(solver)

#     if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
#         print("Feasible schedule found.")

#         # Extract session start times
#         scheduled_starts = {sid: int(var.solution_value()) for sid, var in start_time_vars.items()}

#         # Extract room assignments
#         room_assignments = {
#             sid: rid for (sid, rid), var in room_vars.items() if var.solution_value() > 0.5
#         }

#         # Display schedule
#         print("\nFinal Schedule:")
#         sessions_df = data["sessions"].set_index("Session ID")
#         for sid, start_idx in sorted(scheduled_starts.items(), key=lambda x: x[1]):
#             session = sessions_df.loc[sid]
#             start_time = time_slots[start_idx]
#             duration = int(session["Duration (mins)"])
#             end_time = start_time + timedelta(minutes=duration)
#             room = room_assignments.get(sid, "Virtual")
#             print(f"{sid: <12} | {session['Session Name']: <20} | {room: <10} | {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%H:%M')}")

#         # Display rationale for each session
#         print("\nSchedule Explanations:")
#         for sid in sorted(scheduled_starts, key=lambda s: scheduled_starts[s]):
#             print(f"\nSession {sid} ({sessions_df.loc[sid]['Session Name']}):")
#             reasons = explain_schedule_decision(sid, scheduled_starts, data, room_assignments)
#             for r in reasons:
#                 print("-", r)

#         # Visualization of the schedule
#         visualize_schedule(
#             start_times=scheduled_starts,         
#             time_slots=time_slots,                 
#             sessions_df=data["sessions"],          
#             room_assignments=room_assignments,    
#             instructor_assignments=data["instructor_assignments"],  
#             group_training=data.get("group_training"),              
#             groups=data.get("groups"),                               
#         )


#     else:
#         print("No feasible schedule found.")


################################

# import pandas as pd
# from datetime import datetime, timedelta

# from src.data_loader import load_data
# from src.model_3 import build_time_indexed_milp, solve, extract_solution_variables
# from src.visualizer import visualize_schedule

# from ortools.linear_solver import pywraplp

# def generate_time_slots(start_date, num_days=30):
#     time_slots = []
#     current = start_date
#     while len(time_slots) < num_days * 16:
#         if current.weekday() < 5:  # Monday to Friday
#             for i in range(16):  # 16 half-hour slots: 9:00 to 17:00
#                 time_slots.append(current.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(minutes=30 * i))
#         current += timedelta(days=1)
#     return time_slots

# # Explain why a session is scheduled where it is
# def explain_schedule_decision(sid, scheduled_starts, data, room_assignments):
#     reasons = []

#     sessions_df = data["sessions"].set_index("Session ID")
#     session_orders = sessions_df["Order"].to_dict()
#     group_sessions = data["group_training"].groupby("Group ID")["Session ID"].apply(list)

#     for group, sessions in group_sessions.items():
#         if sid in sessions:
#             sessions_sorted = sorted(sessions, key=lambda s: session_orders[s])
#             idx = sessions_sorted.index(sid)
#             if idx > 0:
#                 sid_prev = sessions_sorted[idx - 1]
#                 prev_end = scheduled_starts[sid_prev] + sessions_df.loc[sid_prev]["Duration (mins)"] // 30
#                 if scheduled_starts[sid] >= prev_end:
#                     reasons.append(f"Must follow {sid_prev} due to group precedence (Group {group})")

#     instructor_sessions = data["instructor_assignments"].groupby("Instructor ID")["Session ID"].apply(list)
#     for instructor, sessions in instructor_sessions.items():
#         if sid in sessions:
#             for other_sid in sessions:
#                 if other_sid != sid and other_sid in scheduled_starts:
#                     si = scheduled_starts[sid]
#                     sj = scheduled_starts[other_sid]
#                     di = sessions_df.loc[sid]["Duration (mins)"] // 30
#                     dj = sessions_df.loc[other_sid]["Duration (mins)"] // 30
#                     if si < sj + dj and sj < si + di:
#                         reasons.append(f"Avoids conflict with {other_sid} (same instructor: {instructor})")

#     if sid in room_assignments:
#         rid = room_assignments[sid]
#         cap = data["rooms"].set_index("Room ID").loc[rid]["Capacity"]
#         groups = data["group_training"][data["group_training"]["Session ID"] == sid]["Group ID"].tolist()
#         group_sizes = data["groups"].set_index("Group ID").loc[groups]["Group Size"].sum()
#         if group_sizes > cap:
#             reasons.append(f"Not assigned room {rid}: capacity too small for group size {group_sizes}")
#         else:
#             reasons.append(f"Assigned to room {rid} (capacity {cap}) for group size {group_sizes}")
#     else:
#         reasons.append("Virtual session (no room needed)")

#     return reasons

# if __name__ == "__main__":
#     excel_path = "data/Onboarding_Data.xlsx"
#     data = load_data(excel_path)

#     # Generate time slots - now using list of indices instead of datetime objects
#     start_date = datetime(2025, 8, 1)
#     datetime_slots = generate_time_slots(start_date)
#     time_slots = list(range(len(datetime_slots)))  # Convert to indices for the model

#     # Build and solve the time-indexed MILP model
#     solver, variables = build_time_indexed_milp(data, time_slots)
#     solver, status = solve(solver)

#     if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
#         print("Feasible schedule found.")

#         # Extract solution variables in compatible format
#         start_time_vars, room_vars = extract_solution_variables(solver, variables, data, time_slots)

#         # Extract session start times
#         scheduled_starts = {sid: int(var.solution_value()) for sid, var in start_time_vars.items()}

#         # Extract room assignments
#         room_assignments = {
#             sid: rid for (sid, rid), var in room_vars.items() if var.solution_value() > 0.5
#         }

#         # Display schedule
#         print("\nFinal Schedule:")
#         sessions_df = data["sessions"].set_index("Session ID")
#         for sid, start_idx in sorted(scheduled_starts.items(), key=lambda x: x[1]):
#             session = sessions_df.loc[sid]
#             start_time = datetime_slots[start_idx]
#             duration = int(session["Duration (mins)"])
#             end_time = start_time + timedelta(minutes=duration)
#             room = room_assignments.get(sid, "Virtual")
#             print(f"{sid: <12} | {session['Session Name']: <20} | {room: <10} | {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%H:%M')}")

#         # Display rationale for each session
#         print("\nSchedule Explanations:")
#         for sid in sorted(scheduled_starts, key=lambda s: scheduled_starts[s]):
#             print(f"\nSession {sid} ({sessions_df.loc[sid]['Session Name']}):")
#             reasons = explain_schedule_decision(sid, scheduled_starts, data, room_assignments)
#             for r in reasons:
#                 print("-", r)

#         # Visualization of the schedule
#         visualize_schedule(
#             start_times=scheduled_starts,         
#             time_slots=datetime_slots,  # Pass datetime slots for visualization           
#             sessions_df=data["sessions"],          
#             room_assignments=room_assignments,    
#             instructor_assignments=data["instructor_assignments"],  
#             group_training=data.get("group_training"),              
#             groups=data.get("groups"),                               
#         )

#     else:
#         print("No feasible schedule found.")

import pandas as pd
from datetime import datetime, timedelta

from src.data_loader import load_data
from src.model_4 import build_time_indexed_milp, build_relaxed_milp, solve, extract_solution_variables, analyze_relaxation_violations
from src.visualizer import visualize_schedule

from ortools.linear_solver import pywraplp

def generate_time_slots(start_date, num_days=30):
    time_slots = []
    current = start_date
    while len(time_slots) < num_days * 16:
        if current.weekday() < 5:  # Monday to Friday
            for i in range(16):  # 6 half-hour slots: 9:00 to 12:00
                time_slots.append(current.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(minutes=30 * i))
        current += timedelta(days=1)
    return time_slots

# Explain why a session is scheduled where it is
def explain_schedule_decision(sid, scheduled_starts, data, room_assignments):
    reasons = []

    sessions_df = data["sessions"].set_index("Session ID")
    session_orders = sessions_df["Order"].to_dict()
    group_sessions = data["group_training"].groupby("Group ID")["Session ID"].apply(list)

    for group, sessions in group_sessions.items():
        if sid in sessions:
            sessions_sorted = sorted(sessions, key=lambda s: session_orders[s])
            idx = sessions_sorted.index(sid)
            if idx > 0:
                sid_prev = sessions_sorted[idx - 1]
                prev_end = scheduled_starts[sid_prev] + sessions_df.loc[sid_prev]["Duration (mins)"] // 30
                if scheduled_starts[sid] >= prev_end:
                    reasons.append(f"Must follow {sid_prev} due to group precedence (Group {group})")

    instructor_sessions = data["instructor_assignments"].groupby("Instructor ID")["Session ID"].apply(list)
    for instructor, sessions in instructor_sessions.items():
        if sid in sessions:
            for other_sid in sessions:
                if other_sid != sid and other_sid in scheduled_starts:
                    si = scheduled_starts[sid]
                    sj = scheduled_starts[other_sid]
                    di = sessions_df.loc[sid]["Duration (mins)"] // 30
                    dj = sessions_df.loc[other_sid]["Duration (mins)"] // 30
                    if si < sj + dj and sj < si + di:
                        reasons.append(f"Avoids conflict with {other_sid} (same instructor: {instructor})")

    if sid in room_assignments:
        rid = room_assignments[sid]
        cap = data["rooms"].set_index("Room ID").loc[rid]["Capacity"]
        groups = data["group_training"][data["group_training"]["Session ID"] == sid]["Group ID"].tolist()
        group_sizes = data["groups"].set_index("Group ID").loc[groups]["Group Size"].sum()
        if group_sizes > cap:
            reasons.append(f"Not assigned room {rid}: capacity too small for group size {group_sizes}")
        else:
            reasons.append(f"Assigned to room {rid} (capacity {cap}) for group size {group_sizes}")
    else:
        reasons.append("Virtual session (no room needed)")

    return reasons

if __name__ == "__main__":
    excel_path = "data/Synthetic_Onboarding_Data.xlsx"
    data = load_data(excel_path)

    # Generate time slots - now using list of indices instead of datetime objects
    start_date = datetime(2025, 8, 1)
    datetime_slots = generate_time_slots(start_date)
    time_slots = list(range(len(datetime_slots)))  # Convert to indices for the model

    # Build and solve the time-indexed MILP model
    solver, variables = build_time_indexed_milp(data, time_slots)
    solver, status = solve(solver)

    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        print("Feasible schedule found with original model.")
        slack_vars = None  # No relaxation needed
        
    else:
        print("No feasible schedule found with original model.")
        print("Trying relaxed model to find best possible solution...")
        
        # Try relaxed model
        solver, variables, slack_vars = build_relaxed_milp(data, time_slots)
        solver, status = solve(solver)
        
        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            print("Relaxed model found a solution.")
            # Analyze violations
            analyze_relaxation_violations(solver, slack_vars, data)
        else:
            print("Even relaxed model failed to find a solution.")
            print("Consider reducing the number of sessions or expanding time windows.")
            exit(1)

    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        print("Feasible schedule found.")

        # Extract solution variables in compatible format
        start_time_vars, room_vars = extract_solution_variables(solver, variables, data, time_slots)

        # Extract session start times
        scheduled_starts = {sid: int(var.solution_value()) for sid, var in start_time_vars.items()}

        # Extract room assignments
        room_assignments = {
            sid: rid for (sid, rid), var in room_vars.items() if var.solution_value() > 0.5
        }

        # Display schedule
        print("\nFinal Schedule:")
        sessions_df = data["sessions"].set_index("Session ID")
        for sid, start_idx in sorted(scheduled_starts.items(), key=lambda x: x[1]):
            session = sessions_df.loc[sid]
            start_time = datetime_slots[start_idx]
            duration = int(session["Duration (mins)"])
            end_time = start_time + timedelta(minutes=duration)
            room = room_assignments.get(sid, "Virtual")
            print(f"{sid: <12} | {session['Session Name']: <20} | {room: <10} | {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%H:%M')}")

        # Display rationale for each session
        print("\nSchedule Explanations:")
        for sid in sorted(scheduled_starts, key=lambda s: scheduled_starts[s]):
            print(f"\nSession {sid} ({sessions_df.loc[sid]['Session Name']}):")
            reasons = explain_schedule_decision(sid, scheduled_starts, data, room_assignments)
            for r in reasons:
                print("-", r)

        # Visualization of the schedule
        visualize_schedule(
            start_times=scheduled_starts,         
            time_slots=datetime_slots,  # Pass datetime slots for visualization           
            sessions_df=data["sessions"],          
            room_assignments=room_assignments,    
            instructor_assignments=data["instructor_assignments"],  
            group_training=data.get("group_training"),              
            groups=data.get("groups"),                               
        )

    else:
        print("No feasible schedule found.")