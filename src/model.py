from ortools.linear_solver import pywraplp
from collections import defaultdict

def build_model(data, time_slots):
    solver = pywraplp.Solver.CreateSolver('CBC')

    session_ids = data["sessions"]["Session ID"].tolist()
    durations = dict(zip(data["sessions"]["Session ID"], data["sessions"]["Duration (mins)"]))
    durations_in_slots = {sid: dur // 30 for sid, dur in durations.items()}
    max_slot = len(time_slots)
    
    start_time_vars = {}
    for sid in session_ids:
        max_start = max_slot - durations_in_slots[sid]
        start_time_vars[sid] = solver.IntVar(0, max_start, f'start_{sid}')

    group_assignments = data["group_training"]
    group_sessions = defaultdict(list)
    for _, row in group_assignments.iterrows():
        group_sessions[row["Group ID"]].append(row["Session ID"])

    #order constraints for group sessions
    session_orders = dict(zip(data["sessions"]["Session ID"], data["sessions"]["Order"]))
    for group, sessions in group_sessions.items():
        sessions_sorted = sorted(sessions, key=lambda x: session_orders[x])
        for i in range(len(sessions_sorted) - 1):
            sid_early = sessions_sorted[i]
            sid_late = sessions_sorted[i + 1]
            solver.Add(
                start_time_vars[sid_late] >= start_time_vars[sid_early] + durations_in_slots[sid_early]
            )

    #non-overlap constraints for groups & sessions
    M = max_slot + max(durations_in_slots.values()) 

    for group, sessions in group_sessions.items():
        for i in range(len(sessions)):
            for j in range(i + 1, len(sessions)):
                sid1, sid2 = sessions[i], sessions[j]
                d1, d2 = durations_in_slots[sid1], durations_in_slots[sid2]
                y = solver.BoolVar(f'group_{group}_no_overlap_{sid1}_{sid2}')

                solver.Add(start_time_vars[sid1] + d1 <= start_time_vars[sid2] + M * (1 - y))
                solver.Add(start_time_vars[sid2] + d2 <= start_time_vars[sid1] + M * y)

    #non-overlap constraints for instructors
    instructor_assignments = data["instructor_assignments"]
    instructor_to_sessions = defaultdict(list)
    for _, row in instructor_assignments.iterrows():
        instructor_to_sessions[row["Instructor ID"]].append(row["Session ID"])

    for instructor, sessions in instructor_to_sessions.items():
        for i in range(len(sessions)):
            for j in range(i + 1, len(sessions)):
                sid1, sid2 = sessions[i], sessions[j]
                if sid1 not in start_time_vars or sid2 not in start_time_vars:
                    continue
                d1, d2 = durations_in_slots[sid1], durations_in_slots[sid2]
                y = solver.BoolVar(f'instr_{instructor}_no_overlap_{sid1}_{sid2}')
                solver.Add(start_time_vars[sid1] + d1 <= start_time_vars[sid2] + M * (1 - y))
                solver.Add(start_time_vars[sid2] + d2 <= start_time_vars[sid1] + M * y)

    #room assignments
    room_ids = data["rooms"]["Room ID"].tolist()
    room_caps = dict(zip(data["rooms"]["Room ID"], data["rooms"]["Capacity"]))
    in_person_sessions = data["sessions"][data["sessions"]["Modality"] == "in-person"]["Session ID"].tolist()

    room_vars = {}
    for sid in in_person_sessions:
        for rid in room_ids:
            room_vars[(sid, rid)] = solver.BoolVar(f'room_{rid}_for_{sid}')

    #can only assign one room per session
    for sid in in_person_sessions:
        solver.Add(solver.Sum([room_vars[(sid, rid)] for rid in room_ids]) == 1)

    #room non-overlap constraints
    for rid in room_ids:
        sessions_in_room = [sid for sid in in_person_sessions if (sid, rid) in room_vars]
        for i in range(len(sessions_in_room)):
            for j in range(i + 1, len(sessions_in_room)):
                sid1, sid2 = sessions_in_room[i], sessions_in_room[j]
                d1, d2 = durations_in_slots[sid1], durations_in_slots[sid2]
                y = solver.BoolVar(f'room_{rid}_no_overlap_{sid1}_{sid2}')
                
                solver.Add(
                    start_time_vars[sid1] + d1 <= start_time_vars[sid2] + M * (1 - y) + M * (2 - room_vars[(sid1, rid)] - room_vars[(sid2, rid)])
                )
                solver.Add(
                    start_time_vars[sid2] + d2 <= start_time_vars[sid1] + M * y + M * (2 - room_vars[(sid1, rid)] - room_vars[(sid2, rid)])
                )

    #capacity constraints for in-person sessions
    group_sizes = dict(zip(data["groups"]["Group ID"], data["groups"]["Group Size"]))
    session_groups = defaultdict(list)
    for _, row in data["group_training"].iterrows():
        session_groups[row["Session ID"]].append(row["Group ID"])

    for sid in in_person_sessions:
        total_group_size = sum(group_sizes[gid] for gid in session_groups.get(sid, []))
        for rid in room_ids:
            if total_group_size > room_caps[rid]:
                solver.Add(room_vars[(sid, rid)] == 0)

    # Objective: maximize total attendance
    total_attendance_terms = []
    for _, row in data["group_training"].iterrows():
        gid = row["Group ID"]
        sid = row["Session ID"]
        if sid in in_person_sessions:
            total_attendance_terms.append(group_sizes[gid])
        else:
            total_attendance_terms.append(group_sizes[gid])

    solver.Maximize(solver.Sum(total_attendance_terms))

    return solver, start_time_vars, room_vars

