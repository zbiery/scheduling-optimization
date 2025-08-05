from ortools.linear_solver import pywraplp
from collections import defaultdict
import pandas as pd

def build_time_indexed_milp(data, time_slots):
    """
    Build time-indexed MILP model for onboarding training scheduling
    
    Args:
        data: Dictionary containing sessions, groups, rooms, instructors, assignments
        time_slots: List of time slot indices (e.g., [0, 1, 2, ..., 539] for 90 days * 6 slots/day)
    
    Returns:
        solver: OR-Tools solver object
        variables: Dictionary of decision variables
    """
    solver = pywraplp.Solver.CreateSolver('CBC')
    if not solver:
        raise Exception("CBC solver unavailable.")

    # Extract data - preserving data transformations from original model
    session_ids = data["sessions"]["Session ID"].tolist()
    durations = dict(zip(data["sessions"]["Session ID"], data["sessions"]["Duration (mins)"]))
    durations_in_slots = {sid: dur // 30 for sid, dur in durations.items()}
    ranks = dict(zip(data["sessions"]["Session ID"], data["sessions"]["Order"]))
    
    room_ids = data["rooms"]["Room ID"].tolist()
    room_caps = dict(zip(data["rooms"]["Room ID"], data["rooms"]["Capacity"]))
    
    group_ids = data["groups"]["Group ID"].tolist()
    group_sizes = dict(zip(data["groups"]["Group ID"], data["groups"]["Group Size"]))
    
    # Determine session modality
    in_person_sessions = data["sessions"][data["sessions"]["Modality"] == "in-person"]["Session ID"].tolist()
    virtual_sessions = data["sessions"][data["sessions"]["Modality"] == "virtual"]["Session ID"].tolist()
    
    # Build group-session mappings - preserving structure from original
    group_sessions = defaultdict(list)
    for _, row in data["group_training"].iterrows():
        group_sessions[row["Group ID"]].append(row["Session ID"])
    
    # Build instructor-session mappings - preserving structure from original
    instructor_sessions = defaultdict(list)
    session_instructors = defaultdict(list)
    for _, row in data["instructor_assignments"].iterrows():
        instructor_sessions[row["Instructor ID"]].append(row["Session ID"])
        session_instructors[row["Session ID"]].append(row["Instructor ID"])
    
    max_time = len(time_slots)
    
    # =================
    # DECISION VARIABLES
    # =================
    
    variables = {}
    
    # w[s,t]: Session s starts at time t
    w = {}
    for s in session_ids:
        for t in time_slots:
            if t <= max_time - durations_in_slots[s]:  # Can't start too late
                w[(s, t)] = solver.BoolVar(f'start_{s}_at_{t}')
            else:
                w[(s, t)] = None
    variables['w'] = w
    
    # u[s,t]: Session s is active (running) during time t
    u = {}
    for s in session_ids:
        for t in time_slots:
            u[(s, t)] = solver.BoolVar(f'active_{s}_at_{t}')
    variables['u'] = u
    
    # y[s,r]: In-person session s is assigned to room r
    y = {}
    for s in in_person_sessions:
        for r in room_ids:
            y[(s, r)] = solver.BoolVar(f'session_{s}_room_{r}')
    variables['y'] = y
    
    # z[g,s]: Group g attends session s (for required sessions, this will be forced to 1)
    z = {}
    for g in group_ids:
        for s in session_ids:
            if s in group_sessions[g]:
                z[(g, s)] = solver.BoolVar(f'group_{g}_attends_{s}')
            else:
                z[(g, s)] = None  # Group doesn't need this session
    variables['z'] = z
    
    # =================
    # CONSTRAINTS
    # =================
    
    # 1. Each session starts at exactly one time slot
    for s in session_ids:
        valid_starts = [w[(s, t)] for t in time_slots if w[(s, t)] is not None]
        solver.Add(solver.Sum(valid_starts) == 1)
    
    # 2. Session duration consistency (simplified version)
    for s in session_ids:
        d_s = durations_in_slots[s]
        for t in time_slots:
            # u[s,t] = sum of w[s, t-k] for k = 0 to d_s-1
            start_vars = []
            for k in range(d_s):
                if t - k >= 0 and w.get((s, t - k)) is not None:
                    start_vars.append(w[(s, t - k)])
            
            if start_vars:
                solver.Add(u[(s, t)] == solver.Sum(start_vars))
            else:
                solver.Add(u[(s, t)] == 0)
    
    # 3. Training order dependencies (strict precedence)
    for g in group_ids:
        sessions_for_group = group_sessions[g]
        sessions_sorted = sorted(sessions_for_group, key=lambda x: ranks[x])
        
        for i in range(len(sessions_sorted) - 1):
            s1 = sessions_sorted[i]  # Earlier session
            s2 = sessions_sorted[i + 1]  # Later session
            
            # s2 cannot start until s1 finishes
            # Sum(t * w[s2,t]) >= Sum((t + d_s1) * w[s1,t])
            s2_start_time = solver.Sum([t * w[(s2, t)] for t in time_slots if w[(s2, t)] is not None])
            s1_end_time = solver.Sum([(t + durations_in_slots[s1]) * w[(s1, t)] for t in time_slots if w[(s1, t)] is not None])
            
            solver.Add(s2_start_time >= s1_end_time)
    
    # 4. Group participation constraints
    for g in group_ids:
        for s in session_ids:
            if s in group_sessions[g]:
                # Group must attend required sessions
                solver.Add(z[(g, s)] == 1)
            # Non-required sessions are automatically z[(g,s)] = None
    
    # 5. Group non-overlap constraints
    for g in group_ids:
        for t in time_slots:
            # Each group attends at most one session per time slot
            active_sessions = []
            for s in group_sessions[g]:
                if z[(g, s)] is not None:
                    active_sessions.append(u[(s, t)])
            
            if active_sessions:
                solver.Add(solver.Sum(active_sessions) <= 1)
    
    # 6. Instructor availability constraints
    for instructor_id in instructor_sessions:
        for t in time_slots:
            # Each instructor teaches at most one session per time slot
            sessions_for_instructor = instructor_sessions[instructor_id]
            active_sessions = [u[(s, t)] for s in sessions_for_instructor]
            solver.Add(solver.Sum(active_sessions) <= 1)
    
    # 7. Room assignment constraints
    for s in in_person_sessions:
        # Each in-person session assigned to exactly one room
        room_assignments = [y[(s, r)] for r in room_ids]
        solver.Add(solver.Sum(room_assignments) == 1)
    
    # 8. Room availability constraints
    for r in room_ids:
        for t in time_slots:
            # Each room hosts at most one session per time slot
            sessions_in_room = []
            for s in in_person_sessions:
                # u[s,t] * y[s,r] indicates session s is active at time t in room r
                # We need to linearize this product
                active_in_room = solver.BoolVar(f'session_{s}_active_in_room_{r}_at_{t}')
                solver.Add(active_in_room <= u[(s, t)])
                solver.Add(active_in_room <= y[(s, r)])
                solver.Add(active_in_room >= u[(s, t)] + y[(s, r)] - 1)
                sessions_in_room.append(active_in_room)
            
            solver.Add(solver.Sum(sessions_in_room) <= 1)
    
    # 9. Room capacity constraints
    for s in in_person_sessions:
        # Calculate total attendees (groups + instructors)
        total_group_size = sum(group_sizes[g] for g in group_ids if s in group_sessions[g])
        num_instructors = len(session_instructors[s])
        total_attendees = total_group_size + num_instructors
        
        # Session can only be assigned to rooms with sufficient capacity
        for r in room_ids:
            if total_attendees > room_caps[r]:
                solver.Add(y[(s, r)] == 0)
    
    # =================
    # OBJECTIVE FUNCTION
    # =================
    
    # Maximize total attendance
    attendance_terms = []
    for g in group_ids:
        for s in group_sessions[g]:
            if z[(g, s)] is not None:
                attendance_terms.append(group_sizes[g] * z[(g, s)])
    
    solver.Maximize(solver.Sum(attendance_terms))
    
    return solver, variables

def extract_solution_variables(solver, variables, data, time_slots):
    """
    Extract solution variables in format compatible with original main script
    """
    w = variables['w']
    y = variables['y']
    
    session_ids = data["sessions"]["Session ID"].tolist()
    room_ids = data["rooms"]["Room ID"].tolist()
    in_person_sessions = data["sessions"][data["sessions"]["Modality"] == "in-person"]["Session ID"].tolist()
    
    # Create start_time_vars dict compatible with original model format
    # Create a dummy variable object that has solution_value() method
    class DummyVar:
        def __init__(self, value):
            self._value = value
        def solution_value(self):
            return self._value

    start_time_vars = {}
    for s in session_ids:
        for t in time_slots:
            if w.get((s, t)) is not None and w[(s, t)].solution_value() > 0.5:
                start_time_vars[s] = DummyVar(t)
                break

    # Create room_vars dict compatible with original model format
    room_vars = {}
    for s in in_person_sessions:
        for r in room_ids:
            if y.get((s, r)) is not None:
                room_vars[(s, r)] = DummyVar(y[(s, r)].solution_value())
    
    return start_time_vars, room_vars

def convert_time_slots_to_datetime(schedule_df, start_date='2024-01-01', slot_duration_mins=30):
    """
    Convert time slots to actual datetime for better visualization
    """
    import datetime
    
    start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    
    schedule_df = schedule_df.copy()
    schedule_df['Start DateTime'] = schedule_df['Start Time Slot'].apply(
        lambda slot: start_datetime + datetime.timedelta(minutes=slot * slot_duration_mins)
    )
    schedule_df['End DateTime'] = schedule_df['End Time Slot'].apply(
        lambda slot: start_datetime + datetime.timedelta(minutes=slot * slot_duration_mins)
    )
    
    return schedule_df

def solve(solver):
    """
    Solve the MILP model and return solver and status
    Compatible with original solver interface
    """
    status = solver.Solve()
    return solver, status