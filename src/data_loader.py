import pandas as pd

def load_data(excel_path):
    sheets = pd.read_excel(excel_path, sheet_name=None)

    sessions = sheets["Sessions"]
    sessions["Duration (mins)"] = (sessions["Duration (hrs)"] * 60).round().astype(int)
    sessions["Duration (mins)"] = (sessions["Duration (mins)"] / 30).round() * 30
    sessions["Duration (mins)"] = sessions["Duration (mins)"].clip(lower=30, upper=180).astype(int)

    return {
        "sessions": sessions,
        "group_training": sheets["Group_Training_Assignments"],
        "groups": sheets["Groups"],
        "instructors": sheets["Instructors"],
        "instructor_availability": sheets["Instructor_Availability"],
        "rooms": sheets["Rooms"],
        "room_availability": sheets["Room_Availability"],
        "instructor_assignments": sheets["Instructor_Assignments"]
    }
