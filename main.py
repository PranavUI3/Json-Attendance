
import json
import os
from pathlib import Path
from datetime import date

from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Check your .env file.")

client = OpenAI(api_key=API_KEY)
DATA_FILE = Path(__file__).with_name("attendance.json")


# ---------------- JSON STORAGE ----------------

def load_data():
    with DATA_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data.get("students"), dict):
        raise ValueError("Invalid attendance.json structure.")

    return data


def save_data(data):
    temp_file = DATA_FILE.with_suffix(".tmp")

    with temp_file.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    temp_file.replace(DATA_FILE)


# ---------------- ATTENDANCE TOOLS ----------------

def list_students():
    data = load_data()

    return [
        {"student_id": sid, "name": student["name"]}
        for sid, student in data["students"].items()
    ]


def get_attendance(student_id):
    data = load_data()
    student = data["students"].get(student_id)

    if student is None:
        return {"error": "Student ID not found."}

    records = student["attendance"]
    total = len(records)
    present = sum(
        record["status"] == "Present"
        for record in records
    )
    absent = total - present

    percentage = round(present / total * 100, 2) if total else 0

    return {
        "student_id": student_id,
        "name": student["name"],
        "total_classes": total,
        "present": present,
        "absent": absent,
        "percentage": percentage
    }


def get_all_attendance():
    data = load_data()
    results = []

    for student_id, student in data["students"].items():
        result = get_attendance(student_id)
        results.append(result)

    return results


def mark_attendance(student_id, attendance_date, status):
    if status not in ("Present", "Absent"):
        return {"error": "Status must be Present or Absent."}

    try:
        date.fromisoformat(attendance_date)
    except ValueError:
        return {"error": "Use date format YYYY-MM-DD."}

    data = load_data()
    student = data["students"].get(student_id)

    if student is None:
        return {"error": "Student ID not found."}

    # Update the record if this date already exists
    for record in student["attendance"]:
        if record["date"] == attendance_date:
            record["status"] = status
            save_data(data)

            return {
                "message": "Attendance updated.",
                "student_id": student_id,
                "date": attendance_date,
                "status": status
            }

    # Otherwise, add a new record
    student["attendance"].append({
        "date": attendance_date,
        "status": status
    })

    save_data(data)

    return {
        "message": "Attendance saved.",
        "student_id": student_id,
        "date": attendance_date,
        "status": status
    }


def add_student(student_id, name):
    student_id = student_id.strip()
    name = name.strip()

    if not student_id or not name:
        return {"error": "Student ID and name are required."}

    data = load_data()

    if student_id in data["students"]:
        return {"error": "Student ID already exists."}

    data["students"][student_id] = {
        "name": name,
        "attendance": []
    }

    save_data(data)

    return {
        "message": "Student registered successfully.",
        "student_id": student_id,
        "name": name
    }


# ---------------- GPT TOOL DEFINITIONS ----------------

TOOLS = [
    {
        "type": "function",
        "name": "list_students",
        "description": "List all registered students.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "get_attendance",
        "description": "Get one student's attendance percentage and totals.",
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string"}
            },
            "required": ["student_id"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "get_all_attendance",
        "description": (
            "Get attendance details for every student. "
            "Use this to compare students or find highest or lowest attendance."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "mark_attendance",
        "description": (
            "Mark or update a student's attendance for a specific date. "
            "Only call when the user explicitly requests a change."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string"},
                "attendance_date": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["Present", "Absent"]
                }
            },
            "required": [
                "student_id",
                "attendance_date",
                "status"
            ],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "add_student",
        "description": (
            "Register a new student. Only call when the user "
            "explicitly requests registration."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string"},
                "name": {"type": "string"}
            },
            "required": ["student_id", "name"],
            "additionalProperties": False
        },
        "strict": True
    }
]


# ---------------- TOOL EXECUTION ----------------

def execute_tool(name, arguments):
    available_tools = {
        "list_students": list_students,
        "get_attendance": get_attendance,
        "get_all_attendance": get_all_attendance,
        "mark_attendance": mark_attendance,
        "add_student": add_student
    }

    function = available_tools.get(name)

    if function is None:
        return {"error": "Unknown tool requested."}

    try:
        return function(**arguments)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        return {"error": str(error)}


# ---------------- AI CHAT FUNCTION ----------------

def ask_ai(question):
    instructions = """
    You are an AI assistant for a student attendance management system.

    Rules:
    1. Use Python tools for actual attendance information.
    2. Never invent student data or attendance percentages.
    3. Use get_attendance for an individual student.
    4. Use get_all_attendance to compare students or find highest
       or lowest attendance.
    5. Only change attendance when the user explicitly requests it.
    6. Ask for missing information instead of guessing.
    7. Report tool errors honestly.
    8. Keep answers short and clear.
    """

    input_items = [
        {"role": "user", "content": question}
    ]

    for _ in range(5):
        response = client.responses.create(
            model=MODEL,
            instructions=instructions,
            input=input_items,
            tools=TOOLS
        )

        tool_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]

        if not tool_calls:
            print("\nAI:", response.output_text)
            return

        input_items.extend(response.output)

        for call in tool_calls:
            try:
                arguments = json.loads(call.arguments)
                result = execute_tool(call.name, arguments)
            except (json.JSONDecodeError, TypeError):
                result = {"error": "Invalid tool arguments."}

            input_items.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result)
            })

    print("AI: Too many tool calls. Please try again.")


# ---------------- MAIN TERMINAL LOOP ----------------

def main():
    print("=" * 45)
    print("   AI ATTENDANCE MANAGEMENT SYSTEM")
    print("=" * 45)

    try:
        load_data()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print("Could not load attendance.json:", error)
        return

    print("Ask questions about student attendance.")
    print("Examples:")
    print("- Show all students")
    print("- What is S001's attendance percentage?")
    print("- Who has the lowest attendance?")
    print("- Mark S001 present on 2026-10-14")
    print("Type 'exit' to quit.")

    while True:
        try:
            question = input("\nYou: ").strip()

            if question.lower() == "exit":
                print("Goodbye!")
                break

            if not question:
                continue

            ask_ai(question)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as error:
            print("Request failed:", error)


if __name__ == "__main__":
    main()
