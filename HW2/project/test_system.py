"""
End-to-end system test — 5 questions including one multi-turn interaction.
Simulates exactly what the backend does with sessions, but without the frontend.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from HW2.project.agent import agent, build_initial_state
from HW2.project.nodes import ExecutionResult

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOUSING_CSV = os.path.join(SCRIPT_DIR, "datasets", "housing.csv")
VGSALES_CSV = os.path.join(SCRIPT_DIR, "datasets", "custom_dataset.csv")


def run_turn(question: str, csv_path: str, previous_result: ExecutionResult | None = None) -> dict:
    state = build_initial_state(question, csv_path, previous_result)
    final = agent.invoke(state)
    return {
        "question": question,
        "answer": final.get("final_answer", ""),
        "execution_result": final.get("execution_result"),
        "visualization_decision": final.get("visualization_decision", False),
        "visualization_chart_type": final.get("visualization_chart_type", "none"),
        "visualization_figure": final.get("visualization_figure"),
        "evaluation": final.get("evaluation", "FAIL"),
    }


def print_result(result: dict, turn: int):
    print(f"\n{'='*60}")
    print(f"TURN {turn}: {result['question']}")
    print(f"{'='*60}")
    print(f"Evaluation : {result['evaluation']}")
    print(f"Answer     : {result['answer'][:300]}")
    er: ExecutionResult | None = result["execution_result"]
    if er:
        print(f"Result type: {er.data_type}")
        if hasattr(er.data, "shape"):
            print(f"Result shape: {er.data.shape}")
    print(f"Visualize  : {result['visualization_decision']} ({result['visualization_chart_type']})")
    if result["visualization_figure"]:
        print(f"Figure JSON: {len(result['visualization_figure'])} chars")


# ---------------------------------------------------------------------------
# Q1 — Simple scalar (housing)
# ---------------------------------------------------------------------------
r1 = run_turn(
    "What is the average median house value across the dataset?",
    HOUSING_CSV,
)
print_result(r1, 1)

# ---------------------------------------------------------------------------
# Q2 — Comparison across categories (housing)
# ---------------------------------------------------------------------------
r2 = run_turn(
    "Which ocean proximity category has the highest average median house value?",
    HOUSING_CSV,
)
print_result(r2, 2)

# ---------------------------------------------------------------------------
# Q3 — Trend over time (vgsales)
# ---------------------------------------------------------------------------
r3 = run_turn(
    "How have total global video game sales changed over the years?",
    VGSALES_CSV,
)
print_result(r3, 3)

# ---------------------------------------------------------------------------
# Q4 — Multi-turn part 1: genre breakdown (vgsales)
# ---------------------------------------------------------------------------
r4 = run_turn(
    "How do average global sales vary across different genres?",
    VGSALES_CSV,
)
print_result(r4, 4)

# ---------------------------------------------------------------------------
# Q5 — Multi-turn part 2: follow-up on Q4's result (no CSV re-read)
# ---------------------------------------------------------------------------
r5 = run_turn(
    "Now show only the top 3 genres from the previous result, sorted by average sales",
    VGSALES_CSV,
    previous_result=r4["execution_result"],
)
print_result(r5, 5)

print("\n" + "="*60)
print("DONE")
