"""Standalone tests for visualization_node only."""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from HW2.project.nodes import visualization_node, ExecutionResult


def run_test(name: str, question: str, execution_result):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"Question: {question}")
    er = ExecutionResult(execution_result, question)
    print(f"Result type: {er.data_type}")
    print(f"Result preview:\n{str(er.data)[:300]}")
    print("-" * 40)

    state = {
        "question": question,
        "execution_result": er,
    }
    out = visualization_node(state)

    print(f"Decision (visualize): {out.get('visualization_decision')}")
    print(f"Chart type: {out.get('visualization_chart_type')}")
    print(f"Viz error: {out.get('visualization_error')}")

    fig_json = out.get("visualization_figure")
    if fig_json is not None:
        print(f"Figure JSON length: {len(fig_json)} chars")
        print(f"Preview: {fig_json[:120]}...")
    else:
        print("No figure produced (visualization skipped or failed).")


# --- Test 1: comparison — should visualize ---
avg_by_proximity = pd.DataFrame({
    "ocean_proximity": ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
    "median_house_value": [240084.3, 124805.4, 380440.6, 259212.3, 249433.0],
})
run_test(
    "comparison by category",
    "Which ocean proximity category has the highest average median house value?",
    avg_by_proximity,
)

# --- Test 2: trend over time — should visualize ---
sales_by_year = pd.DataFrame({
    "year": list(range(2000, 2016)),
    "total_global_sales": [143.5, 160.2, 175.0, 190.3, 200.1,
                            225.6, 248.9, 272.4, 260.3, 245.1,
                            230.0, 218.4, 200.0, 185.3, 170.1, 155.5],
})
run_test(
    "trend over time",
    "How have total global video game sales changed over the years?",
    sales_by_year,
)

# --- Test 3: scalar — should NOT visualize ---
run_test(
    "scalar answer",
    "What is the average median house value across the dataset?",
    206943.86,
)

# --- Test 4: distribution — should visualize ---
income_series = pd.DataFrame({
    "median_income": [3.5, 4.2, 1.8, 6.1, 2.9, 5.5, 3.3, 7.0, 4.8, 2.1,
                      3.9, 5.1, 1.5, 6.8, 4.0, 3.7, 5.9, 2.4, 4.5, 6.3],
})
run_test(
    "distribution",
    "What is the distribution of median income values?",
    income_series,
)
