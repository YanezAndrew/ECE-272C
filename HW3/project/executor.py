import pandas as pd
from operators import (
    derive_columns,
    filter_rows,
    group_and_aggregate,
    sort_rows,
    limit_rows,
    select_columns,
    distinct_rows,
)

DISPATCH = {
    "derive_columns":     lambda df, s: derive_columns(df, s["derive"]),
    "filter_rows":        lambda df, s: filter_rows(df, s["conditions"]),
    "group_and_aggregate":lambda df, s: group_and_aggregate(df, s["group_by"], s["metrics"]),
    "sort_rows":          lambda df, s: sort_rows(df, s["sort_by"]),
    "limit_rows":         lambda df, s: limit_rows(df, s["k"]),
    "select_columns":     lambda df, s: select_columns(df, s["columns"]),
    "distinct_rows":      lambda df, s: distinct_rows(df, s.get("columns")),
}


def execute(plan: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Run all steps in the plan against df.

    Returns
    -------
    result : pd.DataFrame
        Table produced by the final step.
    trace : list[dict]
        One entry per step: {step, operation, input_rows, output_rows}.
    """
    current = df.copy()
    trace = []

    for i, step in enumerate(plan["steps"]):
        op = step["op"]
        if op not in DISPATCH:
            raise ValueError(f"Unknown operator '{op}' at step {i + 1}")

        input_rows = len(current)
        current = DISPATCH[op](current, step)
        output_rows = len(current)

        trace.append({
            "step": i + 1,
            "operation": op,
            "input_rows": input_rows,
            "output_rows": output_rows,
        })

    return current, trace
