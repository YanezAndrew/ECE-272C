import pandas as pd


def derive_columns(df: pd.DataFrame, derive: list) -> pd.DataFrame:
    """
    Add new computed columns to the table.

    Each entry in `derive` must have:
      new_column : str
      type       : "arithmetic"
      operation  : "add" | "subtract" | "multiply" | "divide" | "floor_divide"
      left       : {"type": "column"|"literal", "value": ...}
      right      : {"type": "column"|"literal", "value": ...}
    """
    df = df.copy()
    for spec in derive:
        left = _resolve_operand(df, spec["left"])
        right = _resolve_operand(df, spec["right"])
        op = spec["operation"]
        if op == "add":
            df[spec["new_column"]] = left + right
        elif op == "subtract":
            df[spec["new_column"]] = left - right
        elif op == "multiply":
            df[spec["new_column"]] = left * right
        elif op == "divide":
            df[spec["new_column"]] = left / right
        elif op == "floor_divide":
            df[spec["new_column"]] = (left // right)
        else:
            raise ValueError(f"Unknown arithmetic operation: {op}")
    return df


def filter_rows(df: pd.DataFrame, conditions: list) -> pd.DataFrame:
    """
    Keep rows that satisfy all conditions (AND logic).

    Each condition must have:
      column   : str
      operator : "<" | ">" | "<=" | ">=" | "==" | "!="
      value    : scalar
    """
    mask = pd.Series([True] * len(df), index=df.index)
    for cond in conditions:
        col = df[cond["column"]]
        val = cond["value"]
        op = cond["operator"]
        if op == "<":
            mask &= col < val
        elif op == ">":
            mask &= col > val
        elif op == "<=":
            mask &= col <= val
        elif op == ">=":
            mask &= col >= val
        elif op == "==":
            mask &= col == val
        elif op == "!=":
            mask &= col != val
        else:
            raise ValueError(f"Unknown filter operator: {op}")
    return df[mask].reset_index(drop=True)


def group_and_aggregate(df: pd.DataFrame, group_by: list, metrics: list) -> pd.DataFrame:
    """
    Group by one or more columns and compute aggregate metrics.

    Each metric must have:
      function : "mean" | "sum" | "count" | "min" | "max"
      column   : str
      as       : str  (output column name)
    """
    agg_map = {}
    rename_map = {}
    for m in metrics:
        func = m["function"]
        col = m["column"]
        alias = m["as"]
        if col not in agg_map:
            agg_map[col] = []
        agg_map[col].append(func)
        rename_map[(col, func)] = alias

    result = df.groupby(group_by).agg(agg_map).reset_index()
    result.columns = [
        rename_map.get((col, func), f"{col}_{func}") if func else col
        for col, func in result.columns
    ]
    return result


def sort_rows(df: pd.DataFrame, sort_by: list) -> pd.DataFrame:
    """
    Sort the table by one or more columns.

    Each entry in `sort_by` must have:
      column    : str
      direction : "asc" | "desc"
    """
    cols = [s["column"] for s in sort_by]
    ascending = [s["direction"] == "asc" for s in sort_by]
    return df.sort_values(by=cols, ascending=ascending).reset_index(drop=True)


def limit_rows(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return the first k rows."""
    return df.head(k).reset_index(drop=True)


def select_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Keep only the specified columns."""
    return df[columns].reset_index(drop=True)


def distinct_rows(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Remove duplicate rows.

    If `columns` is provided, dedup on those columns only.
    Otherwise dedup across all columns.
    """
    return df.drop_duplicates(subset=columns).reset_index(drop=True)


# --- internal helper ---

def _resolve_operand(df: pd.DataFrame, operand: dict):
    if operand["type"] == "column":
        return df[operand["value"]]
    elif operand["type"] == "literal":
        return operand["value"]
    else:
        raise ValueError(f"Unknown operand type: {operand['type']}")
