import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OPERATOR_REFERENCE = """
You have access to the following table operators. Use ONLY these operators.
Each step must use exactly the JSON format shown — no natural language anywhere.

1. derive_columns
   Adds computed columns via arithmetic.
   {
     "op": "derive_columns",
     "derive": [
       {
         "new_column": "<name>",
         "type": "arithmetic",
         "operation": "add" | "subtract" | "multiply" | "divide" | "floor_divide",
         "left":  {"type": "column" | "literal", "value": "<col_name>" | <number>},
         "right": {"type": "column" | "literal", "value": "<col_name>" | <number>}
       }
     ]
   }

2. filter_rows
   Keeps rows matching ALL conditions (AND logic).
   {
     "op": "filter_rows",
     "conditions": [
       {"column": "<col>", "operator": "<" | ">" | "<=" | ">=" | "==" | "!=" | "in" | "not in", "value": <scalar or list>}
     ]
   }
   Use "in" when filtering one column against multiple values (e.g. make in ["BMW","MERZ","LEXS"]).
   IMPORTANT: never stack multiple == conditions on the same column — use "in" instead.

3. group_and_aggregate
   Groups by columns and computes aggregate metrics.
   {
     "op": "group_and_aggregate",
     "group_by": ["<col>", ...],
     "metrics": [
       {"function": "mean" | "sum" | "count" | "min" | "max", "column": "<col>", "as": "<alias>"}
     ]
   }

4. sort_rows
   Sorts by one or more columns.
   {
     "op": "sort_rows",
     "sort_by": [
       {"column": "<col>", "direction": "asc" | "desc"}
     ]
   }

5. limit_rows
   Returns the first k rows.
   {"op": "limit_rows", "k": <int>}

6. select_columns
   Keeps only the specified columns.
   {"op": "select_columns", "columns": ["<col>", ...]}

7. distinct_rows
   Removes duplicate rows. Optionally scoped to a column subset.
   {"op": "distinct_rows", "columns": ["<col>", ...]}
   Or for full-row dedup: {"op": "distinct_rows"}
"""

SYSTEM_PROMPT = f"""You are a data analysis planner. Given a natural language question and a dataset schema, you generate a structured JSON execution plan.

RULES:
- Use ONLY the operators listed below.
- Every value in the plan must be a concrete number or string — never use vague words like "high", "low", or "large".
- If the question refers to "afternoon" use hours 12–18; "morning" use hours 0–11; "evening" use hours 18–23.
- Do NOT generate Python code.
- The executor runs steps in order; each step's output is the input to the next.
- Return ONLY valid JSON — no explanation, no markdown fences, no extra text.

{OPERATOR_REFERENCE}

Output format:
{{
  "steps": [ <step>, <step>, ... ]
}}
"""


def build_schema_description(df: pd.DataFrame) -> str:
    lines = ["Dataset schema:"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = df[col].dropna().iloc[:3].tolist() if not df[col].dropna().empty else []
        lines.append(f"  - {col} ({dtype}): e.g. {sample}")
    lines.append(f"Total rows: {len(df)}")
    return "\n".join(lines)


def generate_plan(question: str, df: pd.DataFrame) -> dict:
    schema = build_schema_description(df)
    user_message = f"{schema}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model wraps the JSON anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    plan = json.loads(raw)
    return plan
