import os
import re
import json
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_ITERATIONS = 5


def _inspect_datasets(dataset_paths: list[str]) -> str:
    """Build a text summary of all datasets for the LLM prompt."""
    parts = []
    for path in dataset_paths:
        try:
            full = pd.read_csv(path, index_col=0)
            sectors = sorted(full["Sector"].dropna().unique().tolist()) if "Sector" in full.columns else []
            price_col = [c for c in full.columns if "PRICE VAR" in c]
            parts.append(
                f"File: {path}\n"
                f"  Load with: pd.read_csv('{path}', index_col=0)\n"
                f"  Rows: {len(full)}  Columns: {len(full.columns)}\n"
                f"  Index: stock ticker symbols (e.g. {list(full.index[:8])})\n"
                f"  Sector column values: {sectors}\n"
                f"  Price/target columns: {price_col}\n"
                f"  ALL column names:\n    {list(full.columns)}\n"
                f"  Sample (first 3 rows, first 6 cols):\n{full.iloc[:3, :6].to_string()}\n"
            )
        except Exception as e:
            parts.append(f"File: {path} — could not read: {e}\n")
    return "\n".join(parts)


def _make_codegen_prompt(
    question: str,
    dataset_paths: list[str],
    dataset_summary: str,
    iteration: int,
    namespace_vars: list[str],
    last_output: str | None,
    validator_feedback: str | None,
) -> list[dict]:
    ns_hint = (
        f"Variables currently in the execution namespace: {namespace_vars}\n"
        if namespace_vars
        else "The execution namespace is empty (first iteration).\n"
    )

    last_hint = (
        f"\nOutput from the last execution step:\n{last_output}\n"
        if last_output
        else ""
    )

    validator_hint = (
        f"\nValidator feedback from previous attempt (address this):\n{validator_feedback}\n"
        if validator_feedback
        else ""
    )

    system = (
        "You are an expert Python data analyst working inside a persistent execution sandbox. "
        "Your job is to write Python code that makes progress toward answering the user's question "
        "about US stock financial data.\n\n"
        "IMPORTANT RULES:\n"
        "- The sandbox maintains state across iterations. Variables, dataframes, and imports from "
        "previous steps are still available — do NOT re-read CSVs if the data is already loaded.\n"
        "- Read CSVs with: pd.read_csv('<path>', index_col=0) so the ticker is the index.\n"
        "- Store the primary result of THIS step in a variable called `result`.\n"
        "- `result` must be a non-empty DataFrame, Series, scalar, or string.\n"
        "- For visualizations, build a Plotly figure and store it as `fig` (use plotly.graph_objects).\n"
        "- Do NOT call fig.show(). Do NOT import matplotlib. Do NOT use print().\n"
        "- Handle NaN values with dropna() or fillna() as needed.\n"
        "- Use ONLY exact column names from the dataset summaries below.\n"
        "- For cross-year analysis, join DataFrames on the index (ticker symbol).\n"
        "- Write clean, correct Python. No markdown fences, no explanation — only the code.\n\n"
        f"Dataset summaries:\n{dataset_summary}\n\n"
        f"{ns_hint}"
        f"{last_hint}"
        f"{validator_hint}"
    )

    user = f"Iteration {iteration}. Question: {question}\n\nWrite the next Python code step."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _make_stopping_prompt(
    question: str, last_output: str, iteration: int
) -> list[dict]:
    system = (
        "You are evaluating whether an iterative data analysis is complete.\n"
        "Return a JSON object with exactly two keys:\n"
        '  "done": true or false\n'
        '  "reason": one sentence explaining why\n'
        "done=true if: the question is fully answered, results are non-empty and meaningful, "
        "and no obvious gaps remain.\n"
        "done=false if: the result is empty, an error occurred, the question is only partially "
        "answered, or more analysis steps are clearly needed.\n"
        "Return ONLY valid JSON. No markdown, no explanation."
    )
    user = (
        f"Question: {question}\n\n"
        f"Iteration: {iteration}\n\n"
        f"Latest execution output:\n{last_output}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _make_answer_prompt(question: str, execution_summary: str) -> list[dict]:
    system = (
        "You are a financial data analyst. Generate a clear, concise, well-structured natural "
        "language answer based strictly on the execution results provided. "
        "Cite actual numbers from the results. Do not speculate beyond the data."
    )
    user = (
        f"Question: {question}\n\n"
        f"Execution results summary:\n{execution_summary}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _make_viz_prompt(question: str, result_description: str) -> list[dict]:
    system = (
        "You are a data visualization expert. Decide whether a Plotly chart would add meaningful "
        "value given the question and result. If yes, specify chart parameters.\n"
        "Respond with ONLY valid JSON — no markdown:\n"
        '{"visualize": true|false, "chart_type": "bar|line|scatter|pie|box|histogram", '
        '"x_col": "<column or empty>", "y_col": "<column or empty>", "title": "<title>"}'
    )
    user = f"Question: {question}\n\nResult description:\n{result_description}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _strip_fences(code: str) -> str:
    code = re.sub(r"^```(?:python)?\n?", "", code.strip())
    code = re.sub(r"\n?```$", "", code)
    return code.strip()


def _describe_result(result) -> str:
    if isinstance(result, pd.DataFrame):
        return (
            f"DataFrame — shape {result.shape}, columns: {list(result.columns)}\n"
            f"{result.head(10).to_string()}"
        )
    if isinstance(result, pd.Series):
        return f"Series — length {len(result)}, name: {result.name}\n{result.head(10).to_string()}"
    return str(result)[:1000]


def run(
    question: str,
    dataset_paths: list[str],
    validator_feedback: str | None = None,
) -> dict:
    """
    Run the iterative analytics agent.

    Returns
    -------
    dict with keys:
      final_answer    : str
      plots           : list[dict]  — each {"title": str, "plotly_json": str}
      execution_trace : list[dict]  — one entry per iteration
      iterations      : int
    """
    dataset_summary = _inspect_datasets(dataset_paths)
    namespace: dict = {}
    execution_trace = []
    plots = []
    last_output: str | None = None
    cumulative_summary_parts = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        namespace_vars = [k for k in namespace if not k.startswith("_")]

        # --- code generation ---
        messages = _make_codegen_prompt(
            question=question,
            dataset_paths=dataset_paths,
            dataset_summary=dataset_summary,
            iteration=iteration,
            namespace_vars=namespace_vars,
            last_output=last_output,
            validator_feedback=validator_feedback if iteration == 1 else None,
        )
        response = client.chat.completions.create(
            model="gpt-4o", temperature=0, messages=messages
        )
        code = _strip_fences(response.choices[0].message.content)

        # --- execution ---
        exec_error: str | None = None
        result = None
        fig = None
        try:
            exec(code, namespace)  # noqa: S102
            result = namespace.get("result")
            fig = namespace.get("fig")
        except Exception as e:
            exec_error = str(e)

        result_description = _describe_result(result) if result is not None else ""
        step_output = result_description if not exec_error else f"ERROR: {exec_error}"
        last_output = step_output

        trace_entry = {
            "iteration": iteration,
            "code": code,
            "result_description": result_description,
            "error": exec_error,
        }
        execution_trace.append(trace_entry)

        if result is not None:
            cumulative_summary_parts.append(
                f"[Iteration {iteration}]\n{result_description}"
            )

        # --- capture plotly figure if generated ---
        if fig is not None:
            try:
                plots.append({
                    "title": (
                        fig.layout.title.text
                        if fig.layout.title and fig.layout.title.text
                        else question
                    ),
                    "plotly_json": fig.to_json(),
                })
                namespace.pop("fig", None)
            except Exception:
                pass

        # --- stopping condition ---
        if exec_error:
            # always retry on error unless out of iterations
            if iteration == MAX_ITERATIONS:
                break
            continue

        stop_messages = _make_stopping_prompt(question, step_output, iteration)
        stop_response = client.chat.completions.create(
            model="gpt-4o", temperature=0, messages=stop_messages
        )
        raw_stop = stop_response.choices[0].message.content.strip()
        raw_stop = re.sub(r"^```(?:json)?\n?", "", raw_stop)
        raw_stop = re.sub(r"\n?```$", "", raw_stop)
        try:
            stop_decision = json.loads(raw_stop)
        except Exception:
            stop_decision = {"done": False, "reason": "parse error"}

        execution_trace[-1]["stop_decision"] = stop_decision

        if stop_decision.get("done", False):
            break

    # --- generate final answer ---
    cumulative_summary = "\n\n".join(cumulative_summary_parts) or "No results produced."
    answer_messages = _make_answer_prompt(question, cumulative_summary)
    answer_response = client.chat.completions.create(
        model="gpt-4o", temperature=0, messages=answer_messages
    )
    final_answer = answer_response.choices[0].message.content.strip()

    return {
        "final_answer": final_answer,
        "plots": plots,
        "execution_trace": execution_trace,
        "iterations": len(execution_trace),
    }


if __name__ == "__main__":
    import sys

    datasets = [
        "datasets/2014_Financial_Data.csv",
        "datasets/2015_Financial_Data.csv",
    ]
    question = sys.argv[1] if len(sys.argv) > 1 else (
        "Which sector had the highest average return on equity in 2014?"
    )
    output = run(question, datasets)
    print(f"\nAnswer: {output['final_answer']}")
    print(f"\nIterations: {output['iterations']}")
    print(f"Plots generated: {len(output['plots'])}")
    for t in output["execution_trace"]:
        print(f"\n[Iteration {t['iteration']}]")
        print(f"  Error: {t['error']}")
        print(f"  Stop decision: {t.get('stop_decision')}")
