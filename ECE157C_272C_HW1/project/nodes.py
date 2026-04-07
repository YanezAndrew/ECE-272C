import os
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)


def _get_csv_info(csv_path: str) -> str:
    """Read first few rows of a CSV to provide column info to the LLM."""
    try:
        import pandas as _pd
        df = _pd.read_csv(csv_path, nrows=5)
        cols = list(df.columns)
        dtypes = df.dtypes.to_dict()
        info = f"Columns: {cols}\nDtypes: {dtypes}\nSample (first 3 rows):\n{df.head(3).to_string()}"
        return info
    except Exception:
        return ""


def codegen_node(state: dict) -> dict:
    """Generate Python/pandas code to answer the question."""
    question = state["question"]
    csv_path = state["csv_path"]
    csv_info = _get_csv_info(csv_path)

    system_prompt = (
        "You are a Python data analyst. Write Python code using pandas to answer "
        "the user's question about a CSV dataset.\n\n"
        f"Dataset info:\n{csv_info}\n\n"
        "Rules:\n"
        f"- Read the CSV from: '{csv_path}'\n"
        "- Store the final output in a variable called `result`.\n"
        "- `result` should be a simple, printable value: a string, number, "
        "pandas DataFrame, or Series.\n"
        "- `result` must NEVER be empty or None. Always produce meaningful output.\n"
        "- For analytical questions, compute relevant statistics and store a "
        "DataFrame or descriptive string summarizing the findings in `result`.\n"
        "- Do NOT use print(). Do NOT use plt.show(). Do NOT import matplotlib.\n"
        "- Only output the Python code. No markdown fences, no explanation.\n"
        "- Handle missing values (NaN) gracefully using dropna() or fillna() as needed.\n"
        "- Use ONLY the exact column names listed above. Do NOT guess column names.\n"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ])

    code = response.content.strip()
    # Strip markdown code fences if present
    code = re.sub(r"^```(?:python)?\n?", "", code)
    code = re.sub(r"\n?```$", "", code)

    state["generated_code"] = code
    state["retry_count"] = state.get("retry_count", 0)
    return state


def execute_node(state: dict) -> dict:
    """Execute the generated code and capture the result."""
    code = state["generated_code"]
    env = {}
    try:
        exec(code, env)
        result = env.get("result")
        if result is None:
            state["execution_result"] = None
            state["execution_error"] = (
                "The code did not store anything in `result`. "
                "You MUST assign the final output to a variable named `result`."
            )
        else:
            # Check if result is an empty DataFrame or Series
            empty = False
            try:
                import pandas as _pd
                if isinstance(result, (_pd.DataFrame, _pd.Series)) and result.empty:
                    empty = True
            except Exception:
                pass
            if empty:
                state["execution_result"] = None
                state["execution_error"] = (
                    "The `result` variable is an empty DataFrame/Series. "
                    "Adjust the code logic so `result` contains meaningful data."
                )
            else:
                state["execution_result"] = result
                state["execution_error"] = None
    except Exception as e:
        state["execution_result"] = None
        state["execution_error"] = str(e)
    return state


def evaluate_node(state: dict) -> dict:
    """Evaluate whether the execution result correctly answers the question."""
    question = state["question"]
    execution_result = state["execution_result"]
    execution_error = state["execution_error"]
    generated_code = state["generated_code"]

    if execution_error:
        evaluation_input = (
            f"The code raised an error:\n{execution_error}\n\n"
            f"Code:\n{generated_code}\n\n"
            f"Question: {question}\n\n"
            "Return ONLY the word FAIL."
        )
    else:
        result_str = str(execution_result)[:2000]
        evaluation_input = (
            f"Question: {question}\n\n"
            f"Generated code:\n{generated_code}\n\n"
            f"Execution result:\n{result_str}\n\n"
            "Evaluate whether the execution produced a non-empty, reasonable result "
            "that addresses the question. A result is PASS if it contains relevant data, "
            "even if the analysis could be more thorough. FAIL only if the result is "
            "empty, None, an error, or completely unrelated to the question.\n"
            "Return ONLY the word PASS or FAIL. Nothing else."
        )

    response = llm.invoke([
        SystemMessage(content="You are a strict evaluator. Return only PASS or FAIL."),
        HumanMessage(content=evaluation_input),
    ])

    evaluation = response.content.strip().upper()
    state["evaluation"] = "PASS" if "PASS" in evaluation else "FAIL"
    return state


def retry_codegen_node(state: dict) -> dict:
    """Re-generate code after a failure, including error feedback."""
    question = state["question"]
    csv_path = state["csv_path"]
    previous_code = state["generated_code"]
    error = state.get("execution_error") or "Result was evaluated as incorrect."
    csv_info = _get_csv_info(csv_path)

    system_prompt = (
        "You are a Python data analyst. Your previous code had an issue. "
        "Fix it and produce corrected code.\n\n"
        f"Dataset info:\n{csv_info}\n\n"
        "Rules:\n"
        f"- Read the CSV from: '{csv_path}'\n"
        "- Store the final output in a variable called `result`.\n"
        "- `result` should be a simple, printable value: a string, number, "
        "pandas DataFrame, or Series.\n"
        "- `result` must NEVER be empty or None. Always produce meaningful output.\n"
        "- For analytical questions, compute relevant statistics and store a "
        "DataFrame or descriptive string summarizing the findings in `result`.\n"
        "- Do NOT use print(). Do NOT use plt.show(). Do NOT import matplotlib.\n"
        "- Only output the Python code. No markdown fences, no explanation.\n"
        "- Handle missing values (NaN) gracefully using dropna() or fillna() as needed.\n"
        "- Use ONLY the exact column names listed above. Do NOT guess column names.\n"
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Previous code:\n{previous_code}\n\n"
        f"Error/issue: {error}\n\n"
        "Write corrected Python code."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    code = response.content.strip()
    code = re.sub(r"^```(?:python)?\n?", "", code)
    code = re.sub(r"\n?```$", "", code)

    state["generated_code"] = code
    state["retry_count"] = state.get("retry_count", 0) + 1
    return state


def respond_node(state: dict) -> dict:
    """Generate a natural language final answer from the execution result."""
    question = state["question"]
    execution_result = state["execution_result"]

    response = llm.invoke([
        SystemMessage(
            content=(
                "You are a helpful data analyst. Generate a clear, concise answer "
                "based only on the execution result provided. Do not make up data."
            )
        ),
        HumanMessage(
            content=(
                f"Question: {question}\n\n"
                f"Execution result:\n{execution_result}\n\n"
                "Provide a clear final answer."
            )
        ),
    ])

    state["final_answer"] = response.content.strip()
    return state
