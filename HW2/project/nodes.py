import os
import re
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class ExecutionResult:
    """Structured representation of a single execution step's output."""

    def __init__(self, data, question: str = ""):
        self.data = data
        self.question = question
        self.data_type = self._infer_type(data)

    def _infer_type(self, data) -> str:
        if isinstance(data, pd.DataFrame):
            return "dataframe"
        if isinstance(data, pd.Series):
            return "series"
        if isinstance(data, (int, float)):
            return "scalar"
        return "string"

    @property
    def is_empty(self) -> bool:
        if isinstance(self.data, (pd.DataFrame, pd.Series)):
            return self.data.empty
        return self.data is None

    def describe(self) -> str:
        """Text summary used in LLM prompts."""
        info = f"Type: {self.data_type}"
        if isinstance(self.data, pd.DataFrame):
            info += f"\nShape: {self.data.shape}"
            info += f"\nColumns: {list(self.data.columns)}"
        elif isinstance(self.data, pd.Series):
            info += f"\nLength: {len(self.data)}"
            info += f"\nName: {self.data.name}"
        info += f"\nSample:\n{str(self.data)[:500]}"
        return info

    def to_json(self) -> str:
        """Serialize for the frontend."""
        if isinstance(self.data, pd.DataFrame):
            return self.data.to_json(orient="records")
        if isinstance(self.data, pd.Series):
            return self.data.to_json()
        return str(self.data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_csv_info(csv_path: str) -> str:
    try:
        df = pd.read_csv(csv_path, nrows=5)
        return (
            f"Columns: {list(df.columns)}\n"
            f"Dtypes: {df.dtypes.to_dict()}\n"
            f"Sample (first 3 rows):\n{df.head(3).to_string()}"
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def codegen_node(state: dict) -> dict:
    question = state["question"]
    csv_path = state["csv_path"]
    csv_info = _get_csv_info(csv_path)
    previous_result: ExecutionResult | None = state.get("previous_result")

    previous_section = ""
    if previous_result is not None:
        previous_section = (
            f"\nThe previous execution result is available as `previous_result`:\n"
            f"{previous_result.describe()}\n\n"
            "If the question refers to the previous result (e.g. 'show top 5', "
            "'filter this', 'sort by X'), operate on `previous_result` directly "
            "instead of re-reading the CSV.\n"
        )

    system_prompt = (
        "You are a Python data analyst. Write Python code using pandas to answer "
        "the user's question about a CSV dataset.\n\n"
        f"Dataset info:\n{csv_info}\n"
        f"{previous_section}\n"
        "Rules:\n"
        f"- Read the CSV from: '{csv_path}' only if needed (not for follow-up questions).\n"
        "- Store the final output in a variable called `result`.\n"
        "- `result` must NEVER be empty or None. Always produce meaningful output.\n"
        "- For comparison or ranking questions (e.g. 'which has the highest/lowest', "
        "'how does X vary across Y'), store the FULL grouped DataFrame in `result`, "
        "not just the single winning value. This allows downstream visualization.\n"
        "- For simple factual lookups (e.g. 'how many rows', 'what is the average'), "
        "a scalar or string is fine.\n"
        "- Do NOT use print(). Do NOT import matplotlib. Do NOT create any plots or charts.\n"
        "- If the user asks to 'plot', 'visualize', or 'chart' something, ignore that part — "
        "just return the relevant data as `result`. A separate component handles visualization.\n"
        "- Only output the Python code. No markdown fences, no explanation.\n"
        "- Handle missing values (NaN) gracefully using dropna() or fillna() as needed.\n"
        "- Use ONLY the exact column names listed above. Do NOT guess column names.\n"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ])

    code = response.content.strip()
    code = re.sub(r"^```(?:python)?\n?", "", code)
    code = re.sub(r"\n?```$", "", code)

    state["generated_code"] = code
    state["retry_count"] = state.get("retry_count", 0)
    return state


def execute_node(state: dict) -> dict:
    code = state["generated_code"]
    previous_result: ExecutionResult | None = state.get("previous_result")

    env = {}
    if previous_result is not None:
        env["previous_result"] = previous_result.data

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
            er = ExecutionResult(result, state["question"])
            if er.is_empty:
                state["execution_result"] = None
                state["execution_error"] = (
                    "The `result` variable is an empty DataFrame/Series. "
                    "Adjust the code logic so `result` contains meaningful data."
                )
            else:
                state["execution_result"] = er
                state["execution_error"] = None
    except Exception as e:
        state["execution_result"] = None
        state["execution_error"] = str(e)
    return state


def evaluate_node(state: dict) -> dict:
    question = state["question"]
    execution_result: ExecutionResult | None = state["execution_result"]
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
        evaluation_input = (
            f"Question: {question}\n\n"
            f"Generated code:\n{generated_code}\n\n"
            f"Execution result:\n{execution_result.describe()}\n\n"
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
    question = state["question"]
    csv_path = state["csv_path"]
    previous_code = state["generated_code"]
    error = state.get("execution_error") or "Result was evaluated as incorrect."
    csv_info = _get_csv_info(csv_path)
    previous_result: ExecutionResult | None = state.get("previous_result")

    previous_section = ""
    if previous_result is not None:
        previous_section = (
            f"\nThe previous execution result is available as `previous_result`:\n"
            f"{previous_result.describe()}\n"
        )

    system_prompt = (
        "You are a Python data analyst. Your previous code had an issue. "
        "Fix it and produce corrected code.\n\n"
        f"Dataset info:\n{csv_info}\n"
        f"{previous_section}\n"
        "Rules:\n"
        f"- Read the CSV from: '{csv_path}' only if needed.\n"
        "- Store the final output in a variable called `result`.\n"
        "- `result` must NEVER be empty or None. Always produce meaningful output.\n"
        "- For comparison or ranking questions, store the FULL grouped DataFrame "
        "in `result`, not just the single winning value.\n"
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
    question = state["question"]
    execution_result: ExecutionResult | None = state["execution_result"]
    execution_error = state.get("execution_error")

    if execution_error or execution_result is None:
        state["final_answer"] = f"I was unable to compute an answer: {execution_error}"
        return state

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
                f"Execution result:\n{execution_result.describe()}\n\n"
                "Provide a clear final answer."
            )
        ),
    ])

    state["final_answer"] = response.content.strip()
    return state


def _build_figure(data, chart_type: str, x_col: str, y_col: str, title: str):
    import plotly.graph_objects as go

    x = data[x_col].tolist() if x_col and x_col in data.columns else None
    y = data[y_col].tolist() if y_col and y_col in data.columns else None

    if chart_type == "bar":
        fig = go.Figure(go.Bar(x=x, y=y))
    elif chart_type == "line":
        fig = go.Figure(go.Scatter(x=x, y=y, mode="lines"))
    elif chart_type == "scatter":
        fig = go.Figure(go.Scatter(x=x, y=y, mode="markers"))
    elif chart_type == "histogram":
        fig = go.Figure(go.Histogram(x=x))
    elif chart_type == "box":
        fig = go.Figure(go.Box(y=y))
    elif chart_type == "pie":
        fig = go.Figure(go.Pie(labels=x, values=y))
    else:
        return None

    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    return fig


def visualization_node(state: dict) -> dict:
    question = state["question"]
    execution_result: ExecutionResult | None = state.get("execution_result")

    if execution_result is None or not isinstance(execution_result.data, pd.DataFrame):
        state["visualization_figure"] = None
        state["visualization_error"] = None
        return state

    columns = list(execution_result.data.columns)

    system_prompt = (
        "You are a data visualization expert. Given a user question and its computed result, "
        "decide whether a chart would add meaningful value.\n\n"
        "Visualization IS useful when the result involves:\n"
        "- Comparisons between categories or groups\n"
        "- Trends over time\n"
        "- Distributions of numerical values\n"
        "- Proportions or part-to-whole relationships\n\n"
        "Visualization is NOT useful for a single row or scalar value.\n\n"
        f"The result is a DataFrame with columns: {columns}\n\n"
        "Respond with ONLY a valid JSON object — no markdown, no extra text:\n"
        '{"visualize": true|false, "chart_type": "bar|line|scatter|histogram|pie|box", '
        '"x_col": "<column name or empty>", "y_col": "<column name or empty>", '
        '"title": "<descriptive chart title>"}'
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Execution result:\n{execution_result.describe()}"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    raw = response.content.strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        decision = json.loads(raw)
    except Exception:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                decision = json.loads(match.group())
            except Exception:
                state["visualization_figure"] = None
                state["visualization_error"] = f"Failed to parse LLM response: {raw[:200]}"
                return state
        else:
            state["visualization_figure"] = None
            state["visualization_error"] = f"Failed to parse LLM response: {raw[:200]}"
            return state

    state["visualization_decision"] = decision.get("visualize", False)
    state["visualization_chart_type"] = decision.get("chart_type", "none")

    if not decision.get("visualize", False):
        state["visualization_figure"] = None
        state["visualization_error"] = None
        return state

    try:
        fig = _build_figure(
            data=execution_result.data,
            chart_type=decision.get("chart_type", "bar"),
            x_col=decision.get("x_col", columns[0]),
            y_col=decision.get("y_col", columns[-1]),
            title=decision.get("title", question),
        )
        if fig is None:
            state["visualization_figure"] = None
            state["visualization_error"] = f"Unsupported chart type: {decision.get('chart_type')}"
        else:
            state["visualization_figure"] = fig.to_json()
            state["visualization_error"] = None
    except Exception as e:
        state["visualization_figure"] = None
        state["visualization_error"] = str(e)

    return state
