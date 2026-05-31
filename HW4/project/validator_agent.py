import os
import json
import re
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _make_validation_prompt(
    question: str,
    analytics_output: dict,
    dataset_paths: list[str],
) -> list[dict]:
    # Build a concise dataset schema hint for the validator
    schema_hints = []
    for path in dataset_paths:
        try:
            df = pd.read_csv(path, index_col=0, nrows=0)
            schema_hints.append(
                f"{os.path.basename(path)}: {len(df.columns)} columns, "
                f"index=ticker, key columns include: Sector, returnOnEquity, "
                f"Net Income, Revenue, PE ratio, Debt to Equity, Market Cap, "
                f"Gross Margin, EPS, Class, {[c for c in df.columns if 'PRICE VAR' in c]}"
            )
        except Exception:
            pass

    schema_text = "\n".join(schema_hints)

    final_answer = analytics_output.get("final_answer", "")
    iterations = analytics_output.get("iterations", 0)

    # Summarize execution trace
    trace_parts = []
    for t in analytics_output.get("execution_trace", []):
        part = f"Iteration {t['iteration']}:"
        if t.get("error"):
            part += f" ERROR — {t['error']}"
        else:
            part += f"\n  Result: {t.get('result_description', '')[:400]}"
            if t.get("stop_decision"):
                part += f"\n  Stop decision: {t['stop_decision']}"
        trace_parts.append(part)
    trace_text = "\n".join(trace_parts)

    plots_count = len(analytics_output.get("plots", []))
    plot_titles = [p.get("title", "") for p in analytics_output.get("plots", [])]

    system = (
        "You are an independent analytical validator for a data analytics system. "
        "Your role is to reason carefully about whether an analysis is correct, complete, "
        "and consistent — NOT just whether code executed without errors.\n\n"
        "You must evaluate:\n"
        "1. CORRECTNESS: Are the numerical results plausible? Do they make financial sense? "
        "   Are there signs of column misuse, wrong aggregation, or data type errors?\n"
        "2. COMPLETENESS: Does the final answer fully address the original question? "
        "   Are all parts of a multi-part question answered?\n"
        "3. CONSISTENCY: Do intermediate results align with the final answer? "
        "   For cross-year analysis, are tickers aligned correctly across tables?\n"
        "4. SUFFICIENCY: Is the result non-trivial? A scalar answer to a complex question "
        "   may be insufficient. Were visualizations generated when they would add value?\n\n"
        "Dataset context:\n"
        f"{schema_text}\n\n"
        "Return a JSON object with exactly these keys:\n"
        '  "verdict": "PASS" | "RETRY" | "SUSPICIOUS"\n'
        '  "feedback": "specific, actionable explanation (1-3 sentences)"\n'
        '  "issues": ["list", "of", "specific", "issues"] (empty list if PASS)\n\n'
        "PASS = analysis is correct, complete, and answers the question well.\n"
        "RETRY = analysis has fixable issues; provide specific feedback so the agent can correct it.\n"
        "SUSPICIOUS = results look numerically wrong or internally inconsistent.\n\n"
        "Return ONLY valid JSON. No markdown, no extra text."
    )

    user = (
        f"Original question: {question}\n\n"
        f"Final answer produced:\n{final_answer}\n\n"
        f"Iterations used: {iterations}\n"
        f"Plots generated: {plots_count} — titles: {plot_titles}\n\n"
        f"Execution trace:\n{trace_text}"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate(
    question: str,
    analytics_output: dict,
    dataset_paths: list[str],
) -> dict:
    """
    Independently validate the analytics output.

    Parameters
    ----------
    question        : the original user question
    analytics_output: dict returned by analytics_agent.run()
    dataset_paths   : list of CSV file paths used in the analysis

    Returns
    -------
    dict with keys:
      verdict  : "PASS" | "RETRY" | "SUSPICIOUS"
      feedback : str
      issues   : list[str]
    """
    messages = _make_validation_prompt(question, analytics_output, dataset_paths)
    response = client.chat.completions.create(
        model="gpt-4o", temperature=0, messages=messages
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        result = json.loads(raw)
    except Exception:
        # Fallback: try to extract JSON from response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except Exception:
                result = {
                    "verdict": "RETRY",
                    "feedback": f"Validator could not parse its own response: {raw[:200]}",
                    "issues": ["validator parse error"],
                }
        else:
            result = {
                "verdict": "RETRY",
                "feedback": f"Validator returned unparseable output: {raw[:200]}",
                "issues": ["validator parse error"],
            }

    # Normalize verdict
    verdict = result.get("verdict", "RETRY").upper()
    if verdict not in {"PASS", "RETRY", "SUSPICIOUS"}:
        verdict = "RETRY"
    result["verdict"] = verdict

    return result


if __name__ == "__main__":
    # Quick test
    import analytics_agent

    datasets = [
        "datasets/2014_Financial_Data.csv",
        "datasets/2015_Financial_Data.csv",
    ]
    question = "Which sector had the highest average gross margin in 2014?"
    print("Running analytics agent...")
    output = analytics_agent.run(question, datasets)
    print(f"Answer: {output['final_answer']}")
    print(f"Iterations: {output['iterations']}")

    print("\nRunning validator...")
    verdict = validate(question, output, datasets)
    print(f"Verdict: {verdict['verdict']}")
    print(f"Feedback: {verdict['feedback']}")
    print(f"Issues: {verdict['issues']}")
