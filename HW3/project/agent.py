import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from planner import generate_plan
from executor import execute

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(question: str, result: pd.DataFrame) -> str:
    """Produce a natural language answer grounded in the computed result table."""
    result_text = result.to_string(index=False)
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a data analyst. Answer the question based strictly on the "
                    "provided computed result table. Be concise and specific — cite the "
                    "actual values from the table. Do not speculate beyond what the data shows."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nComputed result:\n{result_text}",
            },
        ],
    )
    return response.choices[0].message.content.strip()


def run(question: str, dataset_path: str) -> dict:
    """
    Full pipeline: question → plan → execute → answer.

    Returns
    -------
    dict with keys:
      question    : str
      plan        : dict
      trace       : list[dict]
      result      : pd.DataFrame
      answer      : str
    """
    df = pd.read_csv(dataset_path)

    plan = generate_plan(question, df)
    result, trace = execute(plan, df)
    answer = generate_answer(question, result)

    return {
        "question": question,
        "plan": plan,
        "trace": trace,
        "result": result,
        "answer": answer,
    }


def print_run(output: dict) -> None:
    """Pretty-print a run result to stdout."""
    print(f"Question: {output['question']}")
    print()
    print("Plan:")
    print(json.dumps(output["plan"], indent=2))
    print()
    print("Execution trace:")
    print(f"  {'Step':<6} {'Operation':<25} {'Input Rows':<14} {'Output Rows'}")
    print(f"  {'-'*6} {'-'*25} {'-'*14} {'-'*11}")
    for t in output["trace"]:
        print(f"  {t['step']:<6} {t['operation']:<25} {t['input_rows']:<14} {t['output_rows']}")
    print()
    print("Result:")
    print(output["result"].to_string(index=False))
    print()
    print(f"Answer: {output['answer']}")
    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agent.py \"<question>\" [dataset_path]")
        sys.exit(1)

    question = sys.argv[1]
    dataset = sys.argv[2] if len(sys.argv) > 2 else "dataset.csv"
    output = run(question, dataset)
    print_run(output)
