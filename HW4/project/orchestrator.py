import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv

import analytics_agent
import validator_agent
import web_search

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_RETRIES = 3


def _classify(question: str) -> str:
    """Return 'analytics' or 'web'."""
    system = (
        "You are a question router for an AI analytics system that analyzes US stock "
        "financial data (2014–2018).\n\n"
        "Classify the question as one of:\n"
        "  analytics — the question requires analyzing financial data, stock metrics, "
        "sector comparisons, temporal trends, or dataset-specific insights\n"
        "  web — the question requires general world knowledge, current events, "
        "definitions, or information not in the financial dataset\n\n"
        "Return ONLY the single word: analytics or web"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    )
    label = response.choices[0].message.content.strip().lower()
    return "analytics" if "analytics" in label else "web"


def run(
    question: str,
    dataset_paths: list[str],
    on_event=None,
) -> dict:
    """
    Orchestrate the full pipeline for a user question.

    Parameters
    ----------
    question      : user's question string
    dataset_paths : list of CSV file paths available for analysis
    on_event      : optional callback(event_type: str, payload: dict) for streaming

    Returns
    -------
    dict with keys:
      source          : "analytics" | "web"
      final_answer    : str
      plots           : list[dict]
      citations       : list[dict]   (web only)
      validator       : dict | None  (analytics only)
      execution_trace : list         (analytics only)
      iterations      : int
      retries         : int
    """
    def emit(event_type: str, payload: dict):
        if on_event:
            on_event(event_type, payload)

    # --- classify ---
    source = _classify(question)
    emit("classify", {"question": question, "source": source})

    # --- web search branch ---
    if source == "web":
        emit("web_search_start", {"question": question})
        result = web_search.search(question)
        emit("web_search_done", {"answer": result["final_answer"]})
        return {
            "source": "web",
            "final_answer": result["final_answer"],
            "plots": [],
            "citations": result["citations"],
            "validator": None,
            "execution_trace": [],
            "iterations": 0,
            "retries": 0,
        }

    # --- analytics branch with validator retry loop ---
    validator_feedback: str | None = None
    retries = 0
    analytics_output = None
    validator_result = None

    for attempt in range(MAX_RETRIES + 1):
        emit("analytics_start", {"attempt": attempt + 1, "feedback": validator_feedback})

        analytics_output = analytics_agent.run(
            question=question,
            dataset_paths=dataset_paths,
            validator_feedback=validator_feedback,
        )

        emit("analytics_done", {
            "attempt": attempt + 1,
            "iterations": analytics_output["iterations"],
            "answer": analytics_output["final_answer"],
        })

        # --- validate ---
        emit("validator_start", {"attempt": attempt + 1})
        validator_result = validator_agent.validate(
            question=question,
            analytics_output=analytics_output,
            dataset_paths=dataset_paths,
        )
        emit("validator_done", {
            "verdict": validator_result["verdict"],
            "feedback": validator_result["feedback"],
        })

        if validator_result["verdict"] == "PASS":
            break

        if attempt < MAX_RETRIES:
            retries += 1
            validator_feedback = validator_result["feedback"]
            if validator_result["issues"]:
                validator_feedback += " Issues: " + "; ".join(validator_result["issues"])
        else:
            # exhausted retries — use last result regardless
            break

    return {
        "source": "analytics",
        "final_answer": analytics_output["final_answer"],
        "plots": analytics_output["plots"],
        "citations": [],
        "validator": validator_result,
        "execution_trace": analytics_output["execution_trace"],
        "iterations": analytics_output["iterations"],
        "retries": retries,
    }


if __name__ == "__main__":
    import sys

    datasets = [
        "datasets/2014_Financial_Data.csv",
        "datasets/2015_Financial_Data.csv",
        "datasets/2016_Financial_Data.csv",
    ]

    question = sys.argv[1] if len(sys.argv) > 1 else (
        "What caused the 2016 oil-price crash?"
    )

    def print_event(event_type, payload):
        print(f"  [{event_type}] {payload}")

    print(f"\nQuestion: {question}\n")
    result = run(question, datasets, on_event=print_event)

    print(f"\nSource: {result['source']}")
    print(f"Answer:\n{result['final_answer']}")
    if result["validator"]:
        print(f"\nValidator: {result['validator']['verdict']} — {result['validator']['feedback']}")
    if result["citations"]:
        print("\nCitations:")
        for c in result["citations"]:
            print(f"  - {c['title']}: {c['url']}")
    print(f"\nRetries: {result['retries']}  Iterations: {result['iterations']}")
