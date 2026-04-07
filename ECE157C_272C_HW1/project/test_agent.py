import os
import csv
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from agent import agent


def run_agent(question: str, csv_path: str) -> dict:
    """Run the CSV Q&A agent on a single question.

    Returns:
        {
            "generated_code": str,
            "execution_result": object,
            "evaluation": "PASS" or "FAIL",
            "final_answer": str,
        }
    """
    initial_state = {
        "question": question,
        "csv_path": csv_path,
        "generated_code": "",
        "execution_result": None,
        "execution_error": None,
        "evaluation": "",
        "final_answer": "",
        "retry_count": 0,
    }

    result_state = agent.invoke(initial_state)

    return {
        "generated_code": result_state.get("generated_code", ""),
        "execution_result": result_state.get("execution_result"),
        "evaluation": result_state.get("evaluation", "FAIL"),
        "final_answer": result_state.get("final_answer", ""),
    }


# ---------------------------------------------------------------------------
# Fixed dataset questions (housing.csv)
# ---------------------------------------------------------------------------
HOUSING_QUESTIONS = [
    # Simple
    "What is the average median house value across the dataset?",
    "Which ocean proximity category has the highest average median house value?",
    "What are the minimum, maximum, and median values of median house value?",
    # Intermediate
    "How does median income vary across different ocean proximity categories?",
    "Which geographic areas (based on latitude and longitude ranges) have the highest average house prices?",
    "How does population density (defined as population per household) relate to median house value?",
    # Advanced
    "Identify the top 5 most expensive geographic areas and explain the key factors contributing to their high prices.",
    "Find coastal areas where house prices are relatively low despite proximity to the ocean. What factors might explain this?",
    "Identify areas with similar median income levels but significantly different median house value. What factors might explain these differences?",
]


def run_all_questions(questions, csv_path, dataset_name):
    """Run all questions and return a list of result rows."""
    rows = []
    for i, q in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"[{dataset_name}] Question {i}/{len(questions)}: {q}")
        print("=" * 60)
        result = run_agent(q, csv_path)
        print(f"  Evaluation: {result['evaluation']}")
        print(f"  Answer: {result['final_answer'][:200]}...")
        rows.append({
            "dataset_name": dataset_name,
            "question": q,
            "generated_code": result["generated_code"],
            "final_answer": result["final_answer"],
        })
    return rows


def write_results_csv(rows, output_path="results.csv"):
    """Write result rows to a CSV file."""
    fieldnames = ["dataset_name", "question", "generated_code", "final_answer"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Fixed dataset ---
    housing_csv = os.path.join(script_dir, "housing.csv")
    all_rows = run_all_questions(HOUSING_QUESTIONS, housing_csv, "housing")

    # --- Custom dataset (Video Game Sales) ---
    CUSTOM_QUESTIONS = [
        # Basic
        "What is the total global sales across all video games in the dataset?",
        "Which platform has the most games in the dataset?",
        "What are the top 5 best-selling video games of all time by global sales?",
        # Intermediate
        "How do average global sales vary across different genres?",
        "Which publisher has the highest total global sales, and how do their sales break down by region (NA, EU, JP, Other)?",
        "How have total global video game sales changed over the years? Show the trend by year.",
        # Advanced
        "Identify genres that perform disproportionately well in Japan compared to North America. What cultural or market factors might explain these differences?",
        "Find platforms that had a rapid rise and decline in game releases. What does this suggest about the console lifecycle?",
        "Identify publishers whose games have high NA sales but low JP sales, and vice versa. What strategic differences might explain these regional patterns?",
    ]
    custom_csv = os.path.join(script_dir, "custom_dataset.csv")
    all_rows += run_all_questions(CUSTOM_QUESTIONS, custom_csv, "vgsales")

    # Write results
    results_path = os.path.join(script_dir, "results.csv")
    write_results_csv(all_rows, results_path)
