"""Run only custom dataset questions and append to results.csv."""
import os
import csv
from dotenv import load_dotenv

load_dotenv()

from test_agent import run_agent, run_all_questions

CUSTOM_QUESTIONS = [
    "What is the total global sales across all video games in the dataset?",
    "Which platform has the most games in the dataset?",
    "What are the top 5 best-selling video games of all time by global sales?",
    "How do average global sales vary across different genres?",
    "Which publisher has the highest total global sales, and how do their sales break down by region (NA, EU, JP, Other)?",
    "How have total global video game sales changed over the years? Show the trend by year.",
    "Identify genres that perform disproportionately well in Japan compared to North America. What cultural or market factors might explain these differences?",
    "Find platforms that had a rapid rise and decline in game releases. What does this suggest about the console lifecycle?",
    "Identify publishers whose games have high NA sales but low JP sales, and vice versa. What strategic differences might explain these regional patterns?",
]

script_dir = os.path.dirname(os.path.abspath(__file__))
custom_csv = os.path.join(script_dir, "custom_dataset.csv")
rows = run_all_questions(CUSTOM_QUESTIONS, custom_csv, "vgsales")

results_path = os.path.join(script_dir, "results.csv")
fieldnames = ["dataset_name", "question", "generated_code", "final_answer"]
with open(results_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writerows(rows)
print("\nCustom dataset results appended to results.csv")
