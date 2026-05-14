import json
import csv
import traceback
from agent import run, print_run

DATASET = "dataset.csv"

QUESTIONS = [
    # --- Basic ---
    ("B1", "What are the top 5 vehicle makes by number of citations received?"),
    ("B2", "What is the average fine amount for citations issued to out-of-state (non-CA) vehicles?"),
    ("B3", "What are the top 3 most expensive violation types by average fine amount?"),
    ("B4", "How many citations were issued per body style, ordered from most to least?"),
    ("B5", "What are the distinct violation descriptions for citations with a fine of exactly $73?"),
    # --- Intermediate ---
    ("I1", "Among Toyota (TOYT) vehicles, what are the top 3 violation types by citation count?"),
    ("I2", "Which violation types generated total fine revenue above $50,000?"),
    ("I3", "For citations issued during morning hours (issue_time less than 1200), what is the most common violation?"),
    ("I4", "What are the top 5 locations with the most RED ZONE violations?"),
    ("I5", "Among passenger vehicles (body_style PA), which vehicle makes have more than 20 citations with fines above $100?"),
    # --- Advanced ---
    ("A1", "Derive the issue hour from issue_time by floor dividing by 100, then find the top 3 hours by total fine revenue."),
    ("A2", "Among vehicle makes with more than 100 citations, which 5 have the highest average fine amount?"),
    ("A3", "For each violation type, compute the ratio of total fine revenue to citation count. Which violations have a ratio above $80, ranked by ratio descending?"),
    ("A4", "Among California-registered vehicles cited for NO PARK/STREET CLEAN violations, which 3 vehicle makes have the highest total fines among makes with more than 50 such citations?"),
    ("A5", "Derive the issue hour from issue_time by floor dividing by 100, keep only afternoon citations where hour is between 12 and 18, then find the top 3 vehicle makes by total fine revenue among makes with more than 50 such citations."),
]


def main():
    rows = []

    for qid, question in QUESTIONS:
        print(f"{'='*60}")
        print(f"[{qid}] {question}")
        print(f"{'='*60}")
        try:
            output = run(question, DATASET)
            print_run(output)
            rows.append({
                "id": qid,
                "question": question,
                "plan": json.dumps(output["plan"]),
                "final_answer": output["answer"],
            })
        except Exception as e:
            print(f"ERROR on {qid}: {e}")
            traceback.print_exc()
            rows.append({
                "id": qid,
                "question": question,
                "plan": "ERROR",
                "final_answer": str(e),
            })
        print()

    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "plan", "final_answer"])
        writer.writeheader()
        writer.writerows([{"question": r["question"], "plan": r["plan"], "final_answer": r["final_answer"]} for r in rows])

    print(f"Wrote {len(rows)} results to results.csv")


if __name__ == "__main__":
    main()
