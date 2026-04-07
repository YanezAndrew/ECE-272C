"""Re-run the previously failed custom questions to test the fix."""
import os
from dotenv import load_dotenv
load_dotenv()
from test_agent import run_agent

failed_qs = [
    "Identify genres that perform disproportionately well in Japan compared to North America. What cultural or market factors might explain these differences?",
    "Find platforms that had a rapid rise and decline in game releases. What does this suggest about the console lifecycle?",
]

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_dataset.csv")

for q in failed_qs:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print("=" * 60)
    r = run_agent(q, csv_path)
    print(f"  Eval: {r['evaluation']}")
    print(f"  Code:\n{r['generated_code']}")
    print(f"  Result: {r['execution_result']}")
    print(f"  Answer: {r['final_answer'][:300]}")
