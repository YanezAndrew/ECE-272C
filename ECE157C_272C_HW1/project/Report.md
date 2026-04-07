# ECE 157C / 272C — Homework 1 Report

**Name:** Andrew Yanez

---

## 1. Setup and Execution

### Dependencies

```
pip install langgraph langchain langchain-openai pandas python-dotenv
```

### Configuration

1. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-...
   ```
2. The agent uses **GPT-4o** via the OpenAI API (`langchain-openai`).

### How to Run

```bash
# Run all 18 questions (9 housing + 9 custom) and generate results.csv
python3 test_agent.py
```

The script will:
- Run each question through the LangGraph agent
- Print evaluation (PASS/FAIL) and a preview of each answer
- Write all results to `results.csv`

### Project Structure

```
project/
├── agent.py              # LangGraph workflow definition
├── nodes.py              # Node functions (codegen, execute, evaluate, retry, respond)
├── test_agent.py         # run_agent() function + question runner
├── housing.csv           # Fixed dataset (California Housing, 10,000 rows)
├── custom_dataset.csv    # Custom dataset (Video Game Sales, 10,000 rows)
├── results.csv           # Generated results for all 18 questions
└── Report.pdf            # This report
```

---

## 2. Prompt Log (Development Prompts)

Below is a log of prompts used with the AI coding assistant (GitHub Copilot / Claude) during development.

### Prompt 1: Initial Project Scaffolding

**Goal:** Generate the initial project structure with all required files.

**Prompt (paraphrased):**
> "Read through HW1 and come up with a plan to complete the HW. Help me implement, let's use OpenAI because I have a key."

**Outcome:** The assistant created `nodes.py`, `agent.py`, and `test_agent.py` with the full LangGraph pipeline. The initial code structure was functional on the first attempt.

### Prompt 2: Fixing Environment Loading

**Goal:** Fix `OPENAI_API_KEY` not being loaded when running `test_agent.py` directly.

**Prompt (paraphrased):**
> "Run all 9 housing questions."

**Issue:** The `nodes.py` module-level `ChatOpenAI()` instantiation failed because `load_dotenv()` was only called in `agent.py`, which imported from `nodes.py` — so the key wasn't loaded in time.

**Fix:** Added `load_dotenv()` at the top of `nodes.py` before the `ChatOpenAI` instantiation.

**Outcome:** Helpful — resolved the import ordering issue immediately.

### Prompt 3: Custom Dataset Setup

**Goal:** Download and configure the Video Game Sales dataset as the custom dataset.

**Prompt (paraphrased):**
> "Let's do video games sales."

**Outcome:** The assistant used `kagglehub` to download the dataset, sampled it to 10,000 rows, and wrote 9 questions at three difficulty levels. Helpful.

### Prompt 4: Fixing Advanced Question Failures

**Goal:** Debug why 4 out of 9 custom dataset questions failed.

**Prompt (paraphrased):**
> Iterative debugging — inspected generated code, identified that the LLM was guessing wrong column names (e.g., `Japan_Sales` instead of `JP_Sales`).

**Fix:** Added a `_get_csv_info()` helper that reads the first few rows of the CSV and injects column names, dtypes, and sample data into the codegen prompt. Also added instructions: "Use ONLY the exact column names listed above."

**Outcome:** Very helpful — all 4 failing questions started passing after this change.

### Prompt 5: Improving Execute Node Robustness

**Goal:** Detect when generated code fails to assign a value to `result` or produces an empty DataFrame.

**Fix:** Enhanced `execute_node` to check for `None` result and empty DataFrames/Series, and return descriptive error messages to guide the retry.

**Outcome:** The retry node now receives actionable feedback, improving code correction on the second attempt.

### Prompt 6: Improving Evaluator Prompt

**Goal:** Fix overly strict evaluator that marked a correct "top 5 best-selling games" answer as FAIL.

**Fix:** Changed evaluator prompt from "correctly and fully answers" to "produced a non-empty, reasonable result that addresses the question." FAIL is now reserved for empty, None, error, or completely unrelated results.

**Outcome:** Reduced false FAIL evaluations while still catching genuine failures.

---

## 3. Failure Analysis and Debugging

### Failure Case 1: Wrong Column Names (Custom Dataset — Advanced Questions)

**Question:** "Identify genres that perform disproportionately well in Japan compared to North America."

**Did the code work on the first attempt?** No.

**What went wrong:** The LLM generated code referencing `Japan_Sales` and `North_America_Sales`, but the actual column names are `JP_Sales` and `NA_Sales`. The code raised a `KeyError`, the retry also guessed wrong because it had no column information.

**Debugging:** Printed the generated code and saw `print(df.columns)` statements — the LLM was trying to discover columns at runtime instead of using the correct names. The root cause was that the codegen prompt did not include any dataset schema information.

**Prompt improvement:** Added `_get_csv_info()` to read actual column names, dtypes, and sample rows from the CSV, and injected this into both the codegen and retry prompts. Added the rule: "Use ONLY the exact column names listed above. Do NOT guess column names."

**Did the revised prompt improve the result?** Yes — all column-related errors were eliminated. The question passed on the next run.

### Failure Case 2: Empty Result Variable (Custom Dataset — Platform Lifecycle)

**Question:** "Find platforms that had a rapid rise and decline in game releases. What does this suggest about the console lifecycle?"

**Did the code work on the first attempt?** No.

**What went wrong:** The generated code referenced `Year_of_Release` instead of `Year`. The `dropna(subset=['Year_of_Release'])` call raised a `KeyError`. On retry, the code was fixed but used filtering logic that resulted in an empty DataFrame assigned to `result`. The evaluator saw `None` and returned FAIL, but the respond node still generated a "no data found" answer.

**Debugging:** Inspected the execution result and found it was `None`. The execute node was setting `result = env.get("result", "No 'result' variable found.")` — which returned a string even when the actual `result` variable was an empty DataFrame. Enhanced the execute node to explicitly check for empty DataFrames/Series and report them as errors.

**Prompt improvement:** Combined the column-name fix (Failure Case 1) with the empty-result detection. The retry prompt now receives a clear error message: "The `result` variable is an empty DataFrame/Series. Adjust the code logic so `result` contains meaningful data."

**Did the revised prompt improve the result?** Yes — the retry generated code that used correct column names and logic, producing a non-empty result. The question passed.

---

## 4. Self-Grading

| # | Dataset | Difficulty | Question | Eval | Correct? |
|---|---------|-----------|----------|------|----------|
| 1 | housing | Simple | Average median house value | PASS | Yes — $206,943.86 matches manual calculation |
| 2 | housing | Simple | Highest avg by ocean proximity | PASS | Yes — ISLAND is correct |
| 3 | housing | Simple | Min, max, median of house value | PASS | Yes — 14,999 / 500,001 / 178,750 |
| 4 | housing | Intermediate | Median income by ocean proximity | PASS | Yes — values are reasonable |
| 5 | housing | Intermediate | Geographic areas with highest prices | PASS | Yes — SF Bay Area region |
| 6 | housing | Intermediate | Population density vs house value | PASS | Yes — weak negative correlation (-0.044) |
| 7 | housing | Advanced | Top 5 expensive areas + factors | PASS | Yes — correct ranking and reasonable factors |
| 8 | housing | Advanced | Low-price coastal areas | PASS | Yes — identified areas below median with contributing factors |
| 9 | housing | Advanced | Similar income, different value | PASS | Yes — ocean proximity is a key differentiator |
| 10 | vgsales | Basic | Total global sales | PASS | Yes — 5,338.63M |
| 11 | vgsales | Basic | Platform with most games | PASS | Yes — DS |
| 12 | vgsales | Basic | Top 5 best-selling games | PASS | Yes — Wii Sports at #1 |
| 13 | vgsales | Intermediate | Avg sales by genre | PASS | Yes — Platform and Shooter highest |
| 14 | vgsales | Intermediate | Top publisher + regional breakdown | PASS | Yes — Nintendo |
| 15 | vgsales | Intermediate | Sales trend by year | PASS | Yes — peak around 2008-2009 |
| 16 | vgsales | Advanced | JP vs NA genre performance | PASS | Yes — Role-Playing disproportionately high in JP |
| 17 | vgsales | Advanced | Platform rise and decline | PASS | Yes — identified volatile platforms |
| 18 | vgsales | Advanced | Publisher regional patterns | PASS | Yes — Ubisoft/Activision (NA) vs Enix/Namco (JP) |

---

## 5. Advanced Question Reflection

### Housing Dataset — Advanced Questions

#### Q7: Top 5 most expensive geographic areas

**Is the answer the only possible correct answer?**
No. The agent grouped by `ocean_proximity` category, yielding 5 regions. An equally valid approach would be to bin by latitude/longitude to identify specific neighborhoods.

**Can the question be interpreted differently?**
Yes — "geographic areas" is ambiguous. It could mean ocean proximity categories, lat/long grid cells, counties, or cities. Different interpretations yield different top-5 lists.

**Could different answers be valid?**
Yes. Grouping by lat/long bins would identify specific neighborhoods (e.g., parts of San Francisco), while grouping by ocean proximity gives broader regional trends.

**How could the question be rewritten?**
"Identify the top 5 most expensive latitude-longitude grid cells (using 1-degree bins) and explain the key factors contributing to their high prices."

#### Q8: Low-price coastal areas

**Is the answer the only possible correct answer?**
No. The definition of "relatively low" is subjective. The agent used below-median as the threshold, but one could use the 25th percentile or a fixed dollar amount.

**Can the question be interpreted differently?**
Yes — "coastal areas" could mean only `NEAR OCEAN`, or it could include `<1H OCEAN` and `NEAR BAY`. The agent included all three, which is a reasonable but not the only interpretation.

**Could different answers be valid?**
Yes. A stricter definition of "coastal" (only `NEAR OCEAN`) or a different threshold for "relatively low" would produce different results.

**How could the question be rewritten?**
"Among areas classified as NEAR OCEAN, find those with median house values below the 25th percentile. What factors (income, housing age, population) differ from higher-priced NEAR OCEAN areas?"

#### Q9: Similar income, different house value

**Is the answer the only possible correct answer?**
No. The agent grouped by exact median income values, which is very granular. Binning income into ranges (e.g., $2-3, $3-4) would be more meaningful.

**Can the question be interpreted differently?**
Yes — "similar" income levels could mean exact match, within $500, or within the same quartile. "Significantly different" house values is also subjective.

**Could different answers be valid?**
Yes. Different binning strategies and thresholds would surface different area pairs.

**How could the question be rewritten?**
"Group areas into median income bins of width 0.5. Within each bin, find the areas with the highest and lowest median house values. What non-income factors explain the gap?"

### Video Game Sales Dataset — Advanced Questions

#### Q16: Genres disproportionately strong in Japan vs NA

**Is the answer the only possible correct answer?**
No. The agent used a JP/NA sales ratio > 1 as the threshold. Using a different metric (e.g., percentage of total sales within each region) could surface additional genres.

**Can the question be interpreted differently?**
Yes — "disproportionately well" could mean higher absolute sales, higher market share within the region, or a higher ratio relative to overall genre popularity.

**Could different answers be valid?**
Yes. Role-Playing is the clearest answer, but depending on methodology, Puzzle or Strategy might also qualify.

**How could the question be rewritten?**
"For each genre, calculate its share of total JP sales and its share of total NA sales. Which genres have a JP share at least 2x their NA share?"

#### Q17: Platforms with rapid rise and decline

**Is the answer the only possible correct answer?**
No. The agent identified platforms with any peak-trough pattern, which includes nearly every platform. A more selective approach (e.g., requiring the peak to be at least 50 games and the decline to be >50%) would narrow the list.

**Can the question be interpreted differently?**
Yes — "rapid" is subjective. It could mean within 2-3 years, or it could mean a percentage drop threshold. The agent used any year-over-year peak/decline.

**Could different answers be valid?**
Yes. A more focused analysis might highlight only Wii, DS, and PS2 as the clearest examples.

**How could the question be rewritten?**
"Identify platforms where annual game releases more than doubled in under 3 years and then dropped by more than 50% within the next 3 years. What does this suggest about console lifecycle duration?"

#### Q18: Publishers with regional sales disparities

**Is the answer the only possible correct answer?**
No. The thresholds for "high" and "low" sales (1.0 and 0.1 respectively) are arbitrary. Different thresholds would surface different publishers.

**Can the question be interpreted differently?**
Yes — "high" and "low" could be defined in absolute terms, relative to the publisher's total, or relative to the market average. The agent used per-game thresholds rather than aggregate publisher totals.

**Could different answers be valid?**
Yes. Using aggregate publisher-level totals instead of per-game thresholds might highlight different publishers.

**How could the question be rewritten?**
"For publishers with at least 20 games, rank by (total NA sales / total JP sales) ratio. List the top 5 NA-dominant and top 5 JP-dominant publishers. What genre and platform preferences differ between these groups?"
