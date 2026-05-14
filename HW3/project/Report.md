# Homework 3: Plan-and-Execute Data Analysis Agent
**ECE 157C / ECE 272C — University of California, Santa Barbara**
**Due: May 15, 2026**

---

## 1. Dataset Description

**Source:** Los Angeles Open Data Portal — [LA Parking Citations](https://data.lacity.org/Transportation/Parking-Citations/wjz9-h9np) (sampled 20,000 rows via Socrata API, cleaned to 19,966 usable rows)

**Rows / Columns:** 19,966 rows × 16 columns

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| issue_date | string | Date the citation was issued (YYYY-MM-DD) |
| issue_time | float | Time as a 4-digit number (e.g., 1430 = 2:30 PM) |
| rp_state_plate | string | State that issued the vehicle's license plate |
| make | string | Vehicle make abbreviation (e.g., TOYT, HOND, FORD) |
| body_style | string | Vehicle body style code (PA = passenger, PU = pickup, VN = van, etc.) |
| color | string | Vehicle color code |
| location | string | Street address where the citation was issued |
| violation_code | string | Numeric/alpha code identifying the violation |
| violation_description | string | Human-readable description of the violation |
| fine_amount | float | Dollar amount of the fine |
| latitude / longitude | float | GPS coordinates of the citation location |

**Why this dataset requires multi-step analysis:**
Single-step queries cannot answer meaningful questions about this data. For example, finding the vehicle make with the highest average fine *among makes with more than 100 citations* requires aggregation, filtering on the aggregate, and sorting — three distinct operations. Similarly, deriving an "issue hour" from the raw `issue_time` field (e.g., 1430 → 14) requires a column derivation before any grouping can happen. The mix of numeric fines, categorical makes/violations, encoded timestamps, and geographic data naturally produces questions that span multiple operator types.

---

## 2. Question Design

### Basic Questions (5)

| # | Question | Why multi-step | Expected operations | Classification rationale |
|---|----------|----------------|---------------------|--------------------------|
| B1 | What are the top 5 vehicle makes by number of citations received? | Aggregation alone isn't enough — results must be ranked and truncated | group_and_aggregate → sort_rows → limit_rows | 3 sequential ops, no filtering or derivation needed |
| B2 | What is the average fine amount for citations issued to out-of-state (non-CA) vehicles? | Must first narrow the dataset before computing the statistic | filter_rows (state ≠ CA) → group_and_aggregate (mean fine) | Simple filter + aggregate, single condition |
| B3 | What are the top 3 most expensive violation types by average fine amount? | Must aggregate per group then rank | group_and_aggregate (mean fine by violation) → sort_rows (desc) → limit_rows (3) | Aggregate + rank + truncate, no derivation |
| B4 | How many citations were issued per body style, ordered from most to least? | Requires counting per group then ordering | group_and_aggregate (count by body_style) → sort_rows (desc) | Two ops; no filtering needed |
| B5 | What are the distinct violation descriptions for citations with a fine of exactly $73? | Must filter then deduplicate | filter_rows (fine = 73) → select_columns ([violation_description]) → distinct_rows | Filter + project + dedup |

### Intermediate Questions (5)

| # | Question | Why multi-step | Expected operations | Classification rationale |
|---|----------|----------------|---------------------|--------------------------|
| I1 | Among Toyota (TOYT) vehicles, what are the top 3 violation types? | Subset before aggregating; result must be ranked | filter_rows (make = TOYT) → group_and_aggregate (count by violation) → sort_rows (desc) → limit_rows (3) | Adds a pre-filter step before the basic aggregate+rank pattern |
| I2 | Which violation types generated total fine revenue above $50,000? | Aggregate first, then filter on the aggregate result | group_and_aggregate (count + sum fine by violation) → filter_rows (total_fine > 50000) → sort_rows (desc) | Post-aggregation filter — requires thinking in two phases |
| I3 | For citations issued during morning hours (issue_time < 1200), what is the most common violation? | Time-based filter narrows the scope; then requires aggregate + rank | filter_rows (issue_time < 1200) → group_and_aggregate (count by violation) → sort_rows (desc) → limit_rows (1) | Domain-specific numeric condition combined with aggregate + rank |
| I4 | What are the top 5 locations with the most RED ZONE violations? | Must isolate one violation type then aggregate by a different column | filter_rows (violation = RED ZONE) → group_and_aggregate (count by location) → sort_rows (desc) → limit_rows (5) | Cross-column filter → aggregate → rank |
| I5 | Among passenger vehicles (body_style = PA), which vehicle makes have more than 20 citations with fines above $100? | Two independent filter conditions, then aggregate, then post-aggregate filter | filter_rows (body_style = PA) → filter_rows (fine > 100) → group_and_aggregate (count by make) → filter_rows (count > 20) → sort_rows (desc) | Two pre-filters + aggregate + post-aggregate filter = 5 steps |

### Advanced Questions (5)

| # | Question | Why multi-step | Expected operations | Classification rationale |
|---|----------|----------------|---------------------|--------------------------|
| A1 | Derive the issue hour from issue_time (floor divide by 100), then find the top 3 hours by total fine revenue. | Raw data doesn't have an hour column — must derive it before any aggregation | derive_columns (hour = issue_time // 100) → group_and_aggregate (sum fine by hour) → sort_rows (desc) → limit_rows (3) | Requires column derivation before aggregation; 4 steps |
| A2 | Among vehicle makes with more than 100 citations, which 5 have the highest average fine amount? | Must aggregate, post-filter on count, then re-rank on a different metric | group_and_aggregate (count + mean fine by make) → filter_rows (count > 100) → sort_rows (mean fine desc) → limit_rows (5) | Post-aggregation filter on count while ranking on a different column |
| A3 | For each violation type, compute the ratio of total fine revenue to citation count. Which violations have a ratio above $80, and how are they ranked? | Requires aggregation followed by a derived column, then filter on that derived column | group_and_aggregate (count + sum fine by violation) → derive_columns (ratio = total_fine / count) → filter_rows (ratio > 80) → sort_rows (ratio desc) | Post-aggregate derivation + filter on derived column |
| A4 | Among CA-registered vehicles cited for street cleaning (NO PARK/STREET CLEAN), which 3 vehicle makes have the highest total fines among makes with more than 50 such citations? | Two pre-filters on different columns, aggregate, post-aggregate filter, rank | filter_rows (state = CA) → filter_rows (violation = NO PARK/STREET CLEAN) → group_and_aggregate (count + sum fine by make) → filter_rows (count > 50) → sort_rows (sum fine desc) → limit_rows (3) | Longest chain of pre-filters before aggregate + post-aggregate filter; 6 steps |
| A5 | Derive the issue hour, keep only afternoon citations (hour 12–18), then find the top 3 vehicle makes by total fine revenue among makes with more than 50 afternoon citations. | Derivation → time-range filter → aggregate → post-aggregate filter → rank → truncate | derive_columns (hour) → filter_rows (hour ≥ 12) → filter_rows (hour ≤ 18) → group_and_aggregate (count + sum fine by make) → filter_rows (count > 50) → sort_rows (sum fine desc) → limit_rows (3) | Combines derivation, multi-condition time filter, aggregate, post-aggregate filter, and rank; 7 steps |

---

## 3. System Design

### 3.1 Planner
<!-- Describe how the LLM generates the plan:
  - prompt structure
  - how the dataset schema is passed in
  - how the LLM is instructed to resolve vague terms into numeric conditions
  - output format validation -->

### 3.2 Executor
<!-- Describe how the executor applies each step:
  - how it dispatches to operators
  - how it tracks intermediate results
  - how the execution trace is recorded -->

### 3.3 Operators
<!-- Describe each implemented operator:
  - derive_columns
  - filter_rows
  - group_and_aggregate
  - sort_rows
  - limit_rows
  - select_columns
  - distinct_rows
  - (any additional operators) -->

### 3.4 Intermediate Result Passing
<!-- Explain how the output of one step becomes the input of the next, and how the pipeline maintains state. -->

---

## 4. Execution Analysis

### Question 1: <!-- question text -->

**Natural language question:**
<!-- ... -->

**Generated plan:**
```json
// plan JSON will go here
```

**Execution trace:**

| Step | Operation | Input Rows | Output Rows |
|------|-----------|------------|-------------|
| 1 | <!-- op --> | <!-- n --> | <!-- n --> |

**Final answer:**
<!-- ... -->

**Analysis:**
<!-- Why the plan is correct, how each step contributes -->

---

### Question 2: <!-- question text -->

**Natural language question:**
<!-- ... -->

**Generated plan:**
```json
// plan JSON will go here
```

**Execution trace:**

| Step | Operation | Input Rows | Output Rows |
|------|-----------|------------|-------------|
| 1 | <!-- op --> | <!-- n --> | <!-- n --> |

**Final answer:**
<!-- ... -->

**Analysis:**
<!-- Why the plan is correct, how each step contributes -->

---

### Question 3: <!-- question text -->

**Natural language question:**
<!-- ... -->

**Generated plan:**
```json
// plan JSON will go here
```

**Execution trace:**

| Step | Operation | Input Rows | Output Rows |
|------|-----------|------------|-------------|
| 1 | <!-- op --> | <!-- n --> | <!-- n --> |

**Final answer:**
<!-- ... -->

**Analysis:**
<!-- Why the plan is correct, how each step contributes -->

---

## 5. Failure Analysis

### Failure Case 1

**Question:**
<!-- ... -->

**Generated plan:**
```json
// plan JSON will go here
```

**What went wrong:**
<!-- ... -->

**Error type:**
- [ ] Planning error
- [ ] Operator limitation
- [ ] Execution issue

**Fix / improvement:**
<!-- ... -->

---

### Failure Case 2

**Question:**
<!-- ... -->

**Generated plan:**
```json
// plan JSON will go here
```

**What went wrong:**
<!-- ... -->

**Error type:**
- [ ] Planning error
- [ ] Operator limitation
- [ ] Execution issue

**Fix / improvement:**
<!-- ... -->

---

## 6. Comparison with Previous Agents

### Differences Between Agent Designs

| Aspect | HW1 (Code-Generation) | HW2 (Interactive/Memory) | HW3 (Plan-and-Execute) |
|--------|-----------------------|--------------------------|------------------------|
| Reasoning mechanism | <!-- ... --> | <!-- ... --> | <!-- ... --> |
| Execution model | <!-- ... --> | <!-- ... --> | <!-- ... --> |
| Reproducibility | <!-- ... --> | <!-- ... --> | <!-- ... --> |
| Debuggability | <!-- ... --> | <!-- ... --> | <!-- ... --> |

### Advantages of Structured Plans and Operators
<!-- ... -->

### Limitations vs. Code Generation
<!-- ... -->

### Trade-offs

| Trade-off | Discussion |
|-----------|------------|
| Flexibility vs. Reliability | <!-- ... --> |
| Expressiveness vs. Controllability | <!-- ... --> |
| Automation vs. Interpretability | <!-- ... --> |

---

## 7. Reflection

**Easy vs. difficult question types:**
<!-- ... -->

**Limitations of the current operator set:**
<!-- ... -->

**Additional operators that would improve the system:**
<!-- ... -->

**Future extensions:**
<!-- ... -->
