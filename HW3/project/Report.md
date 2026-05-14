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
| B1 | <!-- question --> | <!-- reason --> | <!-- ops --> | Basic |
| B2 | | | | Basic |
| B3 | | | | Basic |
| B4 | | | | Basic |
| B5 | | | | Basic |

### Intermediate Questions (5)

| # | Question | Why multi-step | Expected operations | Classification rationale |
|---|----------|----------------|---------------------|--------------------------|
| I1 | <!-- question --> | <!-- reason --> | <!-- ops --> | Intermediate |
| I2 | | | | Intermediate |
| I3 | | | | Intermediate |
| I4 | | | | Intermediate |
| I5 | | | | Intermediate |

### Advanced Questions (5)

| # | Question | Why multi-step | Expected operations | Classification rationale |
|---|----------|----------------|---------------------|--------------------------|
| A1 | <!-- question --> | <!-- reason --> | <!-- ops --> | Advanced |
| A2 | | | | Advanced |
| A3 | | | | Advanced |
| A4 | | | | Advanced |
| A5 | | | | Advanced |

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
