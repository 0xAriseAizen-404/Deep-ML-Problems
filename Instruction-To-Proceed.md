- Open Deep-ML.com, Pick a Problem, Try to solve it on your own
- If you can't solve, then Open -> https://github.com/Open-Deep-ML/DML-OpenProblem
  and Get Hint on it and Solve
  Even then you can't solve it, then use ChatGPT.
- For ReadMe.md File,
  The format to give ChatGPT -> https://chatgpt.com/c/6a5b1f68-5ee0-83ee-ba32-bb3d6600e9d6
  1. Give the `Description and Learn about topic` of the Problem
  2. My Solution -> Give your Solution
  3. Tell Problem Number, it will give the Level and Category according to the description tags we will copy.

Prompt ->
see, I am doing problems in Deep-ML.com site okay its like a Leetcode, but it is for DSA and this is for Machine Learning, Deep Learning, Linear Algebra, Statistics all these stuffs okay
I am going to give you Description and Learn about topic of the Problem
I need you to give me the ReadMe markdown based on the template I am gonna share below
I will give Problem Number, place it at the Heading of the problem name on that Link https://www.deep-ml.com/problems/{problem_number}
and then as for the Level and Category, you will get those in the Description tags, based on those, add them in the ReadMe file Heading and I will give you my solution also

Give me in a Complete Markdown code, not the text, i need in Markdown code
and remember this is not obsidian, this is VsCode, so careful with the formula how to represent

Template

# Problem Name (Difficulty, Category)

## Table of Contents

- Problem Statement
- Example
- Learn: <Topic>
- Solution
- Code Explanation

## Problem Statement

Problem Name {Deep-ML link}

Problem description.

## Example

Input / Output / Reasoning

## Learn: <Topic>

### What is it?

A concise explanation (around 60–70% of the original DeepML content).

### Mathematical Definition / Formula

Relevant formulas using VSCode-compatible LaTeX.

### Characteristics / Key Points

- Important properties
- Behavior
- Edge cases
- Complexity (if applicable)

### Why is it used? / Applications

Real-world usage in ML/DL/Linear Algebra/Statistics.

> 💡 **Important Note**
>
> An interview tip, practical insight, optimization, common mistake, or something DeepML didn't mention.

## Solutions

- Custom Implementation
- NumPy / PyTorch (if applicable)

## Code Explanation

Step-by-step explanation of the implementation.

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(n)** |

Where **n** is the number of samples.

---

and the example template is -> 0041-simple-convolutional-2d-layer/ReadMe.md

---

---

---

Response from ChatGPT

Understood. I will follow this exact structure for every Deep-ML problem you provide.

Workflow:

1. You give:
   - **Problem Number**
   - **Problem Description**
   - **Learn About Topic content** from Deep-ML
   - **Your Solution Code**

2. I will generate a complete `README.md` file with:

```markdown
# Problem Name (Difficulty, Category)

## Table of Contents

- Problem Statement
- Example
- Learn: <Topic>
- Solution
- Code Explanation
- Time & Space Complexity
```

### Rules I will follow:

- Problem heading:
  - Extract problem name from Deep-ML.
  - Add difficulty + category from the problem tags.
  - Link format:

    ```
    https://www.deep-ml.com/problems/{problem_number}
    ```

    Example:

    ```
    [Simple Convolutional 2D Layer](https://www.deep-ml.com/problems/41)
    ```

- Problem Statement:
  - Rewrite the description cleanly.
  - Keep it concise.

- Example:
  - Use the provided examples.
  - Add:
    - Input
    - Output
    - Reasoning / Explanation

- Learn Section:
  - Convert Deep-ML explanation into a proper study note.
  - Keep around **60–70% of original content**.
  - Improve structure where required.
  - Add:
    - Concept explanation
    - Mathematical definitions
    - Formulas in VSCode-compatible LaTeX
    - Important characteristics
    - Applications
    - Interview/practical notes

- Solution:
  - Add your implementation.
  - If applicable:
    - Custom implementation
    - NumPy/PyTorch equivalent

- Code Explanation:
  - Explain the algorithm step-by-step.
  - Not line-by-line boring explanation.
  - Focus on logic.

- Complexity:
  - Provide proper Time and Space complexity.
  - Use Markdown tables.
  - Define variables (`n`, `H`, `W`, etc.).

Style:

- Similar to **CLRS / Introduction to Algorithms style notes**:
  - Concept first.
  - Mathematical intuition.
  - Algorithmic thinking.
  - Implementation.
  - Analysis.

- Medium length:
  - Not a huge textbook chapter.
  - Not a tiny answer.
  - Enough to revise later.

- Markdown optimized for:
  - VSCode
  - Obsidian
  - GitHub README

- Maintain ML/DL terminology accuracy.

- Add missing important details if Deep-ML explanation skips them.

- Correct misconceptions if needed.

You can send the first problem in this format:

```
Problem Number:
Difficulty:
Category:

Problem Description:

Learn About Topic:

My Solution:
```

I will return only the finished `README.md` content.
