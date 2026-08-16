# Cross Product of Two Vectors

---

## 1. Cross Product in 2D

Let

$$
A = [a_1, a_2]
$$

$$
B = [b_1, b_2]
$$

The **2D cross product** is defined as:

$$
A \times B = a_1b_2 - a_2b_1
$$

This is a **scalar** (single number), not a vector.

### Interpretation

- $A \times B > 0$ → $B$ is **Anti-Clockwise (Left)** of $A$
- $A \times B < 0$ → $B$ is **Clockwise (Right)** of $A$
- $A \times B = 0$ → $A$ and $B$ are **Collinear**

### Area of the Parallelogram

The area of the parallelogram formed by $A$ and $B$ is:

$$
\text{Area} = |A \times B|
$$

$$
\text{Area} = |a_1b_2 - a_2b_1|
$$

This value is also equal to the determinant:

$$
\begin{vmatrix}
a_1 & b_1 \
a_2 & b_2
\end{vmatrix}
$$

Therefore,

$$
A \times B = \pm \text{Area}
$$

where the **sign indicates orientation**.

Also,

$$
A \times B = |A||B|\sin(\theta)
$$

and

$$
\text{Area} = |A||B|\sin(\theta)
$$

---

## 2. Cross Product in 3D

Let

$$
A = [a_1, a_2, a_3]
$$

$$
B = [b_1, b_2, b_3]
$$

The **cross product is a vector**.

It can be computed using:

$$
A \times B =
\begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \
a_1 & a_2 & a_3 \
b_1 & b_2 & b_3
\end{vmatrix}
$$

Expanding along the first row:

$$
A \times B =
\mathbf{i}(a_2b_3-a_3b_2)
-\mathbf{j}(a_1b_3-a_3b_1)
+\mathbf{k}(a_1b_2-a_2b_1)
$$

or

$$
A \times B =
(a_2b_3-a_3b_2,;
a_3b_1-a_1b_3,;
a_1b_2-a_2b_1)
$$

---

## 3. Meaning of $A \times B$ in 3D

The vector $A \times B$ contains:

1. **Magnitude**
2. **Direction**

### Direction

The direction is:

- Perpendicular to both $A$ and $B$
- Determined by the **Right-Hand Rule**

### Right-Hand Rule

- **Index Finger** → $A$
- **Middle Finger** → $B$
- **Thumb** → $A \times B$

Thus,

$$
A \times B =
(\text{Area of Parallelogram})
\times
(\text{Unit Normal Vector})
$$

---

## 4. Magnitude of $A \times B$

The magnitude is:

$$
|A \times B| =
\sqrt{
(a_2b_3-a_3b_2)^2
+
(a_3b_1-a_1b_3)^2
+
(a_1b_2-a_2b_1)^2
}
$$

This magnitude equals the **area of the parallelogram** formed by $A$ and $B$.

Therefore,

$$
|A \times B| =
\text{Area of Parallelogram} =
|A||B|\sin(\theta)
$$

---

## 5. Important Note About Determinants

### In 2D

$$
A \times B = a_1b_2-a_2b_1
$$

which is exactly the determinant:

$$
\begin{vmatrix}
a_1 & b_1 \
a_2 & b_2
\end{vmatrix}
$$

Hence,

$$
\text{Area} = |\text{Determinant}|
$$

### In 3D

$A$ and $B$ together form a $3 \times 2$ matrix:

$$
\begin{bmatrix}
a_1 & b_1 \
a_2 & b_2 \
a_3 & b_3
\end{bmatrix}
$$

A determinant is defined only for **square matrices**.

Since a $3 \times 2$ matrix is **not square**,

> **"Determinant of $A$ and $B$" is not defined.**

Therefore,

$$
|A \times B| \neq \text{Determinant}(A,B)
$$

The cross product vector is built using several $2 \times 2$ determinants, but its **magnitude itself is not a determinant**.

---

## 6. Final Summary

### 2D

$$
A \times B = a_1b_2-a_2b_1
$$

$$
|A \times B| = \text{Area of Parallelogram}
$$

Sign of $A \times B$ gives orientation:

- $+$ → **Anti-Clockwise**
- $-$ → **Clockwise**
- $0$ → **Collinear**

---

### 3D

$$
A \times B =
\text{Vector perpendicular to } A \text{ and } B
$$

**Direction:**

- Right-Hand Rule

**Magnitude:**

$$
|A \times B| =
\text{Area of Parallelogram} =
|A||B|\sin(\theta)
$$

The cross product vector stores:

- **Area** → magnitude
- **Normal direction** → orientation in space
