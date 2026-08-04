# Calculate Image Brightness (Easy, Computer Vision)

## Table of Contents

- Problem Statement
- Example
- Learn: Image Brightness
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

[Calculate Image Brightness](https://www.deep-ml.com/problems/70)

Write a Python function `calculate_brightness(img)` that computes the **average brightness** of a grayscale image.

The image is represented as a **2D matrix**, where each element is a pixel intensity between **0** (black) and **255** (white).

The function should:

- Return the average brightness rounded to two decimal places.
- Return **-1** if:
  - The image is empty.
  - The image has inconsistent row lengths.
  - Any pixel value lies outside the valid range `[0, 255]`.

---

## Example

```python
img = [
    [100, 200],
    [50, 150]
]

print(calculate_brightness(img))
```

### Output

```text
125.0
```

### Explanation

The image contains four pixels.

```text
100   200
 50   150
```

The average brightness is

\[
\frac{100+200+50+150}{4}
=
125
\]

Therefore,

```text
Brightness = 125.0
```

---

# Learn: Image Brightness

## What is it?

**Image brightness** is the average intensity of all pixels in an image.

For a **grayscale image**, every pixel has only one intensity value ranging from

- **0** → Completely black
- **255** → Completely white

Larger values correspond to brighter pixels.

The average brightness provides a simple numerical measure of how bright or dark an image appears overall.

---

## Grayscale Image Representation

A grayscale image is represented as a matrix.

For example,

```text
100  200
 50  150
```

Each entry represents the intensity of one pixel.

Unlike RGB images, there is only **one channel**.

---

## Mathematical Definition / Formula

Suppose an image has

- \(m\) rows
- \(n\) columns

The average brightness is

\[
\text{Brightness}
=
\frac{
\sum_{i=1}^{m}
\sum_{j=1}^{n}
p_{ij}
}
{m\times n}
\]

where

- \(p_{ij}\) is the intensity of the pixel at row \(i\), column \(j\).

---

### Pixel Intensity Range

Every pixel must satisfy

\[
0
\le
p_{ij}
\le
255
\]

Values outside this range are invalid for an 8-bit grayscale image.

---

## Characteristics / Key Points

- Brightness is the arithmetic mean of all pixel values.
- Computed only from pixel intensities.
- Independent of image content.
- Easy to calculate.
- Frequently used as a preprocessing statistic in computer vision.

### Interpretation

| Average Brightness | Image Appearance |
| -----------------: | ---------------- |
| Near **0** | Very dark |
| Around **128** | Medium brightness |
| Near **255** | Very bright |

---

## Edge Cases

A valid grayscale image must satisfy all of the following:

- Image is not empty.
- Every row has the same number of columns.
- Every pixel lies in the range

\[
0
\le
\text{Pixel}
\le
255
\]

If any condition fails, the function returns

```text
-1
```

---

## Why is it used? / Applications

Average brightness is commonly used in computer vision and image processing tasks such as:

- Automatic Exposure Adjustment
- Image Enhancement
- Image Normalization
- Contrast Analysis
- Thresholding
- Object Detection
- OCR Preprocessing
- Camera Calibration
- Medical Image Processing

It is often one of the first statistics computed before applying more advanced image processing algorithms.

> 💡 **Important Note**
>
> Average brightness measures only the **overall intensity** of an image. Two images can have the same average brightness but completely different visual appearances. Metrics such as **contrast**, **histograms**, or **standard deviation of pixel intensities** are often used alongside brightness to better describe an image.

---

# Solution

## NumPy Implementation

```python
import numpy as np

def calculate_brightness(img):
    try:
        img = np.asarray(img, dtype=float)
    except ValueError:
        return -1

    if img.size == 0:
        return -1

    if np.any((img < 0) | (img > 255)):
        return -1

    return np.mean(img)
```

---

## Code Explanation

### Step 1: Convert the Image to a NumPy Array

```python
img = np.asarray(img, dtype=float)
```

This converts the input into a NumPy array, enabling efficient numerical operations.

If the conversion fails (for example, due to inconsistent row lengths), the function returns `-1`.

---

### Step 2: Check for an Empty Image

```python
if img.size == 0:
    return -1
```

An empty image contains no pixels, so its brightness cannot be computed.

---

### Step 3: Validate Pixel Values

```python
if np.any((img < 0) | (img > 255)):
    return -1
```

This ensures every pixel satisfies

\[
0
\le
p_{ij}
\le
255
\]

If any pixel falls outside this range, the image is considered invalid.

---

### Step 4: Compute the Average Brightness

```python
return np.mean(img)
```

`np.mean()` computes

\[
\frac{
\sum p_{ij}
}
{m\times n}
\]

which is the average intensity of all pixels.

---

## Time & Space Complexity

Let

- \(m\) = Number of rows.
- \(n\) = Number of columns.

| Complexity | Value |
| ---------- | ----- |
| Time | **O(m × n)** |
| Space | **O(m × n)** |

The algorithm scans every pixel once to validate the intensity range and compute the mean. Converting the input to a NumPy array also requires **O(m × n)** additional space.