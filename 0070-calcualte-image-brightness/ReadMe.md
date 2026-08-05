# Calculate Image Brightness (Easy, Computer Vision)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Image Brightness](#learn-image-brightness)
- [Solution](#solution)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Calculate Image Brightness](https://www.deep-ml.com/problems/70)

Write a Python function that computes the **average brightness** of a grayscale image.

The function should:

- Accept a 2-D grayscale image.
- Compute the average pixel intensity.
- Return the average brightness rounded to two decimal places.
- Return **-1** if:
  - The image is empty.
  - The image has inconsistent row lengths.
  - Any pixel value lies outside the valid range `[0, 255]`.

---

## Example

### Input

```python
img = [
    [100, 200],
    [50, 150]
]

print(calculate_brightness(img))
```

### Output

```python
125.0
```

### Reasoning

The image contains four pixels.

```text
100 200
50  150
```

The average brightness is

$$
\frac{100+200+50+150}{4}=125
$$

Therefore, the function returns

```text
125.0
```

---

## Learn: Image Brightness

### What is it?

**Image brightness** is the average intensity of all pixels in an image.

For a grayscale image, each pixel has a single intensity value ranging from

- **0** → Completely black
- **255** → Completely white

Larger intensity values correspond to brighter pixels.

The average brightness provides a simple numerical measure of how bright or dark an image appears overall.

---

### Grayscale Image Representation

A grayscale image is represented as a matrix.

For example,

```text
100 200
50  150
```

Each entry represents the intensity of one pixel.

Unlike RGB images, a grayscale image contains only one intensity channel.

---

### Mathematical Definition

Suppose an image has

- $m$ rows.
- $n$ columns.

The average brightness is

$$
\text{Brightness}=\frac{\sum_{i=1}^{m}\sum_{j=1}^{n}p_{ij}}{m\times n}
$$

where

- $p_{ij}$ is the pixel intensity at row $i$ and column $j$.

---

### Pixel Intensity Range

Every pixel must satisfy

$$
0\le p_{ij}\le255
$$

Values outside this range are invalid for an 8-bit grayscale image.

---

### Characteristics / Key Points

- Brightness is the arithmetic mean of all pixel values.
- Computed only from pixel intensities.
- Independent of image content.
- Easy to calculate.
- Frequently used as a preprocessing statistic in computer vision.

---

### Interpretation

| Average Brightness | Image Appearance  |
| -----------------: | ----------------- |
|         Near **0** | Very dark         |
|     Around **128** | Medium brightness |
|       Near **255** | Very bright       |

---

### Edge Cases

A valid grayscale image must satisfy all of the following:

- The image is not empty.
- Every row has the same number of columns.
- Every pixel lies within the valid range.

$$
0\le\text{Pixel}\le255
$$

If any condition fails, the function returns

```text
-1
```

---

### Why is it used? / Applications

Average brightness is commonly used in computer vision and image processing tasks such as

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
> Average brightness measures only the overall intensity of an image. Two images can have the same average brightness but completely different visual appearances. Metrics such as contrast, histograms, or the standard deviation of pixel intensities are often used alongside brightness to better describe an image.

---

## Solution

### NumPy Implementation

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

### Step 1

Convert the input image into a NumPy array.

```python
img = np.asarray(img, dtype=float)
```

This enables efficient numerical operations. If the conversion fails due to inconsistent row lengths, the function returns `-1`.

---

### Step 2

Check whether the image is empty.

```python
if img.size == 0:
    return -1
```

An empty image contains no pixels, so its brightness cannot be computed.

---

### Step 3

Validate the pixel values.

```python
if np.any((img < 0) | (img > 255)):
    return -1
```

This ensures every pixel satisfies

$$
0\le p_{ij}\le255
$$

If any pixel falls outside this range, the image is considered invalid.

---

### Step 4

Compute the average brightness.

```python
return np.mean(img)
```

`np.mean()` computes

$$
\frac{\sum p_{ij}}{m\times n}
$$

which is the average intensity of all pixels.

---

## Time & Space Complexity

Let

- $m$ = Number of rows.
- $n$ = Number of columns.

| Complexity | Value        |
| ---------- | ------------ |
| Time       | **O(m × n)** |
| Space      | **O(1)**     |

The algorithm scans every pixel to validate the intensity range and compute the mean.
