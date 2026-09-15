# [Grayscale Image Contrast Calculator](https://www.deep-ml.com/problems/82) (Easy, Computer Vision)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Understanding Image Contrast](#understanding-image-contrast)
  - [Basic Contrast](#basic-contrast)
  - [RMS Contrast](#rms-contrast)
  - [Michelson Contrast](#michelson-contrast)
  - [Why Contrast Matters](#why-contrast-matters)
- [Solution](#solution)
  - [Approach](#approach)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Complexity Analysis](#complexity-analysis)
- [Key Takeaways](#key-takeaways)

## Problem Statement

Implement a function that calculates the contrast of a grayscale image using the difference between its maximum and minimum pixel values.

For an image $I$, the basic contrast measure is:

$$Contrast=\max(I)-\min(I)$$

The input is a NumPy array representing a grayscale image.

## Example

### Input

```python
img = np.array([[0, 50], [200, 255]])
```

### Output

```text
255
```

### Calculation

The maximum pixel value is:

\(\max(I)=255\)

The minimum pixel value is:

\(\min(I)=0\)

Therefore:

\(Contrast=255-0=255\)

## Understanding Image Contrast

Contrast describes the difference in intensity between the darkest and brightest parts of an image.

For a grayscale image, each pixel represents an intensity value. With the common 8-bit representation:

\(0\leq I(x,y)\leq255\)

where:

- `0` represents black.
- `255` represents white.
- Values between them represent shades of gray.

A simple way to measure the overall intensity range is to subtract the minimum pixel value from the maximum pixel value.

### Basic Contrast

The basic contrast measure used in this problem is:

\(Contrast=\max(I)-\min(I)\)

Consider:

```text
[50, 80, 120, 200]
```

The maximum is `200` and the minimum is `50`.

Therefore:

\(Contrast=200-50=150\)

This measure tells us the total range of intensities present in the image.

A completely uniform image has:

\(\max(I)=\min(I)\)

so:

\(Contrast=0\)

An 8-bit image containing both black and white pixels has the maximum possible range:

\(Contrast=255-0=255\)

### RMS Contrast

RMS contrast measures contrast using the standard deviation of pixel intensities.

\(RMS\ Contrast=\sigma\)

where $\sigma$ is the standard deviation of the pixel intensities.

For an image with mean intensity $\mu$:

\(\sigma=\sqrt{\frac{1}{N}\sum\_{i=1}^{N}(I_i-\mu)^2}\)

Unlike the basic max-min measure, RMS contrast considers how all pixels are distributed around the mean.

For example, two images can have the same minimum and maximum values but different RMS contrast because their intermediate pixels are distributed differently.

### Michelson Contrast

Michelson contrast is commonly used when analyzing periodic patterns such as gratings:

\(C=\frac{I*{max}-I*{min}}{I*{max}+I*{min}}\)

For example, if:

\(I*{max}=200,\quad I*{min}=50\)

then:

\(C=\frac{200-50}{200+50}=\frac{150}{250}=0.6\)

The Michelson measure normalizes the intensity difference by the total intensity.

### Why Contrast Matters

Contrast is important in computer vision because it affects how easily structures and objects can be distinguished.

Applications include:

- Image quality assessment
- Image enhancement
- Preprocessing
- Object detection
- Medical image analysis
- Feature extraction

Higher contrast generally means a larger separation between dark and bright regions, although high contrast does not necessarily mean an image is visually better.

## Solution

### Approach

The problem requires the simplest possible contrast definition:

\(Contrast=\max(I)-\min(I)\)

The algorithm is:

1. Find the maximum pixel value.
2. Find the minimum pixel value.
3. Subtract the minimum from the maximum.
4. Return the result.

NumPy provides `np.max()` and `np.min()` for these operations.

### NumPy Implementation

```python
import numpy as np

def calculate_contrast(img) -> int:
    return np.max(img) - np.min(img)
```

## Code Explanation

### Find the Maximum

```python
np.max(img)
```

This scans the image and returns its largest pixel value.

For:

```python
[[0, 50],
 [200, 255]]
```

the result is:

```text
255
```

### Find the Minimum

```python
np.min(img)
```

This returns the smallest pixel value:

```text
0
```

### Calculate Contrast

```python
np.max(img) - np.min(img)
```

implements:

\(Contrast=\max(I)-\min(I)\)

Therefore:

\(255-0=255\)

The function works directly on the entire NumPy array, so there is no need to manually iterate through individual pixels.

## Complexity Analysis

Let the image contain $N$ pixels.

### Time Complexity

Both `np.max()` and `np.min()` require examining the image:

\(O(N)\)

Therefore the total complexity remains:

\(\boxed{O(N)}\)

### Space Complexity

The calculation requires only a constant amount of additional space apart from the input:

\(\boxed{O(1)}\)

## Key Takeaways

- Grayscale images represent intensity using numerical pixel values.
- The basic contrast measure in this problem is the intensity range:

\(Contrast=\max(I)-\min(I)\)

- A uniform image has contrast `0`.
- An 8-bit image can have a maximum contrast of `255`.
- RMS contrast uses standard deviation and considers the distribution of all pixel values.
- Michelson contrast normalizes the difference between maximum and minimum intensities.
- The NumPy implementation is concise because `np.max()` and `np.min()` operate directly over the entire image.
