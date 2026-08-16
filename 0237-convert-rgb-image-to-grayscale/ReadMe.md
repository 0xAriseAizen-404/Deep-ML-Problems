# Convert RGB Image to Grayscale (Easy, Computer Vision)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: RGB to Grayscale Conversion](#learn-rgb-to-grayscale-conversion)
  - [What is it?](#what-is-it)
  - [Mathematical Definition](#mathematical-definition)
  - [Why Not Simple Averaging?](#why-not-simple-averaging)
  - [Luminosity Method](#luminosity-method)
  - [Characteristics / Key Points](#characteristics--key-points)
  - [Applications](#applications)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Convert RGB Image to Grayscale](https://www.deep-ml.com/problems/237)

Implement a function `rgb_to_grayscale(image)` that converts an RGB image into a grayscale image using the **luminosity method**.

The input image is represented as a 3D array with shape

$$
H \times W \times 3
$$

where:

- $H$ is the image height.
- $W$ is the image width.
- The third dimension contains the Red, Green, and Blue channels.

Each pixel must contain three values in the range $[0,255]$.

The function should:

- Convert every RGB pixel into a grayscale value.
- Use the standard luminosity coefficients.
- Return a 2D list.
- Round grayscale values to integers.
- Return `-1` for invalid dimensions, empty dimensions, malformed pixels, or values outside the range $[0,255]$.

---

## Example

### Input

```python
image = [[[255, 0, 0], [0, 255, 0]],
         [[0, 0, 255], [255, 255, 255]]]

print(rgb_to_grayscale(image))
```

### Output

```text
[[76, 150], [29, 255]]
```

### Reasoning

The luminosity method calculates the grayscale intensity using

$$Gray = 0.299R + 0.587G + 0.114B$$

For the red pixel $(255,0,0)$:

$$Gray = 0.299(255) + 0.587(0) + 0.114(0) = 76.245$$

After rounding:

```text
76
```

For the green pixel $(0,255,0)$:

$$Gray = 0.299(0) + 0.587(255) + 0.114(0) = 149.685$$

After rounding:

```text
150
```

For the blue pixel $(0,0,255)$:

$$Gray = 0.299(0) + 0.587(0) + 0.114(255) = 29.07$$

After rounding:

```text
29
```

For the white pixel $(255,255,255)$:

$$Gray = 0.299(255) + 0.587(255) + 0.114(255) = 255$$

Therefore, the resulting grayscale image is:

```text
[[76, 150],
 [29, 255]]
```

---

## Learn: RGB to Grayscale Conversion

### What is it?

An RGB image represents each pixel using three color channels:

- **R** — Red
- **G** — Green
- **B** — Blue

A pixel can therefore be represented as

$$Pixel = (R,G,B)$$

where each channel generally contains an intensity value from $0$ to $255$.

For example:

```text
(255, 0, 0)
```

represents pure red.

```text
(0, 255, 0)
```

represents pure green.

```text
(0, 0, 255)
```

represents pure blue.

A grayscale image contains only one intensity value per pixel rather than three color values.

Thus, an RGB image with shape

$$H \times W \times 3$$

becomes a grayscale image with shape

$$H \times W$$

The conversion therefore reduces the number of values required to represent every pixel.

---

### RGB Representation

An RGB image can be viewed as three separate matrices:

$$R \in \mathbb{R}^{H \times W}$$

$$G \in \mathbb{R}^{H \times W}$$

$$B \in \mathbb{R}^{H \times W}$$

Together they form

$$Image \in \mathbb{R}^{H \times W \times 3}$$

For a particular pixel at location $(i,j)$:

$$Pixel_{i,j} = (R_{i,j},G_{i,j},B_{i,j})$$

The grayscale conversion maps this three-dimensional color representation to one scalar:

$$Gray_{i,j} = f(R_{i,j},G_{i,j},B_{i,j})$$

The important question is how the function $f$ should combine the three channels.

---

### Why Not Simple Averaging?

The simplest approach would be to calculate the arithmetic mean of the three channels:

$$Gray_{avg} = \frac{R+G+B}{3}$$

For example, for

```text
R = 100
G = 150
B = 200
```

simple averaging gives

$$Gray_{avg} = \frac{100+150+200}{3} = 150$$

However, this does not accurately represent perceived brightness.

Human vision does not perceive red, green, and blue with equal sensitivity.

Green contributes more strongly to perceived brightness than red, while blue contributes the least.

Therefore, grayscale conversion commonly uses a **weighted combination** rather than an equal average.

---

### Luminosity Method

The luminosity method assigns different weights to the RGB channels according to their contribution to perceived brightness.

The standard formula is

$$Gray = 0.299R + 0.587G + 0.114B$$

The coefficients are:

- Red: $0.299$
- Green: $0.587$
- Blue: $0.114$

Their sum is

$$0.299 + 0.587 + 0.114 = 1.0$$

Therefore, if every channel has the same value $k$:

$$Gray = 0.299k + 0.587k + 0.114k$$

which simplifies to

$$Gray = k$$

This property ensures that neutral grayscale colors remain unchanged.

---

### Why These Weights?

The human visual system has different sensitivities to different wavelengths of visible light.

Green contributes the largest amount to perceived brightness.

Red contributes less than green.

Blue contributes the least.

This gives the approximate contribution:

| Channel |  Weight | Contribution |
| ------- | ------: | ------------ |
| Red     | $0.299$ | $29.9%$      |
| Green   | $0.587$ | $58.7%$      |
| Blue    | $0.114$ | $11.4%$      |

Therefore, the luminosity method preserves perceived brightness better than a simple average.

---

### Example Calculation

Consider the RGB pixel

$$Pixel = (100,150,200)$$

Using the luminosity formula:

$$Gray = 0.299(100) + 0.587(150) + 0.114(200)$$

The individual contributions are

$$0.299(100) = 29.9$$

$$0.587(150) = 88.05$$

$$0.114(200) = 22.8$$

Therefore,

$$Gray = 29.9 + 88.05 + 22.8 = 140.75$$

After rounding:

```text
141
```

Thus,

```text
(100, 150, 200) -> 141
```

---

### Grayscale Image Representation

Suppose the RGB image is

```text
[
    [[255, 0, 0], [0, 255, 0]],
    [[0, 0, 255], [255, 255, 255]]
]
```

Its shape is

$$2 \times 2 \times 3$$

Each pixel is converted independently.

The resulting grayscale image is

```text
[
    [76, 150],
    [29, 255]
]
```

Its shape is

$$2 \times 2$$

The spatial dimensions remain unchanged.

Only the color-channel dimension is removed.

---

### Mathematical Definition

For an RGB image $I$ with height $H$ and width $W$, let

$$I_{i,j} = (R_{i,j},G_{i,j},B_{i,j})$$

Then the grayscale image $G$ is defined as

$$G_{i,j} = 0.299R_{i,j} + 0.587G_{i,j} + 0.114B_{i,j}$$

for

$$0 \leq i < H$$

and

$$0 \leq j < W$$

The final result is rounded to an integer according to the problem requirements.

---

### Properties of the Transformation

The grayscale conversion is a **linear transformation** of the RGB values.

For two RGB pixels $p$ and $q$:

$$f(p+q) = f(p) + f(q)$$

and for a scalar $c$:

$$f(cp) = cf(p)$$

within the valid intensity range.

The transformation also reduces three values to one value.

Thus, it is a dimensionality-reduction operation at the pixel level.

---

### Range of the Output

Assuming

$$0 \leq R,G,B \leq 255$$

and all coefficients are non-negative:

$$Gray = 0.299R + 0.587G + 0.114B$$

The minimum occurs when

$$R=G=B=0$$

giving

$$Gray = 0$$

The maximum occurs when

$$R=G=B=255$$

giving

$$Gray = 255$$

Therefore,

$$0 \leq Gray \leq 255$$

The grayscale output remains within the same intensity range.

---

### Important Edge Cases

The input must represent a valid RGB image.

A valid image must have:

- At least one row.
- At least one column.
- Every row must have the same length.
- Every pixel must contain exactly three values.
- Every channel value must lie in $[0,255]$.

For example, this is invalid because the pixel has only two channels:

```text
[[[255, 0]]]
```

This is invalid because the rows have different lengths:

```text
[
    [[255, 0, 0], [0, 255, 0]],
    [[0, 0, 255]]
]
```

This is invalid because a channel is outside the allowed range:

```text
[[[300, 0, 0]]]
```

An empty image is also invalid:

```text
[]
```

The required response for these cases is

```text
-1
```

---

### Shape Transformation

The input shape is

$$H \times W \times 3$$

The output shape is

$$H \times W$$

For every input pixel:

$$3\ values \rightarrow 1\ value$$

Therefore, the total number of stored pixel values changes from

$$3HW$$

to

$$HW$$

This gives a threefold reduction in raw per-pixel channel values.

---

### Characteristics / Key Points

- RGB contains three color channels.
- Grayscale contains one intensity value per pixel.
- The luminosity method uses perceptual weights.
- Green receives the largest coefficient.
- Blue receives the smallest coefficient.
- The standard formula is $0.299R+0.587G+0.114B$.
- The coefficients sum to $1$.
- Valid RGB values lie in $[0,255]$.
- The resulting grayscale values remain in approximately $[0,255]$.
- Spatial dimensions are preserved.
- The channel dimension is removed.
- Every pixel can be processed independently.
- The operation does not require neighboring pixels.
- The transformation is therefore naturally parallelizable.
- Invalid image structures must be detected before conversion.
- Empty dimensions are invalid for this problem.
- Each output value is rounded to an integer.
- Grayscale conversion discards explicit color information.
- Brightness information is retained more effectively than with simple averaging.

---

### Luminosity vs Average

| Method     | Formula                | Main Idea                    |
| ---------- | ---------------------- | ---------------------------- |
| Average    | $\frac{R+G+B}{3}$      | Equal channel contribution   |
| Luminosity | $0.299R+0.587G+0.114B$ | Perceptual channel weighting |

The luminosity method is generally preferred when the grayscale image should preserve perceived brightness.

For example, pure red gives:

$$Gray_{avg} = \frac{255}{3} = 85$$

while luminosity gives:

$$Gray_{lum} = 0.299(255) = 76.245$$

The luminosity value better reflects the lower perceived brightness of pure red compared with pure green.

---

### Why is it used? / Applications

Grayscale conversion is widely used in computer vision and image processing.

#### Edge Detection

Many edge detection algorithms operate on intensity rather than color.

Examples include:

- Sobel filters
- Prewitt filters
- Canny edge detection

The image can first be converted to grayscale before detecting intensity changes.

---

#### Feature Extraction

Many classical computer vision algorithms operate on grayscale images.

Examples include:

- HOG
- SIFT
- Local Binary Patterns
- Texture descriptors

Removing color can simplify the representation while retaining structural information.

---

#### Image Classification

Some classification problems do not require explicit color information.

Converting images to grayscale can reduce the number of input channels from

$$3 \rightarrow 1$$

which reduces computational requirements.

---

#### Medical Imaging

Many medical images are primarily analyzed through intensity patterns.

Examples include:

- X-ray images
- CT scans
- Microscopy
- Certain ultrasound representations

Grayscale representations make intensity differences easier to process.

---

#### Optical Character Recognition

OCR systems frequently operate on grayscale or binary images.

Converting text images to grayscale can make preprocessing and feature extraction easier.

---

#### Computational Efficiency

An RGB image stores three values per pixel.

A grayscale image stores one.

For an image containing $HW$ pixels:

$$Memory_{RGB} \approx 3HW$$

while

$$Memory_{Gray} \approx HW$$

when the same data type is used.

Therefore, grayscale processing can reduce memory usage and computation in pipelines that do not need color.

---

> 💡 **Important Note**
>
> Grayscale conversion is **not** the same as simply removing two RGB channels. The three channels contain different brightness contributions, so a weighted combination is used to preserve perceived intensity.

---

> 💡 **Interview Tip**
>
> When converting RGB to grayscale, remember the standard luminosity formula: $Gray=0.299R+0.587G+0.114B$. The important idea is not memorizing the numbers alone, but understanding that green contributes more to perceived brightness than red and blue.

---

## Solutions

### Custom Implementation

```python
import numpy as np

def rgb_to_grayscale(image):
    if not isinstance(image, (list, np.ndarray)) or len(image) == 0:
        return -1
    if not isinstance(image[0], (list, np.ndarray)) or len(image[0]) == 0:
        return -1

    width = len(image[0])

    for row in image:
        if not isinstance(row, (list, np.ndarray)) or len(row) != width:
            return -1
        for pixel in row:
            if not isinstance(pixel, (list, np.ndarray)) or len(pixel) != 3:
                return -1
            if not all(0 <= value <= 255 for value in pixel):
                return -1

    result = []
    for row in image:
        new_row = []
        for r, g, b in row:
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            new_row.append(round(gray))
        result.append(new_row)

    return result
```

---

## Code Explanation

### Step 1: Validate the Input Type

The function first checks whether the input is a Python list or NumPy array.

```python
if not isinstance(image, (list, np.ndarray)) or len(image) == 0:
    return -1
```

An empty image cannot represent a valid image for this problem.

---

### Step 2: Validate the First Row

The first row establishes the expected width of the image.

```python
if not isinstance(image[0], (list, np.ndarray)) or len(image[0]) == 0:
    return -1
```

The image must have a non-empty width.

---

### Step 3: Check Consistent Row Lengths

Every row must contain the same number of pixels.

```python
width = len(image[0])

for row in image:
    if not isinstance(row, (list, np.ndarray)) or len(row) != width:
        return -1
```

This guarantees that the input represents a rectangular image.

---

### Step 4: Validate Each Pixel

Every pixel must contain exactly three values.

```python
if not isinstance(pixel, (list, np.ndarray)) or len(pixel) != 3:
    return -1
```

The three values correspond to:

```text
[R, G, B]
```

---

### Step 5: Validate Pixel Values

Each RGB value must satisfy

$$0 \leq value \leq 255$$

The implementation checks this using:

```python
if not all(0 <= value <= 255 for value in pixel):
    return -1
```

If any value is outside the valid range, the function immediately returns `-1`.

---

### Step 6: Process Every Pixel

After validation, the function iterates through every row and pixel.

```python
for row in image:
    new_row = []
    for r, g, b in row:
```

Each pixel is decomposed into its three channels.

---

### Step 7: Apply the Luminosity Formula

For every pixel:

```python
gray = 0.299 * r + 0.587 * g + 0.114 * b
```

This performs the weighted sum

$$Gray = 0.299R + 0.587G + 0.114B$$

---

### Step 8: Round the Result

The problem requires integer grayscale values.

```python
new_row.append(round(gray))
```

For example:

```text
149.685 -> 150
```

---

### Step 9: Construct the Output Image

Each converted pixel is appended to its row.

```python
result.append(new_row)
```

After all rows are processed, the result is returned.

The transformation is therefore:

$$H \times W \times 3 \rightarrow H \times W$$

---

### Algorithm

The complete algorithm can be summarized as:

1. Validate that the image is non-empty.
2. Determine the expected row width.
3. Validate every row.
4. Validate every pixel.
5. Ensure every pixel contains exactly three channels.
6. Ensure every RGB value lies in $[0,255]$.
7. For each pixel, compute the luminosity-weighted sum.
8. Round the resulting grayscale value.
9. Store the value in the corresponding output position.
10. Return the grayscale image.

---

### Pseudocode

```text
RGB_TO_GRAYSCALE(image)

    if image is invalid or empty
        return -1

    width = number of pixels in first row

    for each row in image
        if row is invalid or has incorrect width
            return -1

        for each pixel in row
            if pixel does not contain exactly 3 values
                return -1

            if any RGB value is outside [0, 255]
                return -1

    result = empty image

    for each row in image
        new_row = empty row

        for each (R, G, B) in row
            gray = 0.299R + 0.587G + 0.114B
            append round(gray) to new_row

        append new_row to result

    return result
```

---

### Why Validation Comes First

The conversion itself is simple:

$$Gray = 0.299R + 0.587G + 0.114B$$

Most of the implementation is actually concerned with ensuring that the input has the expected structure.

Without validation, malformed input could cause:

- Index errors.
- Incorrect output dimensions.
- Invalid pixel calculations.
- Unexpected numerical results.
- Inconsistent row lengths.

Therefore, the algorithm separates the problem into two stages:

```text
Validation
    ↓
Conversion
```

This is a useful pattern in preprocessing pipelines.

---

### Independent Pixel Processing

An important property of grayscale conversion is that each output pixel depends only on the corresponding RGB pixel.

For pixel $(i,j)$:

$$G_{i,j} = f(R_{i,j},G_{i,j},B_{i,j})$$

There is no dependence on neighboring pixels.

Therefore:

- Pixel $(0,0)$ can be processed independently.
- Pixel $(0,1)$ can be processed independently.
- Pixel $(1,0)$ can be processed independently.
- And so on.

This makes grayscale conversion naturally suitable for vectorization and parallel processing.

---

### Vectorized NumPy View

If the input is already a properly shaped NumPy array, the core mathematical operation can be expressed as:

```python
weights = np.array([0.299, 0.587, 0.114])
gray = np.sum(image * weights, axis=2)
```

The multiplication broadcasts the three weights across every pixel.

The summation collapses the channel dimension:

$$H \times W \times 3 \rightarrow H \times W$$

The Deep-ML problem additionally requires validation and a 2D list as output, so the custom implementation explicitly handles those requirements.

---

## Time & Space Complexity

Let:

- $H$ be the image height.
- $W$ be the image width.
- $C=3$ be the number of RGB channels.

Validation examines every pixel and every channel.

Therefore, validation takes

$$O(HWC)$$

Since $C=3$ is constant:

$$O(HW)$$

The grayscale conversion also processes every pixel once:

$$O(HW)$$

Therefore, the overall time complexity is

$$O(HW)$$

---

### Space Complexity

The output grayscale image contains $H \times W$ values.

Therefore, the output requires

$$O(HW)$$

space.

The implementation also creates temporary row structures while constructing the result, but these are bounded by the output size.

Thus, the auxiliary/output space is

$$O(HW)$$

If output space is excluded from the analysis, the algorithm uses only

$$O(W)$$

additional temporary space for the current row.

| Complexity      | Value                         |
| --------------- | ----------------------------- |
| Time            | **O(H × W)**                  |
| Space           | **O(H × W)** including output |
| Auxiliary Space | **O(W)** excluding output     |

where:

- $H$ is image height.
- $W$ is image width.
- $C=3$ is the fixed number of RGB channels.
