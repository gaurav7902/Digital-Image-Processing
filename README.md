# Digital Image Processing (NCSH302)

A collection of lab practicals, assignments, and notes for the Digital Image Processing course (NCSH302).

## Repository Structure

```
├── Lab Practicals/
│   ├── Lab1/          # Introduction to Digital Image Processing
│   ├── Lab2/          # Basic Image Operations
│   ├── Lab3/          # Pixel Geometry
│   ├── Lab4/          # Image Enhancement - Spatial Domain
│   ├── Lab5/          # Additional Enhancement Techniques
│   ├── Lab6/          # Image Restoration
│   ├── Lab7/          # Image Enhancement - Frequency Domain
│   ├── Lab8/          # Image Deblurring
│   ├── Lab9/          # Image Morphology
│   ├── Lab10/         # Image Segmentation
│   ├── Lab11/         # Advanced Segmentation
│   ├── Lab12/         # Segmentation and Compression
│   ├── requirements.txt
│   ├── Basics of Digital Image Processing.pdf
│   └── DIP_NCSH302_Lab Manual.pdf
│
└── Notes/
    ├── 1_DIP-Fundamentals.pdf
    ├── 2_Pixel Geometry.pdf
    ├── 3_Image Enhancement-Spatial domain.pdf
    ├── 4_Image Enhancement-Frequency domain.pdf
    ├── 5_Image Restoration.pdf
    ├── 6_Image Morphology.pdf
    ├── 7_Image Segmentation-I.pdf
    ├── 8_Image Segmentation-II.pdf
    └── 9_Image Compression.pdf
```

## Notes

The [`Notes/`](/home/gaurav/Desktop/Class/Clg/Digital%20Image%20Processing/Public-Repo/Notes) directory contains the theory material for the complete course sequence:

| Note | Topic                                |
| ---- | ------------------------------------ |
| 1    | DIP Fundamentals                     |
| 2    | Pixel Geometry                       |
| 3    | Image Enhancement — Spatial Domain   |
| 4    | Image Enhancement — Frequency Domain |
| 5    | Image Restoration                    |
| 6    | Image Morphology                     |
| 7    | Image Segmentation I                 |
| 8    | Image Segmentation II                |
| 9    | Image Compression                    |

These notes complement the practical notebooks and provide the mathematical
background for both the built-in and DSA/manual solutions.

## Topics Covered

| Lab    | Topic                        | Key Concepts                                                      |
| ------ | ---------------------------- | ----------------------------------------------------------------- |
| Lab 1  | Introduction                 | MATLAB/Python basics for image processing                         |
| Lab 2  | Basic Operations             | Image loading, manipulation, and visualization                    |
| Lab 3  | Pixel Geometry               | 4-adjacency, 8-adjacency, m-adjacency, path finding               |
| Lab 4  | Spatial Enhancement          | Point processing, histogram operations, filters                   |
| Lab 5  | Enhancement                  | Additional spatial domain techniques                              |
| Lab 6  | Image Restoration            | Noise removal (Gaussian, Salt & Pepper), deblurring               |
| Lab 7  | Frequency Domain             | DFT, Gaussian LPF, Butterworth HPF, FFT analysis                  |
| Lab 8  | Deblurring                   | Inverse filtering, Wiener filtering                               |
| Lab 9  | Morphology                   | Erosion, Dilation, Opening, Closing, Boundary extraction          |
| Lab 10 | Segmentation I               | Thresholding, histogram modes, local thresholding, region growing |
| Lab 11 | Segmentation II              | Region splitting/merging, line detection, gradient edges          |
| Lab 12 | Segmentation and Compression | Canny edges, Hough transform, compression foundations             |

## Installation

1. Clone the repository:

```bash
git clone https://github.com/gaurav7902/Digital-Image-Processing.git
cd Digital-Image-Processing
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r "Lab Practicals/requirements.txt"
```

## Requirements

- Python 3.x
- opencv-python
- matplotlib
- numpy

## Usage

Each lab directory contains:

- A Jupyter notebook (`.ipynb`) with the lab exercises
- The original `1.ipynb` reference notebook, where present, preserved unchanged
- An `Assignment_<no>.ipynb` notebook containing two solutions for every problem:
    - **Built-in library method:** OpenCV/NumPy/MATLAB-style library operations
    - **DSA/manual method:** explicit arrays, loops, formulas, recursion, queues, or other data structures and algorithms
- Sample images for practice
- Assignment PDFs for additional practice

The dual-solution convention applies to Labs 1–12, including the segmentation
assignments. The built-in version shows the practical API, while the DSA version
makes the underlying algorithm inspectable. Only the numbered assignment
notebooks are updated for this convention; existing `1.ipynb` notebooks are kept.

To run an assignment notebook from its lab directory:

```bash
cd "Lab Practicals/Lab10"
jupyter notebook Assignment_8.ipynb
```

To run the preserved introductory notebook:

```bash
jupyter notebook "Lab Practicals/Lab1/1.ipynb"
```

## License

This repository is for educational purposes as part of the NCSH302 course.
