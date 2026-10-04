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
    └── 6_Image Morphology.pdf
```

## Topics Covered

| Lab   | Topic               | Key Concepts                                             |
| ----- | ------------------- | -------------------------------------------------------- |
| Lab 1 | Introduction        | MATLAB/Python basics for image processing                |
| Lab 2 | Basic Operations    | Image loading, manipulation, and visualization           |
| Lab 3 | Pixel Geometry      | 4-adjacency, 8-adjacency, m-adjacency, path finding      |
| Lab 4 | Spatial Enhancement | Point processing, histogram operations, filters          |
| Lab 5 | Enhancement         | Additional spatial domain techniques                     |
| Lab 6 | Image Restoration   | Noise removal (Gaussian, Salt & Pepper), deblurring      |
| Lab 7 | Frequency Domain    | DFT, Gaussian LPF, Butterworth HPF, FFT analysis         |
| Lab 8 | Deblurring          | Inverse filtering, Wiener filtering                      |
| Lab 9 | Morphology          | Erosion, Dilation, Opening, Closing, Boundary extraction |

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
- Python solution files (`lab*_solution.py`) with implementations
- Sample images for practice
- Assignment PDFs for additional practice

To run the Jupyter notebooks:

```bash
jupyter notebook "Lab Practicals/Lab1/1.ipynb"
```

To run the Python solutions:

```bash
python "Lab Practicals/Lab3/lab3_solution.py"
```

## License

This repository is for educational purposes as part of the NCSH302 course.
