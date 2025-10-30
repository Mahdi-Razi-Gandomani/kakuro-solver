# Kakuro Puzzle Solver

This project is a Kakuro puzzle solver that uses image processing, deep learning, and constraint satisfaction algorithms to solve Kakuro puzzles from images. The system processes an image of a Kakuro puzzle, extracts the grid and constraints, solves the puzzle, and overlays the solution on the original image.

---

## Features

- **Image Processing**: Extracts the Kakuro grid and constraints from an image.
- **Deep Learning**: Uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize digits in the puzzle.
- **Constraint Satisfaction**: Solves the Kakuro puzzle using a backtracking algorithm with constraint propagation.
- **Visualization**: Overlays the solved puzzle on the original image for visualization.

---

## Files and Their Purpose

- **`main.py`**: The entry point of the application. It imports functions from `imageProcessing.py` and `csp.py` to process an image of a Kakuro puzzle, solve it , and show the solved puzzle.

- **`imageProcessing.py`**: Contains functions for processing images of Kakuro puzzles such as functions to processes the input image to extract the puzzle grid, or extracts the board configuration from the processed image.

- **`csp.py`**: Implements the constraint satisfaction problem (CSP) algorithm to solve the Kakuro puzzle.

- **`model.py`**: Uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize digits in the puzzle.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Mahdi-Razi-Gandomani/kakuro-solver.git
   cd kakuro-solver
2. Run the solver:
   ```bash
   python3 main.py

