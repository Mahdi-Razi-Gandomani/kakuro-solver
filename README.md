# Kakuro Puzzle Solver

This project is a Kakuro puzzle solver that uses image processing, deep learning, and constraint satisfaction algorithms to solve Kakuro puzzles from images. The system processes an image of a Kakuro puzzle, extracts the grid and constraints, solves the puzzle, and overlays the solution on the original image.

---

## Features

- **Image Processing**: Extracts the Kakuro grid and constraints from an image.
- **Deep Learning**: Uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize digits in the puzzle.
- **Constraint Satisfaction**: Solves the Kakuro puzzle using a backtracking algorithm with constraint propagation.
- **Visualization**: Overlays the solved puzzle on the original image for visualization.

---

## Code Structure

### Image Processing
- **Loading and Preprocessing**:
  - The input image is loaded and converted to grayscale.
  - Edges are detected using the Canny edge detector.
  - Contours are extracted to identify the Kakuro grid.

- **Perspective Transformation**:
  - The grid is transformed using perspective transformation.
  - The grid is divided into cells for further processing.

### Deep Learning
- **MNIST Model**:
  - A CNN model is trained on the MNIST dataset to recognize digits.
  - The model is used to detect and classify digits in the puzzle constraints.

### Puzzle Solving
- **Constraint Extraction**:
  - Constraints (sum hints) are extracted from the grid using the detected digits.
  - Diagonal lines are detected to identify constraint cells.

- **Backtracking Algorithm**:
  - A backtracking algorithm with constraint propagation is used to solve the Kakuro puzzle.
  - The algorithm ensures that the solution satisfies all row and column constraints.

### Visualization
- **Solution Overlay**:
  - The solved puzzle is overlaid on the original image using reverse perspective transformation.

---

## Usage

1. Place the Kakuro puzzle image (e.g., `s1.png`) in the project directory.

2. Run the script to solve the puzzle:

   ```bash
   python kak.py

3. The solved puzzle will be displayed, and the solution will be overlaid on the original image.

---
