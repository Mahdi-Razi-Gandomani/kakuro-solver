from matplotlib import pyplot as plt
from PIL import Image
from imageProcessing import processImage, getBoard, setBoard, reverseProcess
from csp import solve_kakuro

# Define the size of the Kakuro puzzle grid
SIZE = 7

if __name__ == '__main__':
    # Load the puzzle image
    image_path = 's1.png'
    img = Image.open(image_path)
    # Detect puzzle boundaries, correct perspective, and extract individual cells
    puzzle_image, cells, arr_pres, cont = processImage(img, SIZE)
    # Convert the processed cells into a board representation
    board = getBoard(cells, SIZE)
    # Solve the Kakuro puzzle using backtracking
    if(solve_kakuro(board, SIZE)):
        # Draw the solved numbers on the puzzle image
        page = setBoard(puzzle_image, board, SIZE)
        # Reverse the perspective transformation to overlay the solution on the original image
        restored_image = reverseProcess(page, arr_pres, img, cont)
        # Display the solved puzzle
        plt.imshow(restored_image, cmap='gray')
        plt.title('Solved Puzzle')
        plt.axis('off')
        plt.show()
    # If no solution exists
    else:
        print ("No solution exists")
