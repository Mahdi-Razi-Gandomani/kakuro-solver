from matplotlib import pyplot as plt
from PIL import Image
from imageProcessing import processImage, getBoard, setBoard, reverseProcess
from csp import solve_kakuro


SIZE = 7

if __name__ == '__main__':
    image_path = 's1.png'
    img = Image.open(image_path)
    puzzle_image, cells, arr_pres, cont = processImage(img, SIZE)
    board = getBoard(cells, SIZE)
    if(solve_kakuro(board, SIZE)):
        page = setBoard(puzzle_image, board, SIZE)
        restored_image = reverseProcess(page, arr_pres, img, cont)
        plt.imshow(restored_image, cmap='gray')
        plt.title('Solved Puzzle')
        plt.axis('off')
        plt.show()
    else:
        print ("No solution exists")