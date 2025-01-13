import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

SIZE = 7

def MNIST():
    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Define and compile model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')  # 10 classes for digits 0-9
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15, batch_size=128)
    _, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_accuracy:.2f}')
    
    return model


def processImage(img):
    # Image preprocessing
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    area_list = [cv2.contourArea(cont) for cont in contours]
    p_list = [cv2.arcLength(cont, True) for cont in contours]
    page_index = np.argmax(area_list)
    doc_contour = cv2.approxPolyDP(contours[page_index], epsilon=0.005 * p_list[page_index], closed=True)
    cv2.drawContours(img, [doc_contour], -1, (0, 255, 0), 2)

    # Perspective transformation
    def sortPoints(points):
        points = points.reshape((4, 2))
        pointsSorted = np.zeros((4, 1, 2), dtype=np.int32)
        addPoints = points.sum(1)
        pointsSorted[0] = points[np.argmin(addPoints)]
        pointsSorted[3] = points[np.argmax(addPoints)]
        diffPoints = np.diff(points, axis=1)
        pointsSorted[1] = points[np.argmin(diffPoints)]
        pointsSorted[2] = points[np.argmax(diffPoints)]
        return pointsSorted
    
    h, w, _ = img.shape
    doc_contour_reshaped = sortPoints(doc_contour).squeeze()
    p1 = np.float32(doc_contour_reshaped)
    p2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    arr_pres = cv2.getPerspectiveTransform(p1, p2)
    puzzle_image = cv2.warpPerspective(img, arr_pres, (w, h))

    # Extract cells
    cells = []
    M, N = puzzle_image.shape[0] // SIZE, puzzle_image.shape[1] // SIZE
    for i in range(SIZE):
        cells.append([])
        for j in range(SIZE):
            cell = puzzle_image[i * M:(i + 1) * M, j * N:(j + 1) * N]
            cells[i].append(cell)

    # Display the original and processed
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(puzzle_image, cmap='gray')
    plt.title('Processed Image')
    plt.axis('off')
    plt.show()
    
    return puzzle_image, cells, arr_pres, doc_contour


def reverseProcess(processed_image, arr_pres, original_img, doc_contour):
    # Reverse perspective transformation
    result = np.array(original_img)
    arr_pres_inv = np.linalg.inv(arr_pres)
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(processed_image, arr_pres_inv, (w, h))
    warped = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    # Combine images
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.fillPoly(mask, [doc_contour], (255, 255, 255))
    mask_bool = mask > 0
    result = np.where(mask_bool, warped, result)
    
    return result


def detect_diagonal_line(cell):
     # Detect diagonal lines (Constraint cells)
    edges = cv2.Canny(cell, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)
    if lines is not None:
        for _, theta in lines[:, 0]:
            angle = np.degrees(theta)
            angle = angle % 180
            if 40 <= angle <= 50 or 130 <= angle <= 140:
                return True 

    return False


def getDigit(image, model):
        # Find and filter contours by area and aspect ratio to find digits
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            if 100 < area < 600 and aspect_ratio < 1:
                filtered_contours.append((x, y, w, h))
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        
        # Extract and preprocess digits for prediction
        digit_images = []
        for (x, y, w, h) in filtered_contours:
            digit = binary_image[y : y + h, x : x + w]
            digit = cv2.resize(digit, (8, 20))
            final_digit = np.zeros((28, 28))
            final_digit[4:24, 10:18] = digit
            d = x < y
            digit_images.append((final_digit, x, d))
        
        # Predict digit labels using the model
        dgts = []
        for digit in digit_images:
            resized_digit = digit[0].astype('float32') / 255.0
            reshaped_digit = np.expand_dims(resized_digit, axis=-1)
            processed_digits_array = np.expand_dims(reshaped_digit, axis=0)
            predictions = model.predict(processed_digits_array)
            predicted_labels = np.argmax(predictions, axis=1)
            dgts.append((predicted_labels, digit[1], digit[2]))
        
        # Separate vertical and horizontal constraints
        dgts.sort(key=lambda x: x[1])
        l_list = [digit for digit in dgts if digit[2]]
        r_list = [digit for digit in dgts if not digit[2]]
        l, r = -1, -1
        if len(l_list) > 0:
            l = int("".join(str(item[0][0]) for item in l_list))
        if len(r_list) > 0:
            r = int("".join(str(item[0][0]) for item in r_list))

        return l, r


def getBoard(cells):
    # Generate board from cells
    model = MNIST()
    board = [[(0, 0) for _ in range(SIZE)] for _ in range(SIZE)]
    for i in range(SIZE):
        for j in range(SIZE):
            if detect_diagonal_line(cells[i][j]):
                l, r = getDigit(cells[i][j], model)
                board[i][j] = (l, r)
            else:
                board[i][j] = (0, 0)
    
    return board


def setBoard(puzzle_image):
    # Update board with solutions
    M, N = puzzle_image.shape[0] // SIZE, puzzle_image.shape[1] // SIZE
    for i in range(SIZE):
        for j in range(SIZE):
            if board[i][j][0] == 0:
                text = str(board[i][j][1])
                position = (j * M + M//2, i * N + N//2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3
                color = (0, 0, 255)
                thickness = 5
                cv2.putText(puzzle_image, text, position, font, font_scale, color, thickness)
                
    return puzzle_image


############# CSP FOR KAKURO #############

# Find the next empty cell in the board
def find_empty_location(board, pos):
    for row in range(SIZE):
        for col in range(SIZE):
            if(board[row][col]== (0, 0)):
                pos[0]= row
                pos[1]= col
                return True
    return False

# Find constraints (sum hints) for a given cell
def find_cons(board, row, col):
    r = row
    
    # Find vertical constraint
    while r >= 0:
        if board[r][c][0] != 0:
            v = (r, c)
            break
        r -= 1

    c = col

    # Find horizontal constraint
    while c >= 0:
        if board[r][c][0] != 0:
            h = (r, c)
            break
        c -= 1
    
    return v, h


# Check if placing a number is valid
def is_valid(board, row, col, num):
    v, h = find_cons(board, row, col)

    # Check vertical sum validity
    r = v[0]
    c = v[1]
    vv = board[r][c][0]
    sum = num
    rem = 0
    r += 1
    while r < SIZE and board[r][c][0] == 0:
        if board[r][c][1] == num:
            return False
        sum += board[r][c][1]
        if board[r][c][1] == 0:
            rem += 1
        r += 1

    if vv != -1 and sum > vv:
        return False
    if vv != -1 and rem == 1 and sum != vv:
        return False
    if (rem - 1) * 9 + sum < vv:
        return False

    # Check horizontal sum validity
    r = h[0]
    c = h[1]
    hv = board[r][c][1]
    sum = num
    rem = 0
    c += 1
    while c < SIZE and board[r][c][0] == 0:
        if board[r][c][1] == num:
            return False
        sum += board[r][c][1]
        if board[r][c][1] == 0:
            rem += 1
        c += 1

    if hv != -1 and sum > hv:
        return False
    if hv != -1 and rem == 1 and sum != hv:
        return False
    if (rem - 1) * 9 + sum < hv:
        return False
    
    return True


# Solve the Kakuro puzzle using backtracking
def solve_sudoku(board):    
    pos =[0, 0]

    # If no empty cell is found, puzzle is solved
    if(not find_empty_location(board, pos)):
        return True

    row = pos[0]
    col = pos[1]
    
    for num in range(1, 10):
        if(is_valid(board, row, col, num)):
            board[row][col]= (0, num)
            if(solve_sudoku(board)):
                return True

            board[row][col] = (0, 0)
                  
    return False


if __name__ == '__main__':
    image_path = 's1.png'
    img = Image.open(image_path)
    puzzle_image, cells, arr_pres, cont = processImage(img)
    board = getBoard(cells)
    if(solve_sudoku(board)):
        page = setBoard(puzzle_image)
        restored_image = reverseProcess(page, arr_pres, img, cont)
        plt.imshow(restored_image, cmap='gray')
        plt.title('Solved Puzzle')
        plt.axis('off')
        plt.show()
    else:
        print ("No solution exists")