import numpy as np
import cv2
from matplotlib import pyplot as plt
from model import MNIST


def processImage(img, size):
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
    M, N = puzzle_image.shape[0] // size, puzzle_image.shape[1] // size
    for i in range(size):
        cells.append([])
        for j in range(size):
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
    h, w = result.shape[:2]
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


def getBoard(cells, size):
    # Generate board from cells
    model = MNIST()
    board = [[(0, 0) for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if detect_diagonal_line(cells[i][j]):
                l, r = getDigit(cells[i][j], model)
                board[i][j] = (l, r)
            else:
                board[i][j] = (0, 0)
    
    return board


def setBoard(puzzle_image, board, size):
    # Update board with solutions
    M, N = puzzle_image.shape[0] // size, puzzle_image.shape[1] // size
    for i in range(size):
        for j in range(size):
            if board[i][j][0] == 0:
                text = str(board[i][j][1])
                position = (j * M + M//2, i * N + N//2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3
                color = (0, 0, 255)
                thickness = 5
                cv2.putText(puzzle_image, text, position, font, font_scale, color, thickness)
                
    return puzzle_image