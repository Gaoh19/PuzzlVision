import cv2
import numpy as np
import tensorflow as tf
from helpers import process_helpers

def create_grid_mask(vertical, horizontal):
    """
    Create a mask representing the Sudoku grid.

    Parameters:
        vertical (numpy.ndarray): Image with vertical lines.
        horizontal (numpy.ndarray): Image with horizontal lines.

    Returns:
        numpy.ndarray: Mask representing the Sudoku grid.
    """
    grid = cv2.add(horizontal, vertical)
    grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    pts = cv2.HoughLines(grid, .3, np.pi / 90, 200)
    lines = process_helpers.draw_lines(grid, pts)
    mask = cv2.bitwise_not(lines)

    return mask

def get_grid_lines(img, length=10):
    """
    Get grid lines from the input image.

    Parameters:
        img (numpy.ndarray): Input image.
        length (int): Distance between lines.

    Returns:
        tuple: Vertical and horizontal lines.
    """
    horizontal = process_helpers.grid_line_helper(img, 1, length)
    vertical = process_helpers.grid_line_helper(img, 0, length)
    return vertical, horizontal

def find_contours(img, original):
    """
    Find contours in the image and identify the Sudoku grid.

    Parameters:
        img (numpy.ndarray): Thresholded image.
        original (numpy.ndarray): Original image.

    Returns:
        list: List of extreme corners of the identified Sudoku grid.
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, closed=True)
        num_corners = len(approx)

        if num_corners == 4 and area > 1000:
            polygon = cnt
            break

    if polygon is not None:
        top_left = process_helpers.find_extreme_corners(polygon, min, np.add)
        top_right = process_helpers.find_extreme_corners(polygon, max, np.subtract)
        bot_left = process_helpers.find_extreme_corners(polygon, min, np.subtract)
        bot_right = process_helpers.find_extreme_corners(polygon, max, np.add)

        if bot_right[1] - top_right[1] == 0 or not (0.95 < ((top_right[0] - top_left[0]) / (bot_right[1] - top_right[1])) < 1.05):
            return []

        cv2.drawContours(original, [polygon], 0, (0, 0, 255), 3)

        [process_helpers.draw_extreme_corners(x, original) for x in [top_left, top_right, bot_right, bot_left]]

        return [top_left, top_right, bot_right, bot_left]

    return []

def warp_image(corners, original):
    """
    Warp the image based on identified corner points.

    Parameters:
        corners (list): Corner points of the Sudoku grid.
        original (numpy.ndarray): Original image.

    Returns:
        tuple: Warped image and transformation matrix.
    """
    corners = np.array(corners, dtype='float32')
    top_left, top_right, bot_right, bot_left = corners

    width = int(max([
        np.linalg.norm(top_right - bot_right),
        np.linalg.norm(top_left - bot_left),
        np.linalg.norm(bot_right - bot_left),
        np.linalg.norm(top_left - top_right)
    ]))

    mapping = np.array([[0, 0], [width - 1, 0], [width - 1, width - 1], [0, width - 1]], dtype='float32')
    matrix = cv2.getPerspectiveTransform(corners, mapping)

    return cv2.warpPerspective(original, matrix, (width, width)), matrix

def split_into_squares(warped_img):
    """
    Split the warped image into individual Sudoku squares.

    Parameters:
        warped_img (numpy.ndarray): Warped image.

    Returns:
        list: List of individual Sudoku squares.
    """
    squares = []
    width = warped_img.shape[0] // 9

    for j in range(9):
        for i in range(9):
            p1 = (i * width, j * width)
            p2 = ((i + 1) * width, (j + 1) * width)
            squares.append(warped_img[p1[1]:p2[1], p1[0]:p2[0]])

    return squares

def clean_squares(squares):
    """
    Clean Sudoku squares by removing unwanted regions.

    Parameters:
        squares (list): List of Sudoku squares.

    Returns:
        list: Cleaned list of Sudoku squares.
    """
    cleaned_squares = []

    for square in squares:
        new_img, is_number = process_helpers.clean_helper(square)

        if is_number:
            cleaned_squares.append(new_img)
        else:
            cleaned_squares.append(0)

    return cleaned_squares

def recognize_digits(squares_processed, model):
    """
    Recognize digits in the cleaned Sudoku squares.

    Parameters:
        squares_processed (list): List of cleaned Sudoku squares.
        model (tf.keras.Model): Trained digit recognition model.

    Returns:
        str: Recognized digits as a string.
    """
    s = ""
    formatted_squares = []
    location_of_zeroes = set()
    blank_image = np.zeros_like(cv2.resize(squares_processed[0], (32, 32)))

    for i in range(len(squares_processed)):
        if type(squares_processed[i]) == int:
            location_of_zeroes.add(i)
            formatted_squares.append(blank_image)
        else:
            img = cv2.resize(squares_processed[i], (32, 32))
            formatted_squares.append(img)

    formatted_squares = np.array(formatted_squares)
    all_preds = list(map(np.argmax, model(tf.convert_to_tensor(formatted_squares))))

    for i in range(len(all_preds)):
        if i in location_of_zeroes:
            s += "0"
        else:
            s += str(all_preds[i] + 1)

    return s

def draw_digits_on_warped(warped_img, solved_puzzle, squares_processed):
    """
    Draw the solved Sudoku puzzle digits on the warped image.

    Parameters:
        warped_img (numpy.ndarray): Warped image.
        solved_puzzle (str): Solved Sudoku puzzle digits.
        squares_processed (list): List of cleaned Sudoku squares.

    Returns:
        numpy.ndarray: Image with drawn digits.
    """
    width = warped_img.shape[0] // 9
    img_w_text = np.zeros_like(warped_img)
    index = 0

    for j in range(9):
        for i in range(9):
            if type(squares_processed[index]) == int:
                p1 = (i * width, j * width)
                p2 = ((i + 1) * width, (j + 1) * width)

                center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                text_size, _ = cv2.getTextSize(str(solved_puzzle[index]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 4)
                text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

                cv2.putText(warped_img, str(solved_puzzle[index]),
                            text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            index += 1

    return img_w_text

def unwarp_image(img_src, img_dest, pts, time):
    """
    Unwarp the image and combine with the destination image.

    Parameters:
        img_src (numpy.ndarray): Source image.
        img_dest (numpy.ndarray): Destination image.
        pts (list): Corner points of the original Sudoku grid.
        time (str): Time taken to solve the puzzle.

    Returns:
        numpy.ndarray: Unwarped and combined image.
    """
    pts = np.array(pts)
    height, width = img_src.shape[0], img_src.shape[1]
    pts_source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, width - 1]],
                          dtype='float32')

    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(img_src, h, (img_dest.shape[1], img_dest.shape[0]))
    cv2.fillConvexPoly(img_dest, pts, 0, 16)

    dst_img = cv2.add(img_dest, warped)

    dst_img_height, dst_img_width = dst_img.shape[0], dst_img.shape[1]
    cv2.putText(dst_img, time, (dst_img_width - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    return dst_img
