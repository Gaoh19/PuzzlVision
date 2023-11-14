# Import necessary libraries
import time as t
import cv2
from current import process, sudoku
from models import model_wrapper
from preprocessing import preprocess

# Set the frame dimensions for the camera
frameWidth = 960
frameHeight = 720

# Open the camera capture object
cap = cv2.VideoCapture(0)

# Set the frame rate
frame_rate = 30

# Set the frame width and height
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# Set the brightness of the camera
cap.set(10, 150)

# Load the model with weights
my_model = model_wrapper.model_wrapper(None, False, None, "model.hdf5")

# Initialize variables for timing
prev = 0

# Create a dictionary to store seen puzzles and their solutions
seen = dict()

# Main loop for capturing and processing video frames
while True:
    # Calculate time elapsed since the last frame
    time_elapsed = t.time() - prev

    # Read a frame from the camera
    success, img = cap.read()

    # Check if it's time to process a new frame based on the frame rate
    if time_elapsed > 1. / frame_rate:
        prev = t.time()  # Update the time of the last processed frame

        # Create copies of the original frame for processing and displaying
        img_result = img.copy()
        img_corners = img.copy()

        # Preprocess the image to enhance features
        processed_img = preprocess.preprocess(img)
        
        # Find the contours (corners) in the image
        corners = process.find_contours(processed_img, img_corners)

        # If corners are found, proceed with Sudoku grid extraction and recognition
        if corners:
            # Warp the image to a top-down view
            warped, matrix = process.warp_image(corners, img)
            warped_processed = preprocess.preprocess(warped)

            # Extract grid lines
            vertical_lines, horizontal_lines = process.get_grid_lines(warped_processed)
            mask = process.create_grid_mask(vertical_lines, horizontal_lines)
            numbers = cv2.bitwise_and(warped_processed, mask)

            # Split the grid into individual squares
            squares = process.split_into_squares(numbers)
            squares_processed = process.clean_squares(squares)

            # Recognize digits in each square using the trained model
            squares_guesses = process.recognize_digits(squares_processed, my_model)

            # If the puzzle has been seen and determined to be unsolvable, continue to the next frame
            if squares_guesses in seen and seen[squares_guesses] is False:
                continue

            # If the puzzle has been solved before, fetch the solution
            if squares_guesses in seen:
                process.draw_digits_on_warped(warped, seen[squares_guesses][0], squares_processed)
                img_result = process.unwarp_image(warped, img_result, corners, seen[squares_guesses][1])

            # If the puzzle is new, solve it and store the solution
            else:
                solved_puzzle, time = sudoku.solve_wrapper(squares_guesses)
                if solved_puzzle is not None:
                    process.draw_digits_on_warped(warped, solved_puzzle, squares_processed)
                    img_result = process.unwarp_image(warped, img_result, corners, time)
                    seen[squares_guesses] = [solved_puzzle, time]
                else:
                    seen[squares_guesses] = False  # Mark the puzzle as unsolvable

    # Display the result
    cv2.imshow('window', img_result)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all windows and release the camera
cv2.destroyAllWindows()
cap.release()
