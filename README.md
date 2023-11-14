# PuzzlVision - Image Based Sudoku Neural Network

Before anything else, I'd like to thank the random Samaritan on the internet who was kind enough to provide a detailed article on the build-up of the project. It was immensely helpful. Shoutout to [McGill School of Computer Science](https://www.cs.mcgill.ca/~aassaf9/python/sudoku.txt) for providing the Sudoku algorithm.

The GitHub repository hosts an Image-based Sudoku Solver project, combining Computer Vision and Machine Learning. Leveraging OpenCV for image processing and TensorFlow for ML integration, the solver captures real-time video frames to recognize and solve Sudoku puzzles. The repository includes code for contour detection, perspective transformation, and grid recognition, showcasing a comprehensive solution. A pre-trained neural network facilitates accurate digit recognition, making the solver adept at handling diverse puzzles. The project is well-documented, offering insights into its implementation and potential extensions. The integration of Tkinter for a user-friendly interface adds to its accessibility and usability.

This program serves as a way to calculate the solution to any 9x9 sudoku puzzle via webcam.
It identifies the puzzle through the webcam, processes it uses OpenCV, runs against a neural network to predict the digits, 
and runs an efficient sudoku solver to determine the answer. It then displays the answer on the same frame if it is solvable.

## Overview

PuzzlVision is an innovative project that combines computer vision and machine learning to create an image-based Sudoku solver. The system captures real-time video frames, processes Sudoku grids, and employs a pre-trained neural network for accurate digit recognition, ultimately solving Sudoku puzzles.

## Relevant Packages:

Tested using `Python 3.6` (newer versions may or may not work)

- `opencv-python`: 4.3.0.36
- `numpy`: 1.19.1
- `tensorflow`: 2.2.0
- `sklearn`: 0.0
- `keras`: 2.3.1



## Features

- **Real-Time Sudoku Solver:** Capture and process video frames to solve Sudoku puzzles in real-time.
- **Machine Learning Integration:** Utilize a pre-trained neural network for accurate digit recognition.
- **Comprehensive Workflow:** Include contour detection, perspective transformation, and grid recognition in the image processing pipeline.
- **User-Friendly Interface:** Potential integration with Tkinter for an intuitive user interface for puzzle input and solution display.

## Technologies Used

- **OpenCV:** For image processing and contour detection.
- **TensorFlow:** For machine learning model integration.
- **Neural Network:** A pre-trained model for accurate digit recognition.
- **Tkinter:** (Optional) For creating a user-friendly interface.

## Project Structure

- `helpers/`: Contains auxiliary scripts and utility functions used across the project.
- `current/`: Hosts the main source code for the image-based Sudoku solver. Mainly the primary algorithm and process segregations.
- `original/`: Stores the original, unprocessed images for reference and comparison.
- `preprocessing/`: Includes image preprocessing scripts to enhance image quality before digit recognition.
- `models/`: Holds pre-trained machine learning models for accurate digit recognition.

## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/puzzlvision.git
    cd puzzlvision
    ```

2. Run the main script:

    ```bash
    python app.py
    ```

## Usage

- **Capture Input:** Point the camera towards a Sudoku puzzle.
- **Real-Time Processing:** The system processes video frames and displays the solution.
- **User Interface (Optional):** If using Tkinter, interact with the user-friendly interface for puzzle input.

## License

This project is licensed under the [MIT Licence](https://github.com/Gaoh19/PuzzlVision/blob/main/LICENSE).

## Acknowledgments

- [Credits to the Sudoku Solver Algorithm](https://www.cs.mcgill.ca/~aassaf9/python/sudoku.txt)
- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [McGill School of Computer Science](https://www.cs.mcgill.ca)
- [ChatGPT](https://chat.openai.com/)

