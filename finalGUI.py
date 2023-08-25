import cv2
import tkinter as tk
from tkinter import PhotoImage, Button
import threading  # Add this import for threading
import time

# Your imports for process, sudoku, model_wrapper, and preprocess
from current import process, sudoku
from models import model_wrapper
from preprocessing import preprocess

class SudokuSolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sudoku Solver GUI")

        self.cap = cv2.VideoCapture(0)
        self.frame_rate = 30
        self.processing_thread = None
        self.running = False

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.processed_label = tk.Label(root)
        self.processed_label.pack()

        self.start_button = Button(root, text="Start Processing", command=self.start_processing)
        self.start_button.pack()

        self.stop_button = Button(root, text="Stop Processing", command=self.stop_processing)
        self.stop_button.pack()

        self.load_model()

    def load_model(self):
        # Load your model using model_wrapper
        self.my_model = model_wrapper.model_wrapper(None, False, None, "model.hdf5")

    def start_processing(self):
        if not self.running:
            self.running = True
            self.processing_thread = threading.Thread(target=self.process_video)
            self.processing_thread.start()

    def stop_processing(self):
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()

    def process_video(self):
        prev = time.time()

        while self.running:
            time_elapsed = time.time() - prev
            success, img = self.cap.read()

            if time_elapsed > 1. / self.frame_rate:
                prev = time.time()
                # Your processing logic here (similar to your existing while loop)

                # Display the video feed
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tk = PhotoImage(data=img_rgb.tobytes())
                self.video_label.config(image=img_tk)
                self.video_label.image = img_tk

                # Display the processed output
                processed_output = process_and_get_processed_output(img)  # Replace with your processing function
                processed_img_tk = PhotoImage(data=processed_output.tobytes())
                self.processed_label.config(image=processed_img_tk)
                self.processed_label.image = processed_img_tk

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = SudokuSolverGUI(root)
    app.run()

