import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Function to open file dialog and select an image
def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select Maze Image",
                                           filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    return file_path

# Get the image path from the user
image_path = select_image()
if not image_path:
    print("No image selected. Exiting...")
    exit()

# Load the maze image
maze = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply thresholding to convert to binary (black & white)
_, binary_maze = cv2.threshold(maze, 128, 255, cv2.THRESH_BINARY_INV)

# Display the processed maze
plt.figure(figsize=(6,6))
plt.imshow(binary_maze, cmap='gray')
plt.title("Processed Maze")
plt.show()
