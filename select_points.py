import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Function to open file dialog and select an image
def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Maze Image",
                                           filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    return file_path

# Load the image
image_path = select_image()
if not image_path:
    print("No image selected. Exiting...")
    exit()

maze = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
_, binary_maze = cv2.threshold(maze, 128, 255, cv2.THRESH_BINARY_INV)
maze_color = cv2.cvtColor(binary_maze, cv2.COLOR_GRAY2BGR)  # Convert to color image for visualization

# Store points
points = []

# Mouse click event
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left click to select points
        points.append((x, y))
        if len(points) == 1:
            cv2.circle(maze_color, (x, y), 5, (0, 255, 0), -1)  # Green for start
            print(f"Start point selected at: {x}, {y}")
        elif len(points) == 2:
            cv2.circle(maze_color, (x, y), 5, (0, 0, 255), -1)  # Red for end
            print(f"End point selected at: {x}, {y}")
        cv2.imshow("Select Start and End Points", maze_color)

# Show the maze image and wait for user input
cv2.imshow("Select Start and End Points", maze_color)
cv2.setMouseCallback("Select Start and End Points", select_points)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the selected points
if len(points) == 2:
    start, end = points
    print(f"Start: {start}, End: {end}")
else:
    print("Both start and end points must be selected!")
