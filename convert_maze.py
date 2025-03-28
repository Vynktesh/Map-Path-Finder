import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    """Preprocess image to extract street edges."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Adaptive threshold for better road detection
    adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)

    blurred = cv2.GaussianBlur(adaptive_thresh, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Increase connectivity for street networks
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    return edges

def extract_graph(edges):
    """Convert street edges into a graph."""
    G = nx.Graph()
    h, w = edges.shape

    # Increased node density for better connectivity
    for y in range(0, h, 3):
        for x in range(0, w, 3):
            if edges[y, x] > 0:
                G.add_node((x, y))
                for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3), (-3, -3), (3, -3), (-3, 3), (3, 3)]:
                    nx_pos, ny_pos = x + dx, y + dy
                    if 0 <= nx_pos < w and 0 <= ny_pos < h and edges[ny_pos, nx_pos] > 0:
                        G.add_edge((x, y), (nx_pos, ny_pos))

    return G

def find_nearest_node(G, point):
    """Find the closest valid node in the graph to a given point."""
    if point in G:
        return point

    return min(G.nodes, key=lambda node: np.linalg.norm(np.array(node) - np.array(point)))

def find_shortest_path(G, start, end):
    """Find shortest path, ensuring start and end exist in the graph."""
    start = find_nearest_node(G, start)
    end = find_nearest_node(G, end)

    print(f"Adjusted Start: {start}, Adjusted End: {end}")  # Debugging information

    try:
        path = nx.shortest_path(G, source=start, target=end, weight=None)
        return path
    except nx.NetworkXNoPath:
        return None

def visualize_graph(image_path, G):
    """Debugging: Show graph nodes on the map."""
    image = cv2.imread(image_path)
    for (x, y) in G.nodes:
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Green dots for nodes
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Graph Overlay on Map")
    plt.show()

def visualize_path(image_path, path):
    """Display the map with the shortest path drawn."""
    image = cv2.imread(image_path)

    if path:
        for i in range(1, len(path)):
            cv2.line(image, path[i-1], path[i], (255, 0, 0), 2)  # Blue path
        cv2.circle(image, path[0], 7, (0, 255, 0), -1)  # Green - Start
        cv2.circle(image, path[-1], 7, (0, 0, 255), -1) # Red - End

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Path Visualization")
    plt.show()

def get_user_points(image_path):
    """Allow user to select start and end points by clicking."""
    global start, end
    start, end = None, None

    def mouse_callback(event, x, y, flags, param):
        global start, end
        if event == cv2.EVENT_LBUTTONDOWN:
            if start is None:
                start = (x, y)
                print(f"Start selected: {start}")
            elif end is None:
                end = (x, y)
                print(f"End selected: {end}")
                cv2.destroyAllWindows()  # Close selection window

    image = cv2.imread(image_path)
    cv2.imshow("Select Start and End Points", image)
    cv2.setMouseCallback("Select Start and End Points", mouse_callback)

    while end is None:
        cv2.waitKey(1)

    return start, end

# Main execution
image_path = "image.png"
edges = preprocess_image(image_path)
G = extract_graph(edges)

# Debugging: Show extracted graph
visualize_graph(image_path, G)

start, end = get_user_points(image_path)

path = find_shortest_path(G, start, end)
if path:
    visualize_path(image_path, path)
else:
    print("❌ No path found. Try selecting different points.")
