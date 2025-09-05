import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def calculate_complexity(grid):
    if grid is None:
        return None
    num_walls = np.sum(grid == '%')
    grid_size = np.prod(grid.shape)
    complexity = num_walls / grid_size
    return complexity

def is_normally_distributed(data):
    if data is None:
        return False
    filtered_data = [float(value) for value in data.flatten() if value.isdigit()]
    if len(filtered_data) < 3:
        print("Insufficient data points for normality test.")
        return False
    
    _, p_value = stats.shapiro(filtered_data)
    if p_value < 0.05:
        return False
    else:
        return True

def load_grid_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            grid = []
            for line in file:
                grid.append(list(line.strip()))
        return np.array(grid)
    except Exception as e:
        print(f"Error loading grid from file '{file_path}': {e}")
        return None

layouts_folder = "layouts"
complexities = []  # List to store calculated complexities

# Process each .lay file in the layouts folder
for filename in os.listdir(layouts_folder):
    if filename.endswith(".lay"):
        file_path = os.path.join(layouts_folder, filename)
        grid = load_grid_from_file(file_path)
        
        print(f"Processing {filename}:")
        
        if grid is None:
            print(f"Failed to load grid from file '{file_path}'. Skipping...")
            continue
        
        complexity = calculate_complexity(grid)
        if complexity is None:
            print(f"Failed to calculate complexity for '{filename}'.")
        else:
            complexities.append(complexity)  # Append complexity to list
            print("Complexity of the grid:", complexity)
        
        data = grid.flatten()
        if is_normally_distributed(data):
            print("The data extracted from the grid is normally distributed.")
        else:
            print("The data extracted from the grid is not normally distributed.")
        
        print()  # Add a newline for clarity between files

# Plot the complexities
plt.hist(complexities, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Complexity')
plt.ylabel('Frequency')
plt.title('Distribution of Grid Complexities')
plt.grid(True)
plt.show()
