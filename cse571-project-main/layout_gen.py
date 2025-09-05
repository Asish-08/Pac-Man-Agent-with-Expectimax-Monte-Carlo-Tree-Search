import numpy as np
import matplotlib.pyplot as plt

# Define parameters
num_points = 1000  # Number of data points
mu = 100           # Mean
sigma = 20         # Standard deviation
min_val = 50       # Minimum value
max_val = 150      # Maximum value

# Generate normally distributed data
data = np.random.normal(mu, sigma, num_points)

# Clip the data to ensure it falls within the desired range
data = np.clip(data, min_val, max_val)

# Plot the histogram
bin_width = 10  # Interval width
num_bins = int((max_val - min_val) / bin_width)
plt.hist(data, bins=num_bins, range=(min_val, max_val), color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Number of Walls')
plt.ylabel('Frequency')
plt.title('Normal Distribution of Number of Walls')

# Set x-axis ticks to intervals of 10
plt.xticks(np.arange(min_val, max_val+1, 10))

# Display the plot
plt.grid(True)
plt.show()


#import numpy as np

# Parameters
#mean_complexity = 0.35  # Mean complexity
##std_dev = 0.05  # Standard deviation
##num_layouts = 1000  # Number of layouts to generate
#grid_size = 10  # Initial guess for grid size

# Calculate total number of cells in the grid
#total_cells = (grid_size ** 2) * num_layouts

# Calculate the total number of walls based on the mean complexity
#total_walls = mean_complexity * total_cells

# Calculate the actual mean complexity based on the calculated total walls and cells
#actual_mean_complexity = total_walls / total_cells

# Re-estimate grid size using the actual mean complexity
#estimated_grid_size = np.sqrt(total_cells / total_walls)

#print("Total number of cells:", total_cells)
#print("Total number of walls:", total_walls)
#print("Actual mean complexity:", actual_mean_complexity)
#print("Estimated grid size:", estimated_grid_size)
