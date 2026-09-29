# Exported from experiments/vlsid results/Untitled1.ipynb
# Historical code: review paths and side effects before executing.


# %% Original cell 0
import matplotlib.pyplot as plt
import random

def plot_values_from_file(filename):
    """
    Reads numerical values from a text file and plots them as dots on an X-Y plane.

    The index of each value is used as its x-coordinate, and the value itself
    is used as its y-coordinate. The y-axis is automatically scaled to the data's min and max.

    Args:
        filename (str): The path to the text file containing space-separated values.
    """
    values = []
    try:
        # Open the file and read all lines.
        with open(filename, 'r') as f:
            for line in f:
                # For each line, split it into individual string values,
                # convert each to a float, and add to our list.
                # This handles files with values on multiple lines.
                try:
                    line_values = [float(val) for val in line.strip().split()]
                    values.extend(line_values)
                except ValueError:
                    print(f"Warning: Could not convert some items in a line to numbers. Skipping line: '{line.strip()}'")

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    if not values:
        print("No numerical values found in the file to plot.")
        return

    # Create the x-coordinates, which will be the index of each value (0, 1, 2, ...).
    x_coords = range(len(values))
    # The y-coordinates are the values themselves.
    y_coords = values

    # --- Plotting ---
    # Create a figure and axes for the plot. figsize makes the plot larger.
    plt.figure(figsize=(12, 8))

    # Create a scatter plot. 's' controls the size of the dots.
    # For a large number of points, a smaller size (s=1) and lower alpha might be better.
    plt.scatter(x_coords, y_coords, s=1, alpha=0.6)

    # Add a title and labels to the axes for clarity.
    plt.title(f"Visualization of Values from '{filename}'", fontsize=16)
    plt.xlabel("Index of Value", fontsize=12)
    plt.ylabel("Value", fontsize=12)

    # Set the y-axis limits to the min and max of the data for a tight fit.
    if y_coords:
        plt.ylim(min(y_coords), max(y_coords))

    # Add a grid to make the plot easier to read.
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Display the plot.
    print(f"Successfully plotted {len(values)} values.")
    plt.show()


# --- Example of how to use the function ---
if __name__ == "__main__":
    # Define the name of the file to read from.
    # This should be the file you created with the previous script.
    data_file = "appended_values2.txt"

    # Call the function to plot the values from your file.
    plot_values_from_file(data_file)


# %% Original cell 1
import matplotlib.pyplot as plt
import numpy as np

# Read data from file
filename = "appended_values2.txt"  # replace with your txt file name
with open(filename, "r") as f:
    data = f.read().split()

# Convert to float or int
values = [float(x) for x in data]

# Repeat x-axis 0..12 for each block
x = np.tile(np.arange(13), len(values) // 13)
y = values[:len(x)]  # ensure same length

# Scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(x, y, s=10, alpha=0.6, c="blue")  # s = marker size
plt.xlabel("Block Position (0–12)")
plt.ylabel("Value from file")
plt.title("Scatter Plot of Values per Block")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# %% Original cell 2
import matplotlib.pyplot as plt
import numpy as np

# --- Read data from file ---
filename = "appended_values2.txt"  # replace with your txt file name
with open(filename, "r") as f:
    data = f.read().split()

# Convert to float
values = np.array([float(x) for x in data])

# Repeat x-axis 0..12 for each block
x = np.tile(np.arange(13), len(values) // 13)
y = values[:len(x)]  # ensure same length

# --- Plot ---
plt.style.use("seaborn-v0_8")  # professional style

plt.figure(figsize=(14, 7))
scatter = plt.scatter(
    x, y,
    c=y, cmap="viridis",  # color by value
    s=18, alpha=0.7, edgecolors="k", linewidths=0.3
)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label("Value Intensity", fontsize=12)

# Mean line per block position
means = [np.mean(y[x == i]) for i in range(13)]
plt.plot(np.arange(13), means, color="red", linewidth=2, marker="o", label="Mean per Position")

# Labels & title
plt.xlabel("Block Position (0–12)", fontsize=14)
plt.ylabel("Values from File", fontsize=14)
plt.title("Scatter Distribution of Values per Block", fontsize=16, fontweight="bold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()


# %% Original cell 3
import matplotlib.pyplot as plt
import numpy as np

# --- Create a dummy data file for demonstration ---
# This part is just to make the script runnable.
# You can comment it out if you have your "appended_values2.txt" file.
# np.random.seed(42)
# sample_values = []
# for i in range(13):
#     # Create some variation and a general trend for visual interest
#     sample_values.extend(np.random.normal(loc=i + np.sin(i), scale=1.5, size=20))
# with open("appended_values2.txt", "w") as f:
#     f.write(" ".join(map(str, sample_values)))


# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    data = [str(v) for v in sample_values] # Fallback to dummy data if file not found

# --- Prepare data for plotting ---
# Convert data from string to float
values = np.array([float(x) for x in data if x]) # Added 'if x' to handle potential empty strings

# Create the x-axis by repeating the sequence 0, 1, 2, ..., 12
# This ensures that for each block of values, we have the correct x-position
num_blocks = len(values) // 13
x = np.tile(np.arange(13), num_blocks)

# Ensure y has the same length as x, trimming any extra values
y = values[:len(x)]

# --- Create the Plot ---
# Use 'seaborn-v0_8-whitegrid' for a clean look with a white background and grid lines
plt.style.use("seaborn-v0_8-whitegrid")

# Create a figure and axes for the plot
fig, ax = plt.subplots(figsize=(10, 6))

# 1. Create the scatter plot with all the individual data points
# This is the core of the visualization, showing the distribution of every value.
scatter = ax.scatter(
    x, y,
    c=y,          # Color each point based on its y-value
    cmap="viridis", # Use the 'viridis' colormap for good visual perception
    s=20,         # Set the size of the points
    alpha=0.7,    # Set transparency to see overlapping points
    edgecolors="black", # Add a thin black edge to each point for definition
    linewidths=0.5
)

# 2. Add a colorbar to explain what the colors mean
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("Value Intensity", fontsize=14, fontweight="bold")

# 3. Calculate and plot the mean line
# This line summarizes the central tendency of the scattered points at each x-position.
means = [np.mean(y[x == i]) for i in range(13)]
ax.plot(
    np.arange(13),
    means,
    color="crimson",      # A strong red for high visibility
    linewidth=2.5,
    marker="o",           # Add circles at each mean point
    markersize=8,
    label="Mean per Position"
)

# --- Customize Labels, Title, and Ticks ---
# Set the labels for the x and y axes
ax.set_xlabel("Block Position", fontsize=16, fontweight="bold")
ax.set_ylabel("Measured Value", fontsize=16, fontweight="bold")

# Set a clear, descriptive title for the plot
ax.set_title("Distribution of Measured Values per Block Position", fontsize=18, fontweight="bold", pad=20)

# Set the x-axis ticks to be exactly 0, 1, 2, ..., 12
ax.set_xticks(np.arange(13))

# Customize tick label font size for readability
ax.tick_params(axis='both', which='major', labelsize=12)

# Add a legend to identify the mean line
ax.legend(fontsize=12)

# Ensure the layout is tight and clean
plt.tight_layout()

# Display the plot
plt.show()


# %% Original cell 4
# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    data = [str(v) for v in sample_values] # Fallback to dummy data if file not found

# --- Prepare data for plotting ---
# Convert data from string to float
values = np.array([float(x) for x in data if x]) # Added 'if x' to handle potential empty strings

# Create the x-axis by repeating the sequence 0, 1, 2, ..., 12
# This ensures that for each block of values, we have the correct x-position
num_blocks = len(values) // 13
x = np.tile(np.arange(13), num_blocks)

# Ensure y has the same length as x, trimming any extra values
y = values[:len(x)]

# --- Create the Plot ---
# Use 'seaborn-v0_8-whitegrid' for a clean look with a white background and grid lines
plt.style.use("seaborn-v0_8-whitegrid")

# Create a figure and axes for the plot
fig, ax = plt.subplots(figsize=(15, 8))

# 1. Create the scatter plot with all the individual data points
# This is the core of the visualization, showing the distribution of every value.
scatter = ax.scatter(
    x, y,
    c=y,          # Color each point based on its y-value
    cmap="viridis", # Use the 'viridis' colormap for good visual perception
    s=20,         # Set the size of the points
    alpha=0.6,    # Set transparency to see overlapping points
    edgecolors="black", # Add a thin black edge to each point for definition
    linewidths=0.5
)

# 2. Add a colorbar to explain what the colors mean
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("Value Intensity", fontsize=14, fontweight="bold")

# 3. Calculate and plot the mean and standard deviation
# These metrics summarize the central tendency and spread of the data at each position.
x_positions = np.arange(13)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds = np.array([np.std(y[x == i]) for i in x_positions])

# Plot the mean line
ax.plot(
    x_positions,
    means,
    color="crimson",      # A strong red for high visibility
    linewidth=2.5,
    marker="o",           # Add circles at each mean point
    markersize=8,
    label="Mean"
)

# Plot the standard deviation as a shaded area around the mean
ax.fill_between(
    x_positions,
    means - stds, # Lower bound of the shaded area
    means + stds, # Upper bound of the shaded area
    color="crimson",
    alpha=0.2,    # Use transparency to make it a subtle background
    label="Standard Deviation"
)


# --- Customize Labels, Title, and Ticks ---
# Set the labels for the x and y axes
ax.set_xlabel("Block Position", fontsize=16, fontweight="bold")
ax.set_ylabel("Measured Value", fontsize=16, fontweight="bold")

# Set a clear, descriptive title for the plot
ax.set_title("Distribution of Measured Values per Block Position", fontsize=18, fontweight="bold", pad=20)

# Set the x-axis ticks to be exactly 0, 1, 2, ..., 12
ax.set_xticks(x_positions)

# Customize tick label font size for readability
ax.tick_params(axis='both', which='major', labelsize=12)

# Add a legend to identify the mean line and standard deviation
ax.legend(fontsize=12)

# Ensure the layout is tight and clean
plt.tight_layout()

# Display the plot
plt.show()


# %% Original cell 5
# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    data = [str(v) for v in sample_values] # Fallback to dummy data if file not found

# --- Prepare data for plotting ---
# Convert data from string to float
values = np.array([float(x) for x in data if x]) # Added 'if x' to handle potential empty strings

# Create the x-axis by repeating the sequence 0, 1, 2, ..., 12
# This ensures that for each block of values, we have the correct x-position
num_blocks = len(values) // 13
x = np.tile(np.arange(13), num_blocks)

# Ensure y has the same length as x, trimming any extra values
y = values[:len(x)]

# --- Create the Plot ---
# Use 'seaborn-v0_8-whitegrid' for a clean look with a white background and grid lines
plt.style.use("seaborn-v0_8-whitegrid")

# Create a figure and axes for the plot
fig, ax = plt.subplots(figsize=(10, 6))

# 1. Create the scatter plot with all the individual data points
# This is the core of the visualization, showing the distribution of every value.
scatter = ax.scatter(
    x, y,
    c=y,          # Color each point based on its y-value
    cmap="viridis", # Use the 'viridis' colormap for good visual perception
    s=20,         # Set the size of the points
    alpha=0.6,    # Set transparency to see overlapping points
    edgecolors="black", # Add a thin black edge to each point for definition
    linewidths=0.5
)

# 2. Add a colorbar to explain what the colors mean
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("Value Intensity", fontsize=14, fontweight="bold")

# 3. Calculate and plot the mean, standard deviation, and variance
# These metrics summarize the central tendency and spread of the data at each position.
x_positions = np.arange(13)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds = np.array([np.std(y[x == i]) for i in x_positions])
variances = np.array([np.var(y[x == i]) for i in x_positions]) # Calculate variance

# Plot the mean line
ax.plot(
    x_positions,
    means,
    color="crimson",      # A strong red for high visibility
    linewidth=2.5,
    marker="o",           # Add circles at each mean point
    markersize=8,
    label="Mean"
)

# Plot the standard deviation as a shaded area around the mean
ax.fill_between(
    x_positions,
    means - stds, # Lower bound of the shaded area
    means + stds, # Upper bound of the shaded area
    color="crimson",
    alpha=0.2,    # Use transparency to make it a subtle background
    label="Standard Deviation"
)

# # Plot the variance as a separate line
# ax.plot(
#     x_positions,
#     variances,
#     color="darkorange",   # A distinct color for variance
#     linewidth=2,
#     marker="s",           # Use squares for variance points
#     linestyle="--",       # Use a dashed line
#     label="Variance"
# )


# --- Customize Labels, Title, and Ticks ---
# Set the labels for the x and y axes
ax.set_xlabel("Block Position", fontsize=16, fontweight="bold")
ax.set_ylabel("Measured Value", fontsize=16, fontweight="bold")

# Set a clear, descriptive title for the plot
ax.set_title("Distribution of Measured Values per Block Position", fontsize=18, fontweight="bold", pad=20)

# Set the x-axis ticks to be exactly 0, 1, 2, ..., 12
ax.set_xticks(x_positions)

# Customize tick label font size for readability
ax.tick_params(axis='both', which='major', labelsize=12)

# Add a legend to identify all plotted metrics
ax.legend(fontsize=12)

# Ensure the layout is tight and clean
plt.tight_layout()

# Display the plot
plt.show()


# %% Original cell 6
import matplotlib.pyplot as plt
import numpy as np

# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    data = [str(v) for v in np.random.randint(0, 100, 260)]  # fallback dummy data

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])
num_blocks = len(values) // 13
x = np.tile(np.arange(13), num_blocks)
y = values[:len(x)]

# --- Add jitter (spread points horizontally within block) ---
jitter_strength = 0.15  # how wide to spread points inside each block
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(9, 4))

# Scatter plot with jitter
scatter = ax.scatter(
    x_jittered, y,
    c=y, cmap="viridis",
    s=25, alpha=0.6,
    edgecolors="black", linewidths=0.4
)

# Colorbar
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("Value Intensity", fontsize=14, fontweight="bold")

# Mean & std shading
x_positions = np.arange(13)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

ax.plot(
    x_positions, means,
    color="crimson", marker="o", linewidth=2.5, label="Mean"
)

ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.2, label="Std. Dev."
)

# Labels & title
ax.set_xlabel("Block Position", fontsize=16, fontweight="bold")
ax.set_ylabel("Measured Value", fontsize=16, fontweight="bold")
ax.set_title("Distribution of Measured Values per Block Position", fontsize=18, fontweight="bold", pad=20)

ax.set_xticks(x_positions)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.legend(fontsize=12)

plt.tight_layout()
plt.show()


# %% Original cell 7
import matplotlib.pyplot as plt
import numpy as np

# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    data = [str(v) for v in np.random.randint(0, 100, 260)]  # fallback dummy data

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])
num_blocks = len(values) // 13
x = np.tile(np.arange(13), num_blocks)
y = values[:len(x)]

# --- Add jitter (spread points horizontally within block) ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot ---
fig, ax = plt.subplots(figsize=(9, 4))

# Scatter plot with jitter
scatter = ax.scatter(
    x_jittered, y,
    c=y, cmap="viridis",
    s=25, alpha=0.6,
    edgecolors="black", linewidths=0.4
)

# Colorbar
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("Value Intensity", fontsize=14, fontweight="bold")

# Mean & std shading
x_positions = np.arange(13)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

ax.plot(
    x_positions, means,
    color="crimson", marker="o", linewidth=2.5, label="Mean"
)

ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.2, label="Std. Dev."
)

# Labels & title
ax.set_xlabel("Block Position", fontsize=16, fontweight="bold")
ax.set_ylabel("Measured Value", fontsize=16, fontweight="bold")
ax.set_title("Distribution of Measured Values per Block Position", fontsize=18, fontweight="bold", pad=20)

ax.set_xticks(x_positions)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.legend(fontsize=12)

# --- Remove background grid & spines ---
ax.grid(False)  # remove grid lines
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)  # remove top/right borders

plt.tight_layout()
plt.show()


# %% Original cell 8
import matplotlib.pyplot as plt 
import numpy as np

# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    data = [str(v) for v in np.random.randint(0, 100, 260)]  # fallback dummy data

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])
num_blocks = len(values) // 13
x = np.tile(np.arange(13), num_blocks)
y = values[:len(x)]

# --- Add jitter (spread points horizontally within block) ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot ---
fig, ax = plt.subplots(figsize=(9, 4))

# Scatter plot with jitter
scatter = ax.scatter(
    x_jittered, y,
    c=y, cmap="viridis",
    s=25, alpha=0.6,
    edgecolors="black", linewidths=0.4
)

# ✅ Colorbar moved to the LEFT
cbar = fig.colorbar(scatter, ax=ax, location="left")
cbar.set_label("Value Intensity", fontsize=14, fontweight="bold")

# Mean & std shading
x_positions = np.arange(13)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

ax.plot(
    x_positions, means,
    color="crimson", marker="o", linewidth=2.5, label="Mean"
)

ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.2, label="Std. Dev."
)

# Labels & title
ax.set_xlabel("Block Position", fontsize=16, fontweight="bold")
ax.set_title("Distribution of Measured Values per Block Position", fontsize=18, fontweight="bold", pad=20)

ax.set_xticks(x_positions)
ax.tick_params(axis="x", which="major", labelsize=12)
ax.legend(fontsize=12)

# --- Remove background grid & spines ---
ax.grid(False)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)

# --- Remove y-axis ticks and labels ---
ax.set_yticks([])
ax.set_ylabel("")

plt.tight_layout()
plt.show()


# %% Original cell 9
import matplotlib.pyplot as plt 
import numpy as np

# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    data = [str(v) for v in np.random.randint(0, 100, 260)]  # fallback dummy data

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])
num_blocks = len(values) // 13
x = np.tile(np.arange(13), num_blocks)
y = values[:len(x)]

# --- Add jitter (spread points horizontally within block) ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot ---
fig, ax = plt.subplots(figsize=(9, 4))

# Scatter plot with jitter
scatter = ax.scatter(
    x_jittered, y,
    c=y, cmap="viridis",
    s=25, alpha=0.6,
    edgecolors="black", linewidths=0.4,
    vmin=-0.2  # ✅ ensures colorbar starts at -0.2
)

# ✅ Colorbar moved to the LEFT
cbar = fig.colorbar(scatter, ax=ax, location="left")
cbar.set_label("Value Intensity", fontsize=16, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=14, colors="black")

# Mean & std shading
x_positions = np.arange(13)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

ax.plot(
    x_positions, means,
    color="crimson", marker="o", linewidth=2.5, label="Mean"
)

ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.2, label="Std. Dev."
)

# Labels & title
ax.set_xlabel("Block Position", fontsize=15, fontweight="bold", color="black")
# ax.set_title("Distribution of Measured Values per Block Position", fontsize=20, fontweight="bold", pad=20, color="black")

ax.set_xticks(x_positions)
ax.tick_params(axis="x", which="major", labelsize=15, colors="black")  
ax.legend(fontsize=14)

# --- Remove background grid & spines ---
ax.grid(False)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)

# --- Remove y-axis ticks and labels ---
ax.set_yticks([])
ax.set_ylabel("")  

plt.tight_layout()
plt.show()


# %% Original cell 10
import matplotlib.pyplot as plt 
import numpy as np

# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    data = [str(v) for v in np.random.randint(0, 100, 260)]  # fallback dummy data

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])

# ✅ Shift block positions to start at -1
num_blocks = len(values) // 13
x = np.tile(np.arange(-1, 12), num_blocks)   # -1,0,1,...,11
y = values[:len(x)]

# --- Add jitter (spread points horizontally within block) ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot ---
fig, ax = plt.subplots(figsize=(8, 4))  # ✅ Bigger plot

# Scatter plot with jitter
scatter = ax.scatter(
    x_jittered, y,
    c=y, cmap="viridis",
    s=35, alpha=0.7,                 # ✅ Slightly bigger points
    edgecolors="black", linewidths=0.5,
    vmin=-0.4                        # ✅ colorbar starts at -0.2
)

# ✅ Colorbar moved to the LEFT, closer
cbar = fig.colorbar(scatter, ax=ax, location="left", pad=0.02)  
cbar.set_label("Value Intensity", fontsize=18, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=16, colors="black")

# Mean & std shading
x_positions = np.arange(-1, 12)  # ✅ same shifted positions
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

ax.plot(
    x_positions, means,
    color="crimson", marker="o", markersize=8, linewidth=3, label="Mean"
)

ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.25, label="Standard Deviation."
)

# Labels & title
ax.set_xlabel("Transformer Block Position", fontsize=15, fontweight="bold", color="black")
# ax.set_title("Distribution of Measured Values per Block Position", fontsize=22, fontweight="bold", pad=20, color="black")

ax.set_xticks(x_positions)
ax.tick_params(axis="x", which="major", labelsize=16, colors="black")  
ax.legend(fontsize=16)

# --- Remove background grid & spines ---
ax.grid(False)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)

# --- Remove y-axis ticks and labels ---
ax.set_yticks([])
ax.set_ylabel("")  

plt.tight_layout()
plt.show()


# %% Original cell 11
import matplotlib.pyplot as plt 
import numpy as np

# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    data = [str(v) for v in np.random.randint(0, 100, 260)]  # fallback dummy data

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])

# ✅ Shift block positions to start at -1
num_blocks = len(values) // 13
x = np.tile(np.arange(-1, 12), num_blocks)   # -1,0,1,...,11
y = values[:len(x)]
# ax.set_ylim(-0.2, 0.2)



# --- Add jitter (spread points horizontally within block) ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot ---
fig, ax = plt.subplots(figsize=(9, 5))  # ✅ Bigger plot

# Scatter plot with jitter
scatter = ax.scatter(
    x_jittered, y,
    c=y, cmap="viridis",
    s=35, alpha=0.7,                 
    edgecolors="black", linewidths=0.5,
    vmin=-0.2                        
)

# ✅ Colorbar moved to the RIGHT (default)
cbar = fig.colorbar(scatter, ax=ax, pad=0.02)  
cbar.set_label("Value Intensity", fontsize=18, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=16, colors="black")

# Mean & std shading
x_positions = np.arange(-1, 12)  # ✅ same shifted positions
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

ax.plot(
    x_positions, means,
    color="crimson", marker="o", markersize=8, linewidth=3, label="Mean"
)

ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.25, label="Standard Deviation."
)

# ✅ Print mean and std per block
print("\n📊 Block-wise Mean and Standard Deviation:\n")
for i, (m, s) in enumerate(zip(means, stds)):
    print(f"Block {i-1:2d} → Mean = {m:.3f}, Std Dev = {s:.3f}")

# Labels & title
ax.set_xlabel("Transformer Block Position", fontsize=15, fontweight="bold", color="black")
ax.set_ylabel("Average Correlation Co-efficient", fontsize=15, fontweight="bold", color="black")  # ✅ Keep y-axis
ax.set_xticks(x_positions)
ax.tick_params(axis="x", which="major", labelsize=16, colors="black")  
ax.tick_params(axis="y", which="major", labelsize=14, colors="black")  
ax.legend(fontsize=16)

# --- Remove background grid & spines ---
ax.grid(False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.show()


# %% Original cell 12
import matplotlib.pyplot as plt 
import numpy as np

# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    data = [str(v) for v in np.random.randint(0, 100, 260)]  # fallback dummy data

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])

# ✅ Shift block positions to start at -1
num_blocks = len(values) // 13
x = np.tile(np.arange(-1, 12), num_blocks)   # -1,0,1,...,11
y = values[:len(x)]

# --- Add jitter (spread points horizontally within block) ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot ---
fig, ax = plt.subplots(figsize=(8, 4))  # ✅ Bigger plot

# Scatter plot with jitter
scatter = ax.scatter(
    x_jittered, y,
    c=y, cmap="viridis",
    s=35, alpha=0.7,                 
    edgecolors="black", linewidths=0.5,
    vmin=-0.2                        
)

# ✅ Colorbar moved to the LEFT, closer
cbar = fig.colorbar(scatter, ax=ax, location="left", pad=0.02)  
cbar.set_label("Value Intensity", fontsize=18, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=16, colors="black")

# Mean & std shading
x_positions = np.arange(-1, 12)  # ✅ same shifted positions
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

ax.plot(
    x_positions, means,
    color="crimson", marker="o", markersize=8, linewidth=3, label="Mean"
)

ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.25, label="Standard Deviation."
)

# ✅ Print mean and std per block
print("\n📊 Block-wise Mean and Standard Deviation:\n")
for i, (m, s) in enumerate(zip(means, stds)):
    print(f"Block {i-1:2d} → Mean = {m:.3f}, Std Dev = {s:.3f}")

# Labels & title
ax.set_xlabel("Transformer Block Position", fontsize=15, fontweight="bold", color="black")
ax.set_xticks(x_positions)
ax.tick_params(axis="x", which="major", labelsize=16, colors="black")  
ax.legend(fontsize=16)

# --- Remove background grid & spines ---
ax.grid(False)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)

# --- Remove y-axis ticks and labels ---
ax.set_yticks([])
ax.set_ylabel("")  

plt.tight_layout()
plt.show()


# %% Original cell 13
import matplotlib.pyplot as plt
import numpy as np

# --- Read data from file ---
filename = "appended_values2.txt"
with open(filename, "r") as f:
    data = f.read().split()

# Convert to float
values = np.array([float(x) for x in data])

# Repeat x-axis 0..12 for each block
x = np.tile(np.arange(13), len(values) // 13)
y = values[:len(x)]  # ensure same length

# --- Plot ---
plt.figure(figsize=(6, 4), dpi=300)  # high resolution for paper

# Scatter plot (black markers, semi-transparent for density)
plt.scatter(x, y, s=10, alpha=0.5, c="black")

# Mean + std shading per block position
means = [np.mean(y[x == i]) for i in range(13)]
stds = [np.std(y[x == i]) for i in range(13)]
positions = np.arange(13)

plt.plot(positions, means, color="red", linewidth=1.5, marker="o", markersize=4, label="Mean")
plt.fill_between(positions, 
                 np.array(means)-np.array(stds), 
                 np.array(means)+np.array(stds),
                 color="red", alpha=0.15, label="±1 Std. Dev.")

# Labels (professional formatting)
plt.xlabel("Block Position", fontsize=12)
plt.ylabel("Value", fontsize=12)

# No unnecessary title, keep clean
plt.legend(frameon=False, fontsize=10)
plt.grid(True, linestyle="--", alpha=0.3)

# Tight layout for saving
plt.tight_layout()
plt.show()


# %% Original cell 14



# %% Original cell 15



# %% Original cell 16



# %% Original cell 17



# %% Original cell 18



# %% Original cell 19



# %% Original cell 20



# %% Original cell 21



# %% Original cell 22



# %% Original cell 23



# %% Original cell 24



# %% Original cell 25



# %% Original cell 26



# %% Original cell 27



# %% Original cell 28



# %% Original cell 29



# %% Original cell 30



# %% Original cell 31



# %% Original cell 32



# %% Original cell 33



# %% Original cell 34
import matplotlib.pyplot as plt 
import numpy as np

# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    data = [str(v) for v in np.random.randint(0, 100, 260)]  # fallback dummy data

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])

# ✅ Shift block positions to start at -1
num_blocks = len(values) // 13
x = np.tile(np.arange(-1, 12), num_blocks)   # -1,0,1,...,11
y = values[:len(x)]

# --- Add jitter (spread points horizontally within block) ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot ---
fig, ax = plt.subplots(figsize=(9, 5))  # ✅ Bigger plot

# Scatter plot with jitter
scatter = ax.scatter(
    x_jittered, y,
    c=y, cmap="viridis",
    s=35, alpha=0.7,                 
    edgecolors="black", linewidths=0.5,
    vmin=-0.2                        
)

# ✅ Colorbar on the RIGHT (default)
cbar = fig.colorbar(scatter, ax=ax, pad=0.02)  
cbar.set_label("Value Intensity", fontsize=18, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=16, colors="black")

# Mean & std shading
x_positions = np.arange(-1, 12)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

ax.plot(
    x_positions, means,
    color="crimson", marker="o", markersize=8, linewidth=3, label="Mean"
)

ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.25, label="Standard Deviation."
)

# ✅ Print mean and std per block
print("\n📊 Block-wise Mean and Standard Deviation:\n")
for i, (m, s) in enumerate(zip(means, stds)):
    print(f"Block {i-1:2d} → Mean = {m:.3f}, Std Dev = {s:.3f}")

# Labels & title
ax.set_xlabel("Transformer Block Position", fontsize=15, fontweight="bold", color="black")
ax.set_ylabel("Average Correlation Co-efficient", fontsize=15, fontweight="bold", color="black")

# ✅ Custom y-axis scale: -0.2, -0.15, -0.1, ...
ax.set_ylim(-0.2, 0.8)
ax.set_yticks(np.arange(-0.2, 1.01, 0.07))

# Formatting
ax.set_xticks(x_positions)
ax.tick_params(axis="x", which="major", labelsize=16, colors="black")  
ax.tick_params(axis="y", which="major", labelsize=14, colors="black")  
ax.legend(fontsize=16)

# --- Remove background grid & spines ---
ax.grid(False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.show()


# %% Original cell 35
import matplotlib.pyplot as plt 
import numpy as np

# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    data = [str(v) for v in np.random.randint(0, 100, 260)]  # fallback dummy data

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])

# ✅ Shift block positions to start at -1
num_blocks = len(values) // 13
x = np.tile(np.arange(-1, 12), num_blocks)   # -1,0,1,...,11
y = values[:len(x)]

# --- Add jitter (spread points horizontally within block) ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot ---
fig, ax = plt.subplots(figsize=(9, 5))

# Scatter plot with jitter
scatter = ax.scatter(
    x_jittered, y,
    c=y, cmap="viridis",
    s=35, alpha=0.7,                 
    edgecolors="black", linewidths=0.5,
    vmin=-0.2                        
)

# ✅ Colorbar on the RIGHT
cbar = fig.colorbar(scatter, ax=ax, pad=0.02)  
cbar.set_label("Value Intensity", fontsize=15, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=15, colors="black", direction="inout")  

# Mean & std shading
x_positions = np.arange(-1, 12)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

ax.plot(
    x_positions, means,
    color="crimson", marker="o", markersize=8, linewidth=3, label="Mean"
)

ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.25, label="Standard Deviation"
)

# ✅ Print mean and std per block
print("\n📊 Block-wise Mean and Standard Deviation:\n")
for i, (m, s) in enumerate(zip(means, stds)):
    print(f"Block {i-1:2d} → Mean = {m:.3f}, Std Dev = {s:.3f}")

# Labels & title
ax.set_xlabel("Transformer Block Position", fontsize=15, fontweight="bold", color="black")
ax.set_ylabel("Average Correlation Co-efficient", fontsize=15, fontweight="bold", color="black")

# ✅ Custom y-axis ticks
ax.set_ylim(-0.2, 0.8)
ax.set_yticks(np.arange(-0.2, 1.01, 0.07))

# ✅ X-axis ticks
ax.set_xticks(x_positions)

# ✅ Ticks on all sides
ax.tick_params(axis="x", which="major", labelsize=15, colors="black", direction="inout", top=True, bottom=True)  
ax.tick_params(axis="y", which="major", labelsize=15, colors="black", direction="inout", left=True, right=True)  

# ✅ Legend with border
legend = ax.legend(fontsize=15, frameon=True ,  loc="lower right")
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(1.2)

# ✅ Add horizontal grid lines (background)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)

# ✅ Add border around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

plt.tight_layout()
plt.show()


# %% Original cell 36
import matplotlib.pyplot as plt 
import numpy as np

# --- Read data from your file ---
filename = "appended_values3.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    data = [str(v) for v in np.random.randint(0, 100, 260)]  # fallback dummy data

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])

# ✅ Shift block positions to start at -1
num_blocks = len(values) // 13
x = np.tile(np.arange(-1, 12), num_blocks)   # -1,0,1,...,11
y = values[:len(x)]

# --- Add jitter (spread points horizontally within block) ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot ---
fig, ax = plt.subplots(figsize=(9, 5))

# Scatter plot with jitter
scatter = ax.scatter(
    x_jittered, y,
    c=y, cmap="viridis",
    s=35, alpha=0.7,                 
    edgecolors="black", linewidths=0.5,
    vmin=-0.2                        
)

# ✅ Colorbar on the RIGHT
cbar = fig.colorbar(scatter, ax=ax, pad=0.02)  
cbar.set_label("Value Intensity", fontsize=15, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=15, colors="black", direction="inout")  

# Mean & std shading
x_positions = np.arange(-1, 12)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

ax.plot(
    x_positions, means,
    color="crimson", marker="o", markersize=8, linewidth=3, label="Mean"
)

ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.25, label="Standard Deviation"
)

# ✅ Print mean and std per block
print("\n📊 Block-wise Mean and Standard Deviation:\n")
for i, (m, s) in enumerate(zip(means, stds)):
    print(f"Block {i-1:2d} → Mean = {m:.3f}, Std Dev = {s:.3f}")

# Labels & title
ax.set_xlabel("Transformer Block Position", fontsize=15, fontweight="bold", color="black")
ax.set_ylabel("Average Correlation Coefficient", fontsize=15, fontweight="bold", color="black")

# ✅ Custom y-axis ticks
ax.set_ylim(-0.2, 0.8)
ax.set_yticks(np.arange(-0.2, 1.01, 0.09))

# ✅ X-axis ticks
ax.set_xticks(x_positions)

# ✅ Ticks ONLY bottom (x) and left (y)
ax.tick_params(axis="x", which="major", labelsize=15, colors="black", direction="inout", top=False, bottom=True)  
ax.tick_params(axis="y", which="major", labelsize=15, colors="black", direction="inout", left=True, right=False)  

# ✅ Legend with border
legend = ax.legend(fontsize=15, frameon=True)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(1.2)

# ✅ Add ONLY horizontal grid lines
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.xaxis.grid(False)

# ✅ Add border around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

plt.tight_layout()
plt.savefig("plot22678r22.svg", format="svg")
plt.show()


# %% Original cell 37
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- Read data from your file ---
filename = "appended_values3.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    data = [str(v) for v in np.random.randint(0, 100, 260)]  # fallback dummy data

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])

# ✅ Shift block positions to start at -1
num_blocks = len(values) // 13
x = np.tile(np.arange(-1, 12), num_blocks)   # -1,0,1,...,11
y = values[:len(x)]

# --- Compute block-wise means & stds ---
x_positions = np.arange(-1, 12)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

# --- Compute deviation from block mean (for coloring) ---
deviations = np.array([yi - means[xi] for xi, yi in zip(x, y)])

# --- Normalize deviations per block ---
normed_colors = np.zeros_like(deviations)
for i in x_positions:
    block_dev = np.abs(deviations[x == i])  # abs deviation per block
    if len(block_dev) > 0:
        max_dev = block_dev.max()
        if max_dev == 0:  
            normed_colors[x == i] = 0.0
        else:
            normed_colors[x == i] = block_dev / max_dev  # normalize [0,1] within block

# --- Add jitter (spread points horizontally within block) ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot ---
fig, ax = plt.subplots(figsize=(9, 5))

# Use a colormap
cmap = cm.viridis

# Scatter plot with per-block normalized colors
scatter = ax.scatter(
    x_jittered, y,
    c=normed_colors, cmap=cmap,
    s=35, alpha=0.7,
    edgecolors="black", linewidths=0.5
)

# ✅ Colorbar (explaining per-block scaling)
cbar = fig.colorbar(
    cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=1), cmap=cmap),
    ax=ax, pad=0.02
)
cbar.set_label("Relative Deviation within Block", fontsize=15, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=15, colors="black", direction="inout")

# --- Plot block means & std shading ---
ax.plot(
    x_positions, means,
    color="crimson", marker="o", markersize=8, linewidth=3, label="Mean"
)
ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.25, label="Standard Deviation"
)

# ✅ Print block statistics
print("\n📊 Block-wise Mean, Std Dev, and Normalized Deviation Ranges:\n")
for i, (m, s) in enumerate(zip(means, stds)):
    block_dev = deviations[x == (i-1)]
    block_norm = normed_colors[x == (i-1)]
    if len(block_dev) > 0:
        print(f"Block {i-1:2d} → Mean = {m:.3f}, Std Dev = {s:.3f}, "
              f"Deviation Range = [{block_dev.min():.3f}, {block_dev.max():.3f}], "
              f"Norm Range = [{block_norm.min():.2f}, {block_norm.max():.2f}]")

# Labels & title
ax.set_xlabel("Transformer Block Position", fontsize=15, fontweight="bold", color="black")
ax.set_ylabel("Average Correlation Coefficient", fontsize=15, fontweight="bold", color="black")

# ✅ Custom y-axis ticks
ax.set_ylim(-0.2, 0.8)
ax.set_yticks(np.arange(-0.2, 1.01, 0.09))

# ✅ X-axis ticks
ax.set_xticks(x_positions)

# ✅ Ticks formatting
ax.tick_params(axis="x", which="major", labelsize=15, colors="black", direction="inout", top=False, bottom=True)
ax.tick_params(axis="y", which="major", labelsize=15, colors="black", direction="inout", left=True, right=False)

# ✅ Legend with border
legend = ax.legend(fontsize=15, frameon=True)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(1.2)

# ✅ Add ONLY horizontal grid lines
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.xaxis.grid(False)

# ✅ Add border around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

plt.tight_layout()
plt.savefig("plot_relative_deviation_per_block.svg", format="svg")
plt.show()


# %% Original cell 38
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# # --- Create a dummy data file for demonstration ---
# # This part is just to make the script runnable.
# np.random.seed(42)
# sample_values = []
# # We'll create data for 13 blocks, from position -1 to 11
# for i in range(-1, 12):
#     # Create some variation and a general trend for visual interest
#     sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
# with open("appended_values3.txt", "w") as f:
#     f.write(" ".join(map(str, sample_values)))


# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    data = [str(v) for v in sample_values] # Fallback to dummy data

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])

# Shift block positions to start at -1, as requested
num_blocks = len(values) // 13
x = np.tile(np.arange(-1, 12), num_blocks) # Block positions are now -1, 0, 1, ..., 11
y = values[:len(x)]

# --- Compute block-wise statistics ---
x_positions = np.arange(-1, 12)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

# --- Compute ABSOLUTE deviation and NORMALIZE per block ---
# 1. Calculate the absolute distance of each point from its block's mean
mean_map = {pos: mean for pos, mean in zip(x_positions, means)}
abs_deviations = np.array([abs(yi - mean_map[xi]) for xi, yi in zip(x, y)])

# 2. Normalize these deviations for each block to a [0, 1] scale
# This makes the color scale relative to the max deviation within each block.
normed_colors = np.zeros_like(abs_deviations, dtype=float)
for i in x_positions:
    block_mask = (x == i)
    block_abs_dev = abs_deviations[block_mask]
    if len(block_abs_dev) > 0:
        max_dev = np.max(block_abs_dev)
        # Avoid division by zero if all points in a block are the same
        if max_dev > 0:
            normed_colors[block_mask] = block_abs_dev / max_dev

# --- Add jitter to spread points horizontally ---
jitter_strength = 0.2
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(14, 8))

# 1. Scatter plot with jitter, colored by NORMALIZED ABSOLUTE deviation
# 'viridis' is a sequential colormap, perfect for showing magnitude.
# Low values (near the mean) will be dark, high values (far from the mean) will be bright yellow.
scatter = ax.scatter(
    x_jittered, y,
    c=normed_colors, cmap="viridis",
    s=30, alpha=0.8,
    edgecolors="black", linewidths=0.5
)

# 2. Colorbar showing the new normalized scale
cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label("Normalized Absolute Deviation from Mean", fontsize=14, fontweight="bold")
cbar.ax.tick_params(labelsize=11)

# 3. Plot the mean line and standard deviation shading
ax.plot(
    x_positions, means,
    color="crimson", marker="o", markersize=8, linewidth=2.5, label="Mean"
)
ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.2, label="Standard Deviation"
)

# --- Customize Labels, Title, and Ticks ---
ax.set_xlabel("Transformer Block Position", fontsize=15, fontweight="bold")
ax.set_ylabel("Average Correlation Coefficient", fontsize=15, fontweight="bold")
ax.set_title("Correlation Coefficient Distribution and Deviation per Block", fontsize=18, fontweight="bold", pad=20)

# Set x-axis ticks to match block positions
ax.set_xticks(x_positions)

# Customize tick label font size
ax.tick_params(axis='both', which='major', labelsize=12)

# Add a legend with a frame
legend = ax.legend(fontsize=12, frameon=True)
legend.get_frame().set_edgecolor("black")

# Keep only horizontal grid lines
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.xaxis.grid(False)

# Add a distinct border around the plot area
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

plt.tight_layout()
plt.show()


# %% Original cell 39
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# # --- Create a dummy data file for demonstration ---
# # This part is just to make the script runnable.
# np.random.seed(42)
# sample_values = []
# # We'll create data for 13 blocks, from position -1 to 11
# for i in range(-1, 12):
#     # Create some variation and a general trend for visual interest
#     sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
# with open("appended_values2.txt", "w") as f:
#     f.write(" ".join(map(str, sample_values)))


# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    # Fallback dummy data in case the file is missing
    sample_values = []
    for i in range(-1, 12):
        sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
    data = [str(v) for v in sample_values]

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])

# Shift block positions to start at -1, as requested
num_blocks = len(values) // 13
x = np.tile(np.arange(-1, 12), num_blocks) # Block positions are now -1, 0, 1, ..., 11
y = values[:len(x)]

# --- Compute block-wise statistics ---
x_positions = np.arange(-1, 12)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

# --- Compute ABSOLUTE deviation with a GLOBAL scale ---
# 1. Calculate the absolute distance of each point from its block's mean
mean_map = {pos: mean for pos, mean in zip(x_positions, means)}
abs_deviations = np.array([abs(yi - mean_map[xi]) for xi, yi in zip(x, y)])

# 2. Find the single maximum deviation across ALL blocks
global_max_dev = np.max(abs_deviations)

# --- Add jitter to spread points horizontally ---
jitter_strength = 0.2
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(14, 8))

# Define the color normalization based on the global max deviation
norm = mcolors.Normalize(vmin=0, vmax=global_max_dev)
cmap = "viridis"

# 1. Scatter plot with jitter, colored by GLOBAL ABSOLUTE deviation
# The color of each point is now directly comparable across all blocks.
scatter = ax.scatter(
    x_jittered, y,
    c=abs_deviations, cmap=cmap, norm=norm,
    s=30, alpha=0.8,
    edgecolors="black", linewidths=0.5
)

# 2. Colorbar showing the new global absolute scale
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
cbar.set_label("Absolute Deviation from Block Mean", fontsize=14, fontweight="bold")
cbar.ax.tick_params(labelsize=11)


# 3. Plot the mean line and standard deviation shading
ax.plot(
    x_positions, means,
    color="crimson", marker="o", markersize=8, linewidth=2.5, label="Mean"
)
ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.2, label="Standard Deviation"
)

# --- Customize Labels, Title, and Ticks ---
ax.set_xlabel("Transformer Block Position", fontsize=15, fontweight="bold")
ax.set_ylabel("Average Correlation Coefficient", fontsize=15, fontweight="bold")
ax.set_title("Correlation Coefficient Distribution and Deviation per Block", fontsize=18, fontweight="bold", pad=20)

# Set x-axis ticks to match block positions
ax.set_xticks(x_positions)

# Customize tick label font size
ax.tick_params(axis='both', which='major', labelsize=12)

# Add a legend with a frame
legend = ax.legend(fontsize=12, frameon=True)
legend.get_frame().set_edgecolor("black")

# Keep only horizontal grid lines
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.xaxis.grid(False)

# Add a distinct border around the plot area
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

plt.tight_layout()
plt.show()


# %% Original cell 40
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# # --- Create a dummy data file for demonstration ---
# np.random.seed(42)
# sample_values = []
# for i in range(-1, 12):
#     sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
# with open("appended_values2.txt", "w") as f:
#     f.write(" ".join(map(str, sample_values)))


# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    # Fallback dummy data in case the file is missing
    sample_values = []
    for i in range(-1, 12):
        sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
    data = [str(v) for v in sample_values]

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])

# Shift block positions to start at -1
num_blocks = len(values) // 13
x = np.tile(np.arange(-1, 12), num_blocks) # Block positions are now -1, 0, 1, ..., 11
y = values[:len(x)]

# --- Compute block-wise statistics ---
x_positions = np.arange(-1, 12)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

# --- Compute ABSOLUTE deviation with a GLOBAL scale ---
# 1. Calculate the absolute distance of each point from its block's mean
mean_map = {pos: mean for pos, mean in zip(x_positions, means)}
abs_deviations = np.array([abs(yi - mean_map[xi]) for xi, yi in zip(x, y)])

# 2. Find the single maximum deviation across ALL blocks
global_max_dev = np.max(abs_deviations)

# --- Add jitter to spread points horizontally ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot using the new design specifications ---
fig, ax = plt.subplots(figsize=(9, 5))

# Define the color normalization based on the global max deviation
norm = mcolors.Normalize(vmin=0, vmax=global_max_dev)
cmap = "viridis"

# 1. Scatter plot with jitter, colored by GLOBAL ABSOLUTE deviation
scatter = ax.scatter(
    x_jittered, y,
    c=abs_deviations, cmap=cmap, norm=norm,
    s=35, alpha=0.9,
    edgecolors="black", linewidths=0.5
)

# 2. Colorbar showing the global absolute scale
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
cbar.set_label("Absolute Deviation from Block Mean", fontsize=15, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=15, colors="black", direction="inout")

# 3. Plot the mean line and standard deviation shading
ax.plot(
    x_positions, means,
    color="crimson", marker="o", markersize=8, linewidth=3, label="Mean"
)
ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.25, label="Standard Deviation"
)

# --- Customize Labels, Title, and Ticks ---
ax.set_xlabel("Transformer Block Position", fontsize=15, fontweight="bold", color="black")
ax.set_ylabel("Average Correlation Coefficient", fontsize=15, fontweight="bold", color="black")

# Custom y-axis ticks
ax.set_ylim(-0.2, 0.8)
ax.set_yticks(np.arange(-0.2, 1.01, 0.09))

# X-axis ticks
ax.set_xticks(x_positions)

# Ticks formatting
ax.tick_params(axis="x", which="major", labelsize=15, colors="black", direction="inout", top=False, bottom=True)
ax.tick_params(axis="y", which="major", labelsize=15, colors="black", direction="inout", left=True, right=False)

# Legend with border
legend = ax.legend(fontsize=15, frameon=True)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(1.2)

# Add ONLY horizontal grid lines
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.xaxis.grid(False)

# Add border around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

plt.tight_layout()
plt.show()


# %% Original cell 41
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# # --- Create a dummy data file for demonstration ---
# np.random.seed(42)
# sample_values = []
# for i in range(-1, 12):
#     sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
# with open("appended_values2.txt", "w") as f:
#     f.write(" ".join(map(str, sample_values)))


# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    # Fallback dummy data in case the file is missing
    sample_values = []
    for i in range(-1, 12):
        sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
    data = [str(v) for v in sample_values]

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])

# Shift block positions to start at -1
num_blocks = len(values) // 13
x = np.tile(np.arange(-1, 12), num_blocks) # Block positions are now -1, 0, 1, ..., 11
y = values[:len(x)]

# --- Compute block-wise statistics ---
x_positions = np.arange(-1, 12)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

# --- Compute ABSOLUTE deviation with a GLOBAL scale ---
# 1. Calculate the absolute distance of each point from its block's mean
mean_map = {pos: mean for pos, mean in zip(x_positions, means)}
abs_deviations = np.array([abs(yi - mean_map[xi]) for xi, yi in zip(x, y)])

# 2. Find the single maximum deviation across ALL blocks
global_max_dev = np.max(abs_deviations)

# --- Add jitter to spread points horizontally ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot using the new design specifications ---
fig, ax = plt.subplots(figsize=(9, 5))

# --- Define DISCRETE color boundaries and normalization ---
# Create sharp color changes at intervals of 0.05
step = 0.05
boundaries = np.arange(0, global_max_dev + step, step)
# Correctly get the colormap using the modern API
cmap = plt.colormaps.get("viridis").resampled(len(boundaries) - 1)
norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

# 1. Scatter plot with jitter, colored by DISCRETE GLOBAL ABSOLUTE deviation
scatter = ax.scatter(
    x_jittered, y,
    c=abs_deviations, cmap=cmap, norm=norm,
    s=35, alpha=0.9,
    edgecolors="black", linewidths=0.5
)

# 2. Colorbar showing the discrete absolute scale
# Use the boundaries to create a stepped colorbar
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(mappable, ax=ax, pad=0.02, boundaries=boundaries)
cbar.set_label("Absolute Deviation from Block Mean", fontsize=15, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=15, colors="black", direction="inout")


# 3. Plot the mean line and standard deviation shading
ax.plot(
    x_positions, means,
    color="crimson", marker="o", markersize=8, linewidth=3, label="Mean"
)
ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.25, label="Standard Deviation"
)

# --- Customize Labels, Title, and Ticks ---
ax.set_xlabel("Transformer Block Position", fontsize=15, fontweight="bold", color="black")
ax.set_ylabel("Average Correlation Coefficient", fontsize=15, fontweight="bold", color="black")

# Custom y-axis ticks
ax.set_ylim(-0.2, 0.8)
ax.set_yticks(np.arange(-0.2, 1.01, 0.09))

# X-axis ticks
ax.set_xticks(x_positions)

# Ticks formatting
ax.tick_params(axis="x", which="major", labelsize=15, colors="black", direction="inout", top=False, bottom=True)
ax.tick_params(axis="y", which="major", labelsize=15, colors="black", direction="inout", left=True, right=False)

# Legend with border
legend = ax.legend(fontsize=15, frameon=True)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(1.2)

# Add ONLY horizontal grid lines
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.xaxis.grid(False)

# Add border around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

plt.tight_layout()
plt.show()


# %% Original cell 42
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# # --- Create a dummy data file for demonstration ---
# np.random.seed(42)
# sample_values = []
# for i in range(-1, 12):
#     sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
# with open("appended_values2.txt", "w") as f:
#     f.write(" ".join(map(str, sample_values)))


# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    # Fallback dummy data in case the file is missing
    sample_values = []
    for i in range(-1, 12):
        sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
    data = [str(v) for v in sample_values]

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])

# Shift block positions to start at -1
num_blocks = len(values) // 13
x = np.tile(np.arange(-1, 12), num_blocks) # Block positions are now -1, 0, 1, ..., 11
y = values[:len(x)]

# --- Compute block-wise statistics ---
x_positions = np.arange(-1, 12)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

# --- Compute ABSOLUTE deviation with a GLOBAL scale ---
# 1. Calculate the absolute distance of each point from its block's mean
mean_map = {pos: mean for pos, mean in zip(x_positions, means)}
abs_deviations = np.array([abs(yi - mean_map[xi]) for xi, yi in zip(x, y)])

# 2. Find the single maximum deviation across ALL blocks
global_max_dev = np.max(abs_deviations)

# --- Add jitter to spread points horizontally ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot using the new design specifications ---
fig, ax = plt.subplots(figsize=(9, 5))

# --- Define DISCRETE color boundaries and normalization ---
# Create sharp color changes at intervals of 0.05
step = 0.05
boundaries = np.arange(0, global_max_dev + step, step)
# Correctly get the colormap using the modern API - Changed to "plasma"
cmap = plt.colormaps.get("plasma").resampled(len(boundaries) - 1)
norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

# 1. Scatter plot with jitter, colored by DISCRETE GLOBAL ABSOLUTE deviation
scatter = ax.scatter(
    x_jittered, y,
    c=abs_deviations, cmap=cmap, norm=norm,
    s=35, alpha=0.9,
    edgecolors="black", linewidths=0.5
)

# 2. Colorbar showing the discrete absolute scale
# Use the boundaries to create a stepped colorbar
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(mappable, ax=ax, pad=0.02, boundaries=boundaries)
cbar.set_label("Absolute Deviation from Block Mean", fontsize=15, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=15, colors="black", direction="inout")


# 3. Plot the mean line and standard deviation shading
ax.plot(
    x_positions, means,
    color="crimson", marker="o", markersize=8, linewidth=3, label="Mean"
)
ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.25, label="Standard Deviation"
)

# --- Customize Labels, Title, and Ticks ---
ax.set_xlabel("Transformer Block Position", fontsize=15, fontweight="bold", color="black")
ax.set_ylabel("Average Correlation Coefficient", fontsize=15, fontweight="bold", color="black")

# Custom y-axis ticks
ax.set_ylim(-0.2, 0.8)
ax.set_yticks(np.arange(-0.2, 1.01, 0.09))

# X-axis ticks
ax.set_xticks(x_positions)

# Ticks formatting
ax.tick_params(axis="x", which="major", labelsize=15, colors="black", direction="inout", top=False, bottom=True)
ax.tick_params(axis="y", which="major", labelsize=15, colors="black", direction="inout", left=True, right=False)

# Legend with border
legend = ax.legend(fontsize=15, frameon=True)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(1.2)

# Add ONLY horizontal grid lines
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.xaxis.grid(False)

# Add border around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

plt.tight_layout()
plt.show()


# %% Original cell 43
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# # --- Create a dummy data file for demonstration ---
# np.random.seed(42)
# sample_values = []
# for i in range(-1, 12):
#     sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
# with open("appended_values2.txt", "w") as f:
#     f.write(" ".join(map(str, sample_values)))


# --- Read data from your file ---
filename = "appended_values2.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    # Fallback dummy data in case the file is missing
    sample_values = []
    for i in range(-1, 12):
        sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
    data = [str(v) for v in sample_values]

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])

# Shift block positions to start at -1
num_blocks = len(values) // 13
x = np.tile(np.arange(-1, 12), num_blocks) # Block positions are now -1, 0, 1, ..., 11
y = values[:len(x)]

# --- Compute block-wise statistics ---
x_positions = np.arange(-1, 12)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

# --- Compute ABSOLUTE deviation with a GLOBAL scale ---
# 1. Calculate the absolute distance of each point from its block's mean
mean_map = {pos: mean for pos, mean in zip(x_positions, means)}
abs_deviations = np.array([abs(yi - mean_map[xi]) for xi, yi in zip(x, y)])

# 2. Find the single maximum deviation across ALL blocks
global_max_dev = np.max(abs_deviations)

# --- Add jitter to spread points horizontally ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot using the new design specifications ---
fig, ax = plt.subplots(figsize=(9, 5))

# --- Define DISCRETE color boundaries and normalization ---
# Create sharp color changes at intervals of 0.05
step = 0.05
boundaries = np.arange(0, global_max_dev + step, step)
# Changed to "YlOrRd" for a light-to-dark, high-contrast colormap
cmap = plt.colormaps.get("YlOrRd").resampled(len(boundaries) - 1)
norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

# 1. Scatter plot with jitter, colored by DISCRETE GLOBAL ABSOLUTE deviation
scatter = ax.scatter(
    x_jittered, y,
    c=abs_deviations, cmap=cmap, norm=norm,
    s=35, alpha=0.9,
    edgecolors="black", linewidths=0.5
)

# 2. Colorbar showing the discrete absolute scale
# Use the boundaries to create a stepped colorbar
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(mappable, ax=ax, pad=0.02, boundaries=boundaries)
cbar.set_label("Absolute Deviation from Block Mean", fontsize=15, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=15, colors="black", direction="inout")


# 3. Plot the mean line and standard deviation shading
ax.plot(
    x_positions, means,
    color="crimson", marker="o", markersize=8, linewidth=3, label="Mean"
)
ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.25, label="Standard Deviation"
)

# --- Customize Labels, Title, and Ticks ---
ax.set_xlabel("Transformer Block Position", fontsize=15, fontweight="bold", color="black")
ax.set_ylabel("Average Correlation Coefficient", fontsize=15, fontweight="bold", color="black")

# Custom y-axis ticks
ax.set_ylim(-0.2, 0.8)
ax.set_yticks(np.arange(-0.2, 1.01, 0.09))

# X-axis ticks
ax.set_xticks(x_positions)

# Ticks formatting
ax.tick_params(axis="x", which="major", labelsize=15, colors="black", direction="inout", top=False, bottom=True)
ax.tick_params(axis="y", which="major", labelsize=15, colors="black", direction="inout", left=True, right=False)

# Legend with border
legend = ax.legend(fontsize=15, frameon=True)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(1.2)

# Add ONLY horizontal grid lines
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.xaxis.grid(False)

# Add border around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

plt.tight_layout()
plt.show()


# %% Original cell 44
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# # --- Create a dummy data file for demonstration ---
# np.random.seed(42)
# sample_values = []
# for i in range(-1, 12):
#     sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
# with open("appended_values2.txt", "w") as f:
#     f.write(" ".join(map(str, sample_values)))


# --- Read data from your file ---
filename = "appended_values3.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    # Fallback dummy data in case the file is missing
    sample_values = []
    for i in range(-1, 12):
        sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
    data = [str(v) for v in sample_values]

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])

# Shift block positions to start at -1
num_blocks = len(values) // 13
x = np.tile(np.arange(-1, 12), num_blocks) # Block positions are now -1, 0, 1, ..., 11
y = values[:len(x)]

# --- Compute block-wise statistics ---
x_positions = np.arange(-1, 12)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

# --- Compute ABSOLUTE deviation with a GLOBAL scale ---
# 1. Calculate the absolute distance of each point from its block's mean
mean_map = {pos: mean for pos, mean in zip(x_positions, means)}
abs_deviations = np.array([abs(yi - mean_map[xi]) for xi, yi in zip(x, y)])

# 2. Find the single maximum deviation across ALL blocks
global_max_dev = np.max(abs_deviations)

# --- Add jitter to spread points horizontally ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Create the Plot using the new design specifications ---
fig, ax = plt.subplots(figsize=(9, 5))

# --- Define DISCRETE color boundaries and normalization ---
# Create sharp color changes at intervals of 0.05
step = 0.05
boundaries = np.arange(0, global_max_dev + step, step)
# Changed to "YlOrRd" for a light-to-dark, high-contrast colormap
cmap = plt.colormaps.get("YlOrRd").resampled(len(boundaries) - 1)
norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

# 1. Scatter plot with jitter, colored by DISCRETE GLOBAL ABSOLUTE deviation
scatter = ax.scatter(
    x_jittered, y,
    c=abs_deviations, cmap=cmap, norm=norm,
    s=35, alpha=0.9,
    edgecolors="black", linewidths=0.5
)

# 2. Colorbar showing the discrete absolute scale
# Use the boundaries to create a stepped colorbar
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(mappable, ax=ax, pad=0.02, boundaries=boundaries)
cbar.set_label("Absolute Deviation from Block Mean", fontsize=15, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=15, colors="black", direction="inout")


# 3. Plot the mean line and standard deviation shading
ax.plot(
    x_positions, means,
    color="crimson", marker="o", markersize=8, linewidth=3, label="Mean"
)
ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.25, label="Standard Deviation"
)

# --- Customize Labels, Title, and Ticks ---
ax.set_xlabel("Transformer Block Position", fontsize=15, fontweight="bold", color="black")
ax.set_ylabel("Average Correlation Coefficient", fontsize=15, fontweight="bold", color="black")

# Custom y-axis ticks
ax.set_ylim(-0.2, 0.8)
ax.set_yticks(np.arange(-0.2, 1.01, 0.09))

# X-axis ticks
ax.set_xticks(x_positions)

# Ticks formatting
ax.tick_params(axis="x", which="major", labelsize=15, colors="black", direction="inout", top=False, bottom=True)
ax.tick_params(axis="y", which="major", labelsize=15, colors="black", direction="inout", left=True, right=False)

# Legend with border
legend = ax.legend(fontsize=15, frameon=True)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(1.2)

# Add BOTH horizontal and vertical grid lines
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.xaxis.grid(True, linestyle="--", alpha=0.6)

# Add border around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

plt.tight_layout()
plt.savefig("final.svg", format="svg")
plt.show()


# %% Original cell 45
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# # --- Create a dummy data file for demonstration ---
# np.random.seed(42)
# sample_values = []
# for i in range(-1, 12):
#     sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
# with open("appended_values2.txt", "w") as f:
#     f.write(" ".join(map(str, sample_values)))


# --- Read data from your file ---
filename = "appended_values3.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    # Fallback dummy data in case the file is missing
    sample_values = []
    for i in range(-1, 12):
        sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
    data = [str(v) for v in sample_values]

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])

# Shift block positions to start at -1
num_blocks = len(values) // 13
x = np.tile(np.arange(-1, 12), num_blocks) # Block positions are now -1, 0, 1, ..., 11
y = values[:len(x)]

# --- Compute block-wise statistics ---
x_positions = np.arange(-1, 12)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

# --- Compute ABSOLUTE deviation with a GLOBAL scale ---
# 1. Calculate the absolute distance of each point from its block's mean
mean_map = {pos: mean for pos, mean in zip(x_positions, means)}
abs_deviations = np.array([abs(yi - mean_map[xi]) for xi, yi in zip(x, y)])

# 2. Find the single maximum deviation across ALL blocks
global_max_dev = np.max(abs_deviations)

# --- Add jitter to spread points horizontally ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Print block-wise statistics ---
print("\n--- Block-wise Statistics ---")
for i, pos in enumerate(x_positions):
    print(f"Block {pos:2d}: Mean = {means[i]:.4f}, Std Dev = {stds[i]:.4f}")
print("---------------------------\n")


# --- Create the Plot using the new design specifications ---
fig, ax = plt.subplots(figsize=(9, 5))

# --- Define DISCRETE color boundaries and normalization ---
# Create sharp color changes at intervals of 0.05
step = 0.05
boundaries = np.arange(0, global_max_dev + step, step)
# Changed to "YlOrRd" for a light-to-dark, high-contrast colormap
cmap = plt.colormaps.get("YlOrRd").resampled(len(boundaries) - 1)
norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

# 1. Scatter plot with jitter, colored by DISCRETE GLOBAL ABSOLUTE deviation
scatter = ax.scatter(
    x_jittered, y,
    c=abs_deviations, cmap=cmap, norm=norm,
    s=35, alpha=0.9,
    edgecolors="black", linewidths=0.5
)

# 2. Colorbar showing the discrete absolute scale
# Use the boundaries to create a stepped colorbar
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(mappable, ax=ax, pad=0.02, boundaries=boundaries)
cbar.set_label("Absolute Deviation from Block Mean", fontsize=15, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=15, colors="black", direction="inout")


# 3. Plot the mean line and standard deviation shading
ax.plot(
    x_positions, means,
    color="crimson", marker="o", markersize=8, linewidth=3, label="Mean"
)
ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.25, label="Standard Deviation"
)

# --- Customize Labels, Title, and Ticks ---
ax.set_xlabel("Transformer Block Position", fontsize=15, fontweight="bold", color="black")
ax.set_ylabel("Average Correlation Coefficient", fontsize=15, fontweight="bold", color="black")

# Custom y-axis ticks
ax.set_ylim(-0.2, 0.8)
ax.set_yticks(np.arange(-0.2, 1.01, 0.09))

# X-axis ticks
ax.set_xticks(x_positions)

# Ticks formatting
ax.tick_params(axis="x", which="major", labelsize=15, colors="black", direction="inout", top=False, bottom=True)
ax.tick_params(axis="y", which="major", labelsize=15, colors="black", direction="inout", left=True, right=False)

# Legend with border
legend = ax.legend(fontsize=15, frameon=True)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(1.2)

# Add BOTH horizontal and vertical grid lines
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.xaxis.grid(True, linestyle="--", alpha=0.6)

# Add border around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

plt.tight_layout()
plt.show()


# %% Original cell 46
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- Read data from file ---
filename = "appended_values3.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    sample_values = []
    for i in range(-1, 12):
        sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
    data = [str(v) for v in sample_values]

# --- Prepare raw data for plotting ---
values = np.array([float(x) for x in data if x])
num_blocks = len(values) // 13
x = np.tile(np.arange(-1, 12), num_blocks)
y = values[:len(x)]  # Raw correlation scores

# --- Add jitter to spread points horizontally ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

x_positions = np.arange(-1, 12)

# --- Create Plot ---
fig, ax = plt.subplots(figsize=(9, 5))

# --- Define DISCRETE color boundaries and norm for RAW correlation scores ---
min_val, max_val = np.min(y), np.max(y)
step = 0.1
# Align discrete boundaries cleanly around raw data range
boundaries = np.arange(np.floor(min_val * 10) / 10, np.ceil(max_val * 10) / 10 + step, step)
cmap = plt.colormaps.get("YlOrRd").resampled(len(boundaries) - 1)
norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

# 1. Scatter plot colored directly by RAW values (y)
scatter = ax.scatter(
    x_jittered, y,
    c=y, cmap=cmap, norm=norm,
    s=35, alpha=0.9,
    edgecolors="black", linewidths=0.5
)

# 2. Colorbar reflecting raw correlation scores
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(mappable, ax=ax, pad=0.02, boundaries=boundaries)
cbar.set_label("Raw Correlation Score", fontsize=15, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=15, colors="black", direction="inout")

# --- Customize Labels, Axis Limits, and Ticks ---
ax.set_xlabel("Transformer Block Position", fontsize=15, fontweight="bold", color="black")
ax.set_ylabel("Raw Correlation Score", fontsize=15, fontweight="bold", color="black")

# Adjust y-limits based on raw data range
ax.set_ylim(-0.5, 1.5)
ax.set_yticks(np.arange(-0.5, 1.6, 0.2))

# X-axis ticks
ax.set_xticks(x_positions)

# Ticks formatting
ax.tick_params(axis="x", which="major", labelsize=15, colors="black", direction="inout", top=False, bottom=True)
ax.tick_params(axis="y", which="major", labelsize=15, colors="black", direction="inout", left=True, right=False)

# Grid lines and border
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.xaxis.grid(True, linestyle="--", alpha=0.6)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

plt.tight_layout()
plt.show()


# %% Original cell 47
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# --- Read data from file ---
filename = "appended_valueseeeeeeeeeeeeeeeeeesssssssssssssssssssssssssssssssssssss222222222222.txt"
num_blocks = 12  # Standard ViT depth (Blocks 0 to 11)
tokens_per_block = 196  # Number of patch tokens per layer (14x14)

try:
    with open(filename, "r") as f:
        data = f.read().split()
    values = np.array([float(x) for x in data if x])
except FileNotFoundError:
    print(f"Error: '{filename}' not found. Generating dummy data for {num_blocks} blocks.")
    # Dummy fallback: 196 raw values per block for 2 forward passes
    sample_values = []
    for _ in range(2):
        for i in range(num_blocks):
            sample_values.extend(np.random.normal(loc=i / 15 + 0.2, scale=0.15, size=tokens_per_block))
    values = np.array(sample_values)

# --- Compute pass alignment and X-axis mapping ---
x_positions = np.arange(0, num_blocks)
points_per_pass = num_blocks * tokens_per_block

# Truncate values to fit complete block passes
num_passes = len(values) // points_per_pass
total_valid_points = num_passes * points_per_pass
y = values[:total_valid_points]

# Create X coordinates: repeats block ID 196 times per pass
single_pass_x = np.repeat(x_positions, tokens_per_block)
x = np.tile(single_pass_x, num_passes)

# --- Compute block-wise statistics ---
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds = np.array([np.std(y[x == i]) for i in x_positions])

# --- Add jitter to spread raw patch points horizontally ---
jitter_strength = 0.2
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Print block-wise statistics ---
print("\n--- Block-wise Raw Correlation Statistics ---")
for i, pos in enumerate(x_positions):
    print(f"Block {pos:2d}: Points = {np.sum(x == pos)}, Mean = {means[i]:.4f}, Std Dev = {stds[i]:.4f}")
print("---------------------------------------------\n")

# --- Create Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

# Define discrete color boundaries based on raw score range
min_val, max_val = np.min(y), np.max(y)
step = 0.1
boundaries = np.arange(np.floor(min_val * 10) / 10, np.ceil(max_val * 10) / 10 + step, step)
cmap = plt.colormaps.get("YlOrRd").resampled(len(boundaries) - 1)
norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

# 1. Scatter plot of all raw correlation scores with jitter
scatter = ax.scatter(
    x_jittered,
    y,
    c=y,
    cmap=cmap,
    norm=norm,
    s=15,
    alpha=0.5,
    edgecolors="none",
)

# 2. Add stepped colorbar representing raw correlation values
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(mappable, ax=ax, pad=0.02, boundaries=boundaries)
cbar.set_label("Raw Correlation Score", fontsize=14, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=12, colors="black", direction="inout")

# 3. Overlay Block Mean line and Standard Deviation band
ax.plot(x_positions, means, color="crimson", marker="o", markersize=8, linewidth=3, label="Block Mean")
ax.fill_between(
    x_positions,
    means - stds,
    means + stds,
    color="crimson",
    alpha=0.25,
    label="±1 Standard Deviation",
)

# --- Customize Axes and Formatting ---
ax.set_xlabel("Transformer Block Position", fontsize=14, fontweight="bold", color="black")
ax.set_ylabel("Raw Pearson Correlation Coefficient", fontsize=14, fontweight="bold", color="black")

# Y-axis scaling around raw value limits
y_lower = np.floor(min_val * 5) / 5 - 0.1
y_upper = np.ceil(max_val * 5) / 5 + 0.1
ax.set_ylim(y_lower, y_upper)
ax.set_yticks(np.arange(y_lower, y_upper + 0.05, 0.2))

# X-axis scaling
ax.set_xticks(x_positions)
ax.set_xticklabels([f"Block {i}" for i in x_positions], rotation=0)

ax.tick_params(axis="x", which="major", labelsize=12, colors="black", direction="inout")
ax.tick_params(axis="y", which="major", labelsize=12, colors="black", direction="inout")

# Legend
legend = ax.legend(fontsize=12, frameon=True, loc="upper left")
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(1.2)

# Grid and Frame Styling
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.xaxis.grid(True, linestyle="--", alpha=0.6)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

plt.tight_layout()
plt.show()


# %% Original cell 48
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# # --- Create a dummy data file for demonstration ---
# np.random.seed(42)
# sample_values = []
# for i in range(-1, 12):
#     sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
# with open("appended_values2.txt", "w") as f:
#     f.write(" ".join(map(str, sample_values)))


# --- Read data from your file ---
filename = "appended_valueseeeeeeeeeeeeeeeeeesssssssssssssssssssssssssssssssssssss222222222222.txt"
try:
    with open(filename, "r") as f:
        data = f.read().split()
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Using dummy data.")
    # Fallback dummy data in case the file is missing
    sample_values = []
    for i in range(-1, 12):
        sample_values.extend(np.random.normal(loc=i/5 + np.sin(i), scale=0.5, size=30))
    data = [str(v) for v in sample_values]

# --- Prepare data for plotting ---
values = np.array([float(x) for x in data if x])

# Shift block positions to start at -1
num_blocks = len(values) // 13
x = np.tile(np.arange(-1, 12), num_blocks) # Block positions are now -1, 0, 1, ..., 11
y = values[:len(x)]

# --- Compute block-wise statistics ---
x_positions = np.arange(-1, 12)
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds  = np.array([np.std(y[x == i]) for i in x_positions])

# --- Compute ABSOLUTE deviation with a GLOBAL scale ---
# 1. Calculate the absolute distance of each point from its block's mean
mean_map = {pos: mean for pos, mean in zip(x_positions, means)}
abs_deviations = np.array([abs(yi - mean_map[xi]) for xi, yi in zip(x, y)])

# 2. Find the single maximum deviation across ALL blocks
global_max_dev = np.max(abs_deviations)

# --- Add jitter to spread points horizontally ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Print block-wise statistics ---
print("\n--- Block-wise Statistics ---")
for i, pos in enumerate(x_positions):
    print(f"Block {pos:2d}: Mean = {means[i]:.4f}, Std Dev = {stds[i]:.4f}")
print("---------------------------\n")


# --- Create the Plot using the new design specifications ---
fig, ax = plt.subplots(figsize=(9, 5))

# --- Define DISCRETE color boundaries and normalization ---
# Create sharp color changes at intervals of 0.05
step = 0.05
boundaries = np.arange(0, global_max_dev + step, step)
# Changed to "YlOrRd" for a light-to-dark, high-contrast colormap
cmap = plt.colormaps.get("YlOrRd").resampled(len(boundaries) - 1)
norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

# 1. Scatter plot with jitter, colored by DISCRETE GLOBAL ABSOLUTE deviation
scatter = ax.scatter(
    x_jittered, y,
    c=abs_deviations, cmap=cmap, norm=norm,
    s=35, alpha=0.9,
    edgecolors="black", linewidths=0.5
)

# 2. Colorbar showing the discrete absolute scale
# Use the boundaries to create a stepped colorbar
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(mappable, ax=ax, pad=0.02, boundaries=boundaries)
cbar.set_label("Absolute Deviation from Block Mean", fontsize=15, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=15, colors="black", direction="inout")


# 3. Plot the mean line and standard deviation shading
ax.plot(
    x_positions, means,
    color="crimson", marker="o", markersize=8, linewidth=3, label="Mean"
)
ax.fill_between(
    x_positions,
    means - stds, means + stds,
    color="crimson", alpha=0.25, label="Standard Deviation"
)

# --- Customize Labels, Title, and Ticks ---
ax.set_xlabel("Transformer Block Position", fontsize=15, fontweight="bold", color="black")
ax.set_ylabel("Average Correlation Coefficient", fontsize=15, fontweight="bold", color="black")

# Custom y-axis ticks
ax.set_ylim(-0.2, 0.8)
ax.set_yticks(np.arange(-0.2, 1.01, 0.09))

# X-axis ticks
ax.set_xticks(x_positions)

# Ticks formatting
ax.tick_params(axis="x", which="major", labelsize=15, colors="black", direction="inout", top=False, bottom=True)
ax.tick_params(axis="y", which="major", labelsize=15, colors="black", direction="inout", left=True, right=False)

# Legend with border
legend = ax.legend(fontsize=15, frameon=True)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(1.2)

# Add BOTH horizontal and vertical grid lines
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.xaxis.grid(True, linestyle="--", alpha=0.6)

# Add border around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

plt.tight_layout()
plt.show()


# %% Original cell 49
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# --- Read data from individual block files ---
x_positions = np.arange(-1, 12)  # Blocks -1, 0, 1, ..., 11
x_list = []
y_list = []

for pos in x_positions:
    filename = f"appended_values_block_{pos}.txt"
    try:
        with open(filename, "r") as f:
            data = f.read().split()
            block_values = [float(x) for x in data if x]
    except FileNotFoundError:
        print(f"Warning: File '{filename}' not found. Generating dummy data for Block {pos}.")
        # Fallback dummy data if file is missing (196 patch points)
        block_values = list(np.random.normal(loc=pos / 15 + 0.2, scale=0.15, size=196))

    y_list.extend(block_values)
    x_list.extend([pos] * len(block_values))

x = np.array(x_list)
y = np.array(y_list)

# --- Compute block-wise statistics ---
means = np.array([np.mean(y[x == i]) for i in x_positions])
stds = np.array([np.std(y[x == i]) for i in x_positions])

# --- Compute ABSOLUTE deviation with a GLOBAL scale ---
# 1. Calculate the absolute distance of each point from its block's mean
mean_map = {pos: mean for pos, mean in zip(x_positions, means)}
abs_deviations = np.array([abs(yi - mean_map[xi]) for xi, yi in zip(x, y)])

# 2. Find the single maximum deviation across ALL blocks
global_max_dev = np.max(abs_deviations)

# --- Add jitter to spread points horizontally ---
jitter_strength = 0.15
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

# --- Print block-wise statistics ---
print("\n--- Block-wise Statistics ---")
for i, pos in enumerate(x_positions):
    print(f"Block {pos:2d}: Points = {np.sum(x == pos)}, Mean = {means[i]:.4f}, Std Dev = {stds[i]:.4f}")
print("---------------------------\n")

# --- Create the Plot using exact design specifications ---
fig, ax = plt.subplots(figsize=(9, 5))

# --- Define DISCRETE color boundaries and normalization ---
step = 0.05
boundaries = np.arange(0, global_max_dev + step, step)
cmap = plt.colormaps.get("YlOrRd").resampled(len(boundaries) - 1)
norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

# 1. Scatter plot with jitter, colored by DISCRETE GLOBAL ABSOLUTE deviation
scatter = ax.scatter(
    x_jittered,
    y,
    c=abs_deviations,
    cmap=cmap,
    norm=norm,
    s=35,
    alpha=0.9,
    edgecolors="black",
    linewidths=0.5,
)

# 2. Colorbar showing the discrete absolute scale
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(mappable, ax=ax, pad=0.02, boundaries=boundaries)
cbar.set_label("Absolute Deviation from Block Mean", fontsize=15, fontweight="bold", color="black")
cbar.ax.tick_params(labelsize=15, colors="black", direction="inout")

# 3. Plot the mean line and standard deviation shading
ax.plot(x_positions, means, color="crimson", marker="o", markersize=8, linewidth=3, label="Mean")
ax.fill_between(
    x_positions,
    means - stds,
    means + stds,
    color="crimson",
    alpha=0.25,
    label="Standard Deviation",
)

# --- Customize Labels, Title, and Ticks ---
ax.set_xlabel("Transformer Layer / Block Index", fontsize=15, fontweight="bold", color="black")
ax.set_ylabel("Patch-CLS Correlation Coefficient", fontsize=15, fontweight="bold", color="black")



# Custom y-axis limits and ticks based on data range
y_min = max(-0.3, np.floor(np.min(y) * 10) / 10)
y_max = min(1.0, np.ceil(np.max(y) * 10) / 10)
ax.set_ylim(y_min, y_max)
ax.set_yticks(np.arange(y_min, y_max + 0.01, 0.1))

# X-axis ticks
ax.set_xticks(x_positions)

# Ticks formatting
ax.tick_params(axis="x", which="major", labelsize=15, colors="black", direction="inout", top=False, bottom=True)
ax.tick_params(axis="y", which="major", labelsize=15, colors="black", direction="inout", left=True, right=False)

# Legend with border
legend = ax.legend(fontsize=15, frameon=True)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(1.2)

# Add BOTH horizontal and vertical grid lines
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.xaxis.grid(True, linestyle="--", alpha=0.6)

# Add border around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

plt.tight_layout()



# --- Export Figures ---
plt.savefig("raw_patch_correlation_320dpi.png", dpi=320, bbox_inches="tight")
plt.savefig("raw_patch_correlation_1000dpi.png", dpi=1000, bbox_inches="tight")
plt.savefig("raw_patch_correlation_500dpi.png", dpi=500, bbox_inches="tight")
plt.show()


# %% Original cell 50
mins  = np.array([np.min(y[x == i]) for i in x_positions])
maxs  = np.array([np.max(y[x == i]) for i in x_positions])


# %% Original cell 51
print("\n---------------- Block-wise Statistics ----------------")
for i, pos in enumerate(x_positions):
    print(
        f"Block {pos:2d}: "
        f"Points = {np.sum(x == pos):3d}, "
        f"Min = {mins[i]:.4f}, "
        f"Max = {maxs[i]:.4f}, "
        f"Mean = {means[i]:.4f}, "
        f"Std = {stds[i]:.4f}"
    )
print("------------------------------------------------------\n")


# %% Original cell 52
block_minus1 = y[x == -1]

print(f"Block -1: Min = {np.min(block_minus1):.4f}, Max = {np.max(block_minus1):.4f}")
