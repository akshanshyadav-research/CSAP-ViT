# Exported from experiments/vit_tiny_results/drop_1_50_vit_tiny/Untitled.ipynb
# Historical code: review paths and side effects before executing.


# %% Original cell 0
import nbformat
import os

# Path to your original notebook
original_notebook = "vit_small_only_similarity_atfirst.ipynb"
output_folder = "notebooks1"
os.makedirs(output_folder, exist_ok=True)

# Load the original notebook
with open(original_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Generate 50 copies with drop_rate from 0.01 to 0.50
for i in range(0, 9):
    drop_rate_value = i / 100.0  # 0.01 to 0.50
    new_nb = nb.copy()

    if new_nb['cells'][0]['cell_type'] == 'code':
        modified_lines = []
        for line in new_nb['cells'][0]['source'].splitlines():
            if line.strip().startswith("drop_rate="):
                modified_lines.append(f"drop_rate={drop_rate_value:.2f}")
            else:
                modified_lines.append(line)
        new_nb['cells'][0]['source'] = "\n".join(modified_lines)

    # Save the modified notebook
    new_notebook = f"{output_folder}/notebook_{i:02d}.ipynb"
    with open(new_notebook, "w") as f:
        nbformat.write(new_nb, f)

    print(f"Created {new_notebook} with drop_rate={drop_rate_value:.2f}")


# %% Original cell 1
import os
from concurrent.futures import ThreadPoolExecutor
import papermill as pm

# Folder containing the notebooks
notebook_folder = "notebooks1"
output_folder = "output1"
os.makedirs(output_folder, exist_ok=True)

# List all notebook files
notebooks = [f"{notebook_folder}/{nb}" for nb in os.listdir(notebook_folder) if nb.endswith(".ipynb")]

# Function to run each notebook
def run_notebook(notebook):
    output_nb = f"{output_folder}/{os.path.basename(notebook)}"
    try:
        pm.execute_notebook(
            notebook,
            output_nb
        )
        print(f"Executed {notebook}, output saved to {output_nb}")
    except Exception as e:
        print(f"Failed: {notebook} with error {e}")

# Parallel execution
num_workers = min(300, os.cpu_count())  # Use available CPU cores or 100 threads
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(run_notebook, notebooks)


# %% Original cell 2
import nbformat
import os

# Path to your original notebook
original_notebook = "vit_small_only_similarity_atfirst.ipynb"
output_folder = "notebooks2"
os.makedirs(output_folder, exist_ok=True)

# Load the original notebook
with open(original_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Generate 50 copies with drop_rate from 0.01 to 0.50
for i in range(9, 18):
    drop_rate_value = i / 100.0  # 0.01 to 0.50
    new_nb = nb.copy()

    if new_nb['cells'][0]['cell_type'] == 'code':
        modified_lines = []
        for line in new_nb['cells'][0]['source'].splitlines():
            if line.strip().startswith("drop_rate="):
                modified_lines.append(f"drop_rate={drop_rate_value:.2f}")
            else:
                modified_lines.append(line)
        new_nb['cells'][0]['source'] = "\n".join(modified_lines)

    # Save the modified notebook
    new_notebook = f"{output_folder}/notebook_{i:02d}.ipynb"
    with open(new_notebook, "w") as f:
        nbformat.write(new_nb, f)

    print(f"Created {new_notebook} with drop_rate={drop_rate_value:.2f}")


# %% Original cell 3
import os
from concurrent.futures import ThreadPoolExecutor
import papermill as pm

# Folder containing the notebooks
notebook_folder = "notebooks2"
output_folder = "output2"
os.makedirs(output_folder, exist_ok=True)

# List all notebook files
notebooks = [f"{notebook_folder}/{nb}" for nb in os.listdir(notebook_folder) if nb.endswith(".ipynb")]

# Function to run each notebook
def run_notebook(notebook):
    output_nb = f"{output_folder}/{os.path.basename(notebook)}"
    try:
        pm.execute_notebook(
            notebook,
            output_nb
        )
        print(f"Executed {notebook}, output saved to {output_nb}")
    except Exception as e:
        print(f"Failed: {notebook} with error {e}")

# Parallel execution
num_workers = min(300, os.cpu_count())  # Use available CPU cores or 100 threads
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(run_notebook, notebooks)


# %% Original cell 4
import nbformat
import os

# Path to your original notebook
original_notebook = "vit_small_only_similarity_atfirst.ipynb"
output_folder = "notebooks3"
os.makedirs(output_folder, exist_ok=True)

# Load the original notebook
with open(original_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Generate 50 copies with drop_rate from 0.01 to 0.50
for i in range(18, 27):
    drop_rate_value = i / 100.0  # 0.01 to 0.50
    new_nb = nb.copy()

    if new_nb['cells'][0]['cell_type'] == 'code':
        modified_lines = []
        for line in new_nb['cells'][0]['source'].splitlines():
            if line.strip().startswith("drop_rate="):
                modified_lines.append(f"drop_rate={drop_rate_value:.2f}")
            else:
                modified_lines.append(line)
        new_nb['cells'][0]['source'] = "\n".join(modified_lines)

    # Save the modified notebook
    new_notebook = f"{output_folder}/notebook_{i:02d}.ipynb"
    with open(new_notebook, "w") as f:
        nbformat.write(new_nb, f)

    print(f"Created {new_notebook} with drop_rate={drop_rate_value:.2f}")


# %% Original cell 5
import os
from concurrent.futures import ThreadPoolExecutor
import papermill as pm

# Folder containing the notebooks
notebook_folder = "notebooks3"
output_folder = "output3"
os.makedirs(output_folder, exist_ok=True)

# List all notebook files
notebooks = [f"{notebook_folder}/{nb}" for nb in os.listdir(notebook_folder) if nb.endswith(".ipynb")]

# Function to run each notebook
def run_notebook(notebook):
    output_nb = f"{output_folder}/{os.path.basename(notebook)}"
    try:
        pm.execute_notebook(
            notebook,
            output_nb
        )
        print(f"Executed {notebook}, output saved to {output_nb}")
    except Exception as e:
        print(f"Failed: {notebook} with error {e}")

# Parallel execution
num_workers = min(300, os.cpu_count())  # Use available CPU cores or 100 threads
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(run_notebook, notebooks)


# %% Original cell 6
import nbformat
import os

# Path to your original notebook
original_notebook = "vit_small_only_similarity_atfirst.ipynb"
output_folder = "notebooks4"
os.makedirs(output_folder, exist_ok=True)

# Load the original notebook
with open(original_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Generate 50 copies with drop_rate from 0.01 to 0.50
for i in range(27, 36):
    drop_rate_value = i / 100.0  # 0.01 to 0.50
    new_nb = nb.copy()

    if new_nb['cells'][0]['cell_type'] == 'code':
        modified_lines = []
        for line in new_nb['cells'][0]['source'].splitlines():
            if line.strip().startswith("drop_rate="):
                modified_lines.append(f"drop_rate={drop_rate_value:.2f}")
            else:
                modified_lines.append(line)
        new_nb['cells'][0]['source'] = "\n".join(modified_lines)

    # Save the modified notebook
    new_notebook = f"{output_folder}/notebook_{i:02d}.ipynb"
    with open(new_notebook, "w") as f:
        nbformat.write(new_nb, f)

    print(f"Created {new_notebook} with drop_rate={drop_rate_value:.2f}")


# %% Original cell 7
import os
from concurrent.futures import ThreadPoolExecutor
import papermill as pm

# Folder containing the notebooks
notebook_folder = "notebooks4"
output_folder = "output4"
os.makedirs(output_folder, exist_ok=True)

# List all notebook files
notebooks = [f"{notebook_folder}/{nb}" for nb in os.listdir(notebook_folder) if nb.endswith(".ipynb")]

# Function to run each notebook
def run_notebook(notebook):
    output_nb = f"{output_folder}/{os.path.basename(notebook)}"
    try:
        pm.execute_notebook(
            notebook,
            output_nb
        )
        print(f"Executed {notebook}, output saved to {output_nb}")
    except Exception as e:
        print(f"Failed: {notebook} with error {e}")

# Parallel execution
num_workers = min(300, os.cpu_count())  # Use available CPU cores or 100 threads
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(run_notebook, notebooks)


# %% Original cell 8
import nbformat
import os

# Path to your original notebook
original_notebook = "vit_small_only_similarity_atfirst.ipynb"
output_folder = "notebooks5"
os.makedirs(output_folder, exist_ok=True)

# Load the original notebook
with open(original_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Generate 50 copies with drop_rate from 0.01 to 0.50
for i in range(36, 45):
    drop_rate_value = i / 100.0  # 0.01 to 0.50
    new_nb = nb.copy()

    if new_nb['cells'][0]['cell_type'] == 'code':
        modified_lines = []
        for line in new_nb['cells'][0]['source'].splitlines():
            if line.strip().startswith("drop_rate="):
                modified_lines.append(f"drop_rate={drop_rate_value:.2f}")
            else:
                modified_lines.append(line)
        new_nb['cells'][0]['source'] = "\n".join(modified_lines)

    # Save the modified notebook
    new_notebook = f"{output_folder}/notebook_{i:02d}.ipynb"
    with open(new_notebook, "w") as f:
        nbformat.write(new_nb, f)

    print(f"Created {new_notebook} with drop_rate={drop_rate_value:.2f}")


# %% Original cell 9
import os
from concurrent.futures import ThreadPoolExecutor
import papermill as pm

# Folder containing the notebooks
notebook_folder = "notebooks5"
output_folder = "output5"
os.makedirs(output_folder, exist_ok=True)

# List all notebook files
notebooks = [f"{notebook_folder}/{nb}" for nb in os.listdir(notebook_folder) if nb.endswith(".ipynb")]

# Function to run each notebook
def run_notebook(notebook):
    output_nb = f"{output_folder}/{os.path.basename(notebook)}"
    try:
        pm.execute_notebook(
            notebook,
            output_nb
        )
        print(f"Executed {notebook}, output saved to {output_nb}")
    except Exception as e:
        print(f"Failed: {notebook} with error {e}")

# Parallel execution
num_workers = min(300, os.cpu_count())  # Use available CPU cores or 100 threads
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(run_notebook, notebooks)


# %% Original cell 10
import nbformat
import os

# Path to your original notebook
original_notebook = "vit_small_only_similarity_atfirst.ipynb"
output_folder = "notebooks6"
os.makedirs(output_folder, exist_ok=True)

# Load the original notebook
with open(original_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Generate 50 copies with drop_rate from 0.01 to 0.50
for i in range(45, 54):
    drop_rate_value = i / 100.0  # 0.01 to 0.50
    new_nb = nb.copy()

    if new_nb['cells'][0]['cell_type'] == 'code':
        modified_lines = []
        for line in new_nb['cells'][0]['source'].splitlines():
            if line.strip().startswith("drop_rate="):
                modified_lines.append(f"drop_rate={drop_rate_value:.2f}")
            else:
                modified_lines.append(line)
        new_nb['cells'][0]['source'] = "\n".join(modified_lines)

    # Save the modified notebook
    new_notebook = f"{output_folder}/notebook_{i:02d}.ipynb"
    with open(new_notebook, "w") as f:
        nbformat.write(new_nb, f)

    print(f"Created {new_notebook} with drop_rate={drop_rate_value:.2f}")


# %% Original cell 11
import os
from concurrent.futures import ThreadPoolExecutor
import papermill as pm

# Folder containing the notebooks
notebook_folder = "notebooks6"
output_folder = "output6"
os.makedirs(output_folder, exist_ok=True)

# List all notebook files
notebooks = [f"{notebook_folder}/{nb}" for nb in os.listdir(notebook_folder) if nb.endswith(".ipynb")]

# Function to run each notebook
def run_notebook(notebook):
    output_nb = f"{output_folder}/{os.path.basename(notebook)}"
    try:
        pm.execute_notebook(
            notebook,
            output_nb
        )
        print(f"Executed {notebook}, output saved to {output_nb}")
    except Exception as e:
        print(f"Failed: {notebook} with error {e}")

# Parallel execution
num_workers = min(300, os.cpu_count())  # Use available CPU cores or 100 threads
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(run_notebook, notebooks)
