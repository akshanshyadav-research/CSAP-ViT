# Exported from experiments/vit_large_results/drop_similarity_1_24/Untitled.ipynb
# Historical code: review paths and side effects before executing.


# %% Original cell 0
import nbformat
import os

# Path to your original notebook
original_notebook = "notebook.ipynb"
output_folder = "notebooks1"
os.makedirs(output_folder, exist_ok=True)

# Load the original notebook
with open(original_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Generate 12 copies with patch_drop_block_number values from 0 to 11
for i in range(0, 8):  # Values from 0 to 11
    new_nb = nb.copy()

    # Modify all occurrences of patch_drop_block_number in the first cell
    if new_nb['cells'][0]['cell_type'] == 'code':
        # Ensure consistent replacement in case [0] is missing or different
        new_nb['cells'][0]['source'] = "\n".join(
            [
                line if "patch_drop_block_number" not in line 
                else f"patch_drop_block_number=[{i}]"
                for line in new_nb['cells'][0]['source'].splitlines()
            ]
        )

    # Save the modified notebook
    new_notebook = f"{output_folder}/notebook_{i}.ipynb"
    with open(new_notebook, "w") as f:
        nbformat.write(new_nb, f)
    
    print(f"Created {new_notebook} with patch_drop_block_number=[{i}]")


# %% Original cell 1
# !pip install papermill


# %% Original cell 2
import os
from concurrent.futures import ThreadPoolExecutor
import papermill as pm

# Folder containing the notebooks
notebook_folder = "notebooks1"
output_folder = "output"
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
num_workers = min(100, os.cpu_count())  # Use available CPU cores or 100 threads
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(run_notebook, notebooks)


# %% Original cell 3
import nbformat
import os

# Path to your original notebook
original_notebook = "notebook.ipynb"
output_folder = "notebooks2"
os.makedirs(output_folder, exist_ok=True)

# Load the original notebook
with open(original_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Generate 12 copies with patch_drop_block_number values from 0 to 11
for i in range(8, 16):  # Values from 0 to 11
    new_nb = nb.copy()

    # Modify all occurrences of patch_drop_block_number in the first cell
    if new_nb['cells'][0]['cell_type'] == 'code':
        # Ensure consistent replacement in case [0] is missing or different
        new_nb['cells'][0]['source'] = "\n".join(
            [
                line if "patch_drop_block_number" not in line 
                else f"patch_drop_block_number=[{i}]"
                for line in new_nb['cells'][0]['source'].splitlines()
            ]
        )

    # Save the modified notebook
    new_notebook = f"{output_folder}/notebook_{i}.ipynb"
    with open(new_notebook, "w") as f:
        nbformat.write(new_nb, f)
    
    print(f"Created {new_notebook} with patch_drop_block_number=[{i}]")


# %% Original cell 4
import os
from concurrent.futures import ThreadPoolExecutor
import papermill as pm

# Folder containing the notebooks
notebook_folder = "notebooks2"
output_folder = "output"
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
num_workers = min(100, os.cpu_count())  # Use available CPU cores or 100 threads
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(run_notebook, notebooks)


# %% Original cell 5
import nbformat
import os

# Path to your original notebook
original_notebook = "notebook.ipynb"
output_folder = "notebooks3"
os.makedirs(output_folder, exist_ok=True)

# Load the original notebook
with open(original_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Generate 12 copies with patch_drop_block_number values from 0 to 11
for i in range(16, 24):  # Values from 0 to 11
    new_nb = nb.copy()

    # Modify all occurrences of patch_drop_block_number in the first cell
    if new_nb['cells'][0]['cell_type'] == 'code':
        # Ensure consistent replacement in case [0] is missing or different
        new_nb['cells'][0]['source'] = "\n".join(
            [
                line if "patch_drop_block_number" not in line 
                else f"patch_drop_block_number=[{i}]"
                for line in new_nb['cells'][0]['source'].splitlines()
            ]
        )

    # Save the modified notebook
    new_notebook = f"{output_folder}/notebook_{i}.ipynb"
    with open(new_notebook, "w") as f:
        nbformat.write(new_nb, f)
    
    print(f"Created {new_notebook} with patch_drop_block_number=[{i}]")


# %% Original cell 6
import os
from concurrent.futures import ThreadPoolExecutor
import papermill as pm

# Folder containing the notebooks
notebook_folder = "notebooks3"
output_folder = "output"
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
num_workers = min(100, os.cpu_count())  # Use available CPU cores or 100 threads
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(run_notebook, notebooks)
