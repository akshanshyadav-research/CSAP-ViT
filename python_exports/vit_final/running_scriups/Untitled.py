# Exported from experiments/vit_final/running_scriups/Untitled.ipynb
# Historical code: review paths and side effects before executing.


# %% Original cell 0
import os

folder_path = "vit_base_results_similarity"  # Replace with the path to your folder

# Count .ipynb files
ipynb_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
print(f"Number of .ipynb files: {len(ipynb_files)}")


# %% Original cell 1
import os

folder_path = "vit_base_results_similarity"  # Replace with the path to your folder

# Loop through all text files
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1].strip()  # Get the last line
                parts = last_line.split(", ")  # Split by commas
                
                # Extract accuracy if processed_image is 10000
                processed_image = int(parts[3].split(":")[1])
                if processed_image == 10000:
                    accuracy = float(parts[2].split(":")[1])
                    print(f"File: {filename}, Accuracy: {accuracy}")


# %% Original cell 2
import os
import re

folder_path = "vit_base_results_similarity"  # Update with the correct folder path

# Loop through all text files
for filename in os.listdir(folder_path):
    match = re.search(r"(\d+)", filename)  # Extract number from filename
    if match and filename.endswith(".txt"):
        file_number = int(match.group(1))  # Convert to integer
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r") as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1].strip()  # Get the last line
                
                # Ensure the line contains expected keywords
                if "processed_image" in last_line and "Accuracy" in last_line:
                    parts = {k.strip(): v.strip() for k, v in (item.split(":") for item in last_line.split(", ") if ":" in item)}
                    
                    # Extract accuracy if processed_image is 10000
                    if int(parts.get("processed_image", 0)) == 10000:
                        accuracy = float(parts.get("Accuracy", 0))
                        print(f"File {file_number}: Accuracy = {accuracy}")


# %% Original cell 3
import os
import re

folder_path = "vit_base_results_similarity"  # Update with the correct folder path

# Loop through all text files
for filename in os.listdir(folder_path):
    match = re.search(r"(\d+)", filename)  # Extract number from filename
    if match and filename.endswith(".txt"):
        file_number = int(match.group(1))  # Convert to integer
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r") as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1].strip()  # Get the last line

                # Ensure the line contains expected keywords
                if "processed_image" in last_line and "Accuracy" in last_line:
                    parts = {}
                    for item in last_line.split(", "):
                        if ":" in item:
                            key, value = item.split(":", 1)  # Only split at the first ":"
                            parts[key.strip()] = value.strip()
                    
                    # Extract accuracy if processed_image is 10000
                    if parts.get("processed_image") and int(parts["processed_image"]) == 10000:
                        accuracy = float(parts["Accuracy"])
                        print(f"File {file_number}: Accuracy = {accuracy}")


# %% Original cell 4

