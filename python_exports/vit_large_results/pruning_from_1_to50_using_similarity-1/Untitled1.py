# Exported from experiments/vit_large_results/pruning_from_1_to50_using_similarity-1/Untitled1.ipynb
# Historical code: review paths and side effects before executing.


# %% Original cell 0
import os
import re
input_folder = "results/"  # Update with your folder path if needed
output_file = "filtered_accuracies.txt"

# Regex patterns
filename_drop_pattern = re.compile(r'accuracy_drop([0-9.]+)percent')
line_pattern = re.compile(r'Accuracy:\s*([0-9.]+).*processed_image:\s*10000')

results = []

for filename in os.listdir(input_folder):
    if filename.endswith(".txt") and "accuracy_drop" in filename:
        file_path = os.path.join(input_folder, filename)

        # Extract drop value from filename
        drop_match = filename_drop_pattern.search(filename)
        if not drop_match:
            continue
        drop_value = float(drop_match.group(1))

        valid_accuracies = []

        with open(file_path, "r") as f:
            for line in f:
                match = line_pattern.search(line)
                if match:
                    accuracy = float(match.group(1))
                    valid_accuracies.append(accuracy)

        if valid_accuracies:
            results.append((filename, drop_value, valid_accuracies))

# Write output
with open(output_file, "w") as f:
    for filename, drop, accuracies in results:
        f.write(f"File: {filename}, Drop: {drop}, Accuracies: {accuracies}\n")

print(f"Filtered accuracies written to {output_file}")
