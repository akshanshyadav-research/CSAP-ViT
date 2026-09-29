# Exported from experiments/replacing_cls_with_dissimar_token/Untitled.ipynb
# Historical code: review paths and side effects before executing.


# %% Original cell 0
# ✅ File path
file_path = "dynamic_thresholds.txt"

# ✅ Read and process the file
middle_values = []

with open(file_path, "r") as f:
    for line in f:
        if "Embedding Size:" in line:
            # Extract the middle value from the line
            parts = line.strip().split('[')[-1].split(']')[0].split(',')
            middle_value = int(parts[1].strip())  # Extract the middle value
            middle_values.append(middle_value)

# ✅ Calculate the average
if middle_values:
    average = sum(middle_values) / len(middle_values)
    print(f"Average of Middle Values: {average:.2f}")
else:
    print("No valid lines found.")
