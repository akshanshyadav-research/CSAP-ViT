# Exported from experiments/vit_small_results/results_merging.ipynb
# Historical code: review paths and side effects before executing.


# %% Original cell 0
import os
import re

# Folder containing the result files
folder_path = "vit_small_results"
output_file = "summary_results.txt"

# Open the output file for writing
with open(output_file, "w") as out_f:
    out_f.write("Filename\tDrop\tAccuracy\n")
    
    # Iterate through all txt files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, "r") as f:
                for line in f:
                    if "processed_image: 10000" in line:
                        # Extract drop from filename using regex
                        drop_match = re.search(r'drop(\d+(?:\.\d+)?)percent', filename)
                        drop = drop_match.group(1) if drop_match else "N/A"
                        
                        # Extract accuracy from the line
                        acc_match = re.search(r'Accuracy:\s*([\d.]+)', line)
                        accuracy = acc_match.group(1) if acc_match else "N/A"
                        
                        # Write to output file
                        out_f.write(f"{filename}\t{drop}\t{accuracy}\n")
                        break  # only the first relevant line is needed
