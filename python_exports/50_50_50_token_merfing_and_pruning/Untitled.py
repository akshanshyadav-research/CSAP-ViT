# Exported from experiments/50_50_50_token_merfing_and_pruning/Untitled.ipynb
# Historical code: review paths and side effects before executing.


# %% Original cell 0
def process_embedding_drops(file_path):
    total = 0
    count = 0

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("embeddings_drops:"):
                try:
                    value = int(line.strip().split(":")[1])
                    total += value
                    count += 1
                except ValueError:
                    continue  # In case the value isn't an integer

    if count == 0:
        print("No valid entries found.")
    else:
        average = total / count
        print(f"Total Entries: {count}")
        print(f"Average embeddings_drops: {average:.2f}")

# Example usage:
process_embedding_drops("merging_completely_based_on_similarity_combineing_3_techniques_drop")
