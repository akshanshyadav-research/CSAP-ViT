# Exported from experiments/visualization_similarity/visualization_of_all_the three_techniques/technique_1_only_attention/Untitled.ipynb
# Historical code: review paths and side effects before executing.


# %% Original cell 0
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# ----------------------------
# 1. Load and Resize Image
# ----------------------------
image_path = '/home/viraj/my_project/val/val5/vallll/ILSVRC2012_val_00006306.JPEG'  # << Replace this with your actual image
image = Image.open(image_path).convert('RGB')
image = image.resize((224, 224))
image_np = np.array(image)

# ----------------------------
# 2. Define Your Mask Tensor
# ----------------------------
mask = torch.tensor([[ True, False, False,  True,  True,  True,  True,  True,  True,  True,
         False, False, False, False, False, False, False, False, False,  True,
          True,  True,  True,  True, False, False, False, False, False, False,
         False,  True,  True,  True,  True,  True,  True,  True, False, False,
         False, False,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True, False,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True, False, False, False,  True,  True,  True,
          True,  True,  True,  True,  True,  True, False, False, False, False,
          True,  True,  True,  True,  True,  True,  True,  True, False, False,
         False, False, False, False,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True, False, False,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True, False, False,
          True, False,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True, False, False, False, False,  True,  True,  True,
          True,  True,  True,  True, False,  True, False, False, False, False,
         False, False, False,  True,  True,  True,  True,  True, False, False,
         False, False,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True, False, False,  True]], device='cuda:1')
# ----------------------------
# 3. Move to CPU and Count
# ----------------------------
mask_cpu = mask.view(-1).cpu()
num_true = mask_cpu.sum().item()
num_false = (~mask_cpu).sum().item()
print(f"True patches: {num_true}")
print(f"False patches: {num_false}")

# ----------------------------
# 4. Visualize the Masked Image
# ----------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image_np)

patch_size = 16
rows, cols = 14, 14  # Since 224 / 16 = 14
mask_np = mask_cpu.view(rows, cols).numpy()

# Add white overlay on patches where mask is False
for i in range(rows):
    for j in range(cols):
        if not mask_np[i, j]:
            rect = patches.Rectangle(
                (j * patch_size, i * patch_size),
                patch_size, patch_size,
                linewidth=0,
                edgecolor=None,
                facecolor='white',
                alpha=0.5  # semi-transparent
            )
            ax.add_patch(rect)

ax.axis('off')
plt.tight_layout()
plt.show()


# %% Original cell 1
import torch

# Step 1: Initial mask of length 196
mask1 = torch.tensor([[False, False, False,  True,  True,  True, False,  True,  True,  True,
         False, False, False, False, False, False, False, False, False,  True,
          True,  True,  True, False, False, False, False, False, False, False,
         False,  True,  True,  True,  True,  True,  True,  True, False, False,
         False, False,  True, False,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True, False,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True, False,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True, False,  True, False, False, False,  True,  True,  True,
          True,  True,  True,  True,  True,  True, False, False, False, False,
         False,  True,  True,  True,  True,  True,  True,  True, False, False,
         False, False, False, False,  True,  True,  True,  True,  True,  True,
          True,  True,  True, False, False, False,  True, False,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True, False, False,
          True, False,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False, False, False, False, False, False, False,  True,  True,
          True,  True,  True,  True, False, False, False, False, False, False,
         False, False, False, False, False,  True,  True, False, False, False,
         False, False,  True, False, False,  True,  True,  True,  True,  True,
          True,  True,  True, False, False, False]], device='cuda:1')

# Step 2: Shorter mask of only the True positions in mask1
mask2 = torch.tensor([[ True,  True,  True, False, False, False, False,  True, False, False,
         False, False, False,  True,  True, False, False,  True, False,  True,
          True, False, False, False, False,  True,  True, False, False, False,
         False, False, False, False, False, False, False,  True,  True,  True,
          True,  True, False, False, False, False, False, False, False, False,
          True,  True,  True,  True, False, False, False, False,  True,  True,
          True,  True,  True,  True, False, False, False, False,  True,  True,
          True,  True, False, False, False, False, False, False, False,  True,
         False,  True,  True,  True,  True, False, False, False,  True, False,
         False, False, False,  True,  True,  True, False, False, False, False,
          True, False, False, False, False, False, False, False, False,  True,
          True, False,  True,  True, False,  True,  True,  True]],
       device='cuda:1')


# mask1: original mask (length 196)
# mask2: filtered mask, shorter, only for True positions in mask1

def update_mask(mask1, mask2):
    clone_mask = mask1.clone()
    j = 0  # index for mask2

    for i in range(mask1.size(1)):
        if mask1[0, i]:  # only consider positions that were originally True
            if not mask2[0, j]:  # if mask2 says to remove it
                clone_mask[0, i] = False
            j += 1  # move in mask2 only when mask1 is True

    return clone_mask

# Example usage
final_mask = update_mask(mask1, mask2)
print(final_mask)



# %% Original cell 2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# ----------------------------
# 1. Load and Resize Image
# ----------------------------
image_path = '/home/viraj/my_project/val/val5/vallll/ILSVRC2012_val_00006306.JPEG'  # << Replace this with your actual image
image = Image.open(image_path).convert('RGB')
image = image.resize((224, 224))
image_np = np.array(image)

# ----------------------------
# 2. Define Your Mask Tensor
# ----------------------------
mask = torch.tensor([[False, False, False,  True,  True,  True, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
          True, False, False, False, False, False, False, False, False, False,
         False, False, False, False,  True,  True, False, False, False, False,
         False, False,  True, False, False,  True,  True, False, False, False,
         False,  True,  True, False, False, False, False, False, False, False,
         False, False, False, False,  True,  True,  True,  True, False,  True,
         False, False, False, False, False, False, False, False,  True,  True,
          True,  True, False, False, False, False, False, False, False, False,
          True,  True,  True,  True,  True,  True, False, False, False, False,
         False, False, False, False, False,  True,  True,  True, False, False,
         False, False, False, False,  True, False, False, False, False, False,
         False, False,  True, False, False, False, False, False,  True,  True,
          True,  True, False, False, False,  True, False, False, False, False,
         False, False, False,  True,  True,  True, False, False, False, False,
          True, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False,  True, False, False,  True, False,  True,  True, False,
          True,  True,  True, False, False, False]], device='cuda:1')
# ----------------------------
# 3. Move to CPU and Count
# ----------------------------
mask_cpu = mask.view(-1).cpu()
num_true = mask_cpu.sum().item()
num_false = (~mask_cpu).sum().item()
print(f"True patches: {num_true}")
print(f"False patches: {num_false}")

# ----------------------------
# 4. Visualize the Masked Image
# ----------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image_np)

patch_size = 16
rows, cols = 14, 14  # Since 224 / 16 = 14
mask_np = mask_cpu.view(rows, cols).numpy()

# Add white overlay on patches where mask is False
for i in range(rows):
    for j in range(cols):
        if not mask_np[i, j]:
            rect = patches.Rectangle(
                (j * patch_size, i * patch_size),
                patch_size, patch_size,
                linewidth=0,
                edgecolor=None,
                facecolor='white',
                alpha=0.5  # semi-transparent
            )
            ax.add_patch(rect)

ax.axis('off')
plt.tight_layout()
plt.show()
