# Exported from experiments/vit_base_similarity_only/patch_pruning_visualization.ipynb
# Historical code: review paths and side effects before executing.


# %% Original cell 0
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

# Sample mask tensor from CUDA device (moved to CPU for processing)
mask_tensor = torch.tensor([[False,  True, False, False,  True, False,  True,  True,  True,  True,
  True, False, False, False, False, False, False, False, False,  True,
  True,  True,  True,  True,  True, False, False, False, False, False,
 False, False,  True,  True,  True,  True, False,  True, False, False,
 False, False, False, False,  True,  True,  True,  True,  True,  True,
  True,  True, False, False,  True, False,  True, False,  True,  True,
  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
 False, False,  True,  True,  True,  True,  True,  True,  True,  True,
  True, False,  True,  True, False, False, False,  True,  True,  True,
  True,  True,  True,  True, False, False, False,  True, False, False,
 False, False,  True, False,  True,  True, False, False, False, False,
 False, False, False, False, False,  True,  True,  True,  True,  True,
  True,  True, False, False, False, False, False, False,  True,  True,
  True,  True,  True,  True,  True,  True,  True,  True, False, False,
 False, False, False, False,  True,  True,  True,  True,  True,  True,
  True, False, False, False, False, False, False, False,  True,  True,
 False,  True,  True, False, False, False, False, False, False, False,
 False, False, False, False, False,  True, False, False, False, False,
 False, False, False, False, False,  True,  True,  True,  True,  True,
  True,  True, False, False, False, False]], device='cuda:1')

# Image processing function
def apply_mask_with_white_shade(image_path, mask_tensor, opacity=0.4):
    mask = mask_tensor.to('cpu').numpy().flatten()

    # Load and resize image
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image_np = np.array(image).astype(np.float32)

    patch_size = 16
    output_image = image_np.copy()
    
    assert mask.shape[0] == 196, "Mask must have 196 elements (14x14 patches)"

    idx = 0
    for i in range(0, 224, patch_size):
        for j in range(0, 224, patch_size):
            if not mask[idx]:
                # Blend the patch with white using the specified opacity
                original_patch = output_image[i:i+patch_size, j:j+patch_size, :]
                white_patch = np.ones_like(original_patch) * 255.0
                blended_patch = (1 - opacity) * original_patch + opacity * white_patch
                output_image[i:i+patch_size, j:j+patch_size, :] = blended_patch
            idx += 1

    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    return Image.fromarray(output_image)

# ---- Example usage ----
# Replace with the path to your image (should be a color image)
image_path = "/home/viraj/my_project/val/val3/n01440764/ILSVRC2012_val_00000293.JPEG"  # <<-- CHANGE THIS to your actual image path
result_image = apply_mask_with_white_shade(image_path, mask_tensor, opacity=0.75)

# Display the result
plt.figure(figsize=(6, 6))

plt.imshow(result_image)
plt.axis('off')
plt.title("Masked Image with White Overlay")
plt.show()


# %% Original cell 1
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

# Sample mask tensor from CUDA device (moved to CPU for processing)
mask_tensor = torch.tensor([[False, False, False, False, False, False, False,  True, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False,  True, False, False,  True, False, False, False, False,
         False, False,  True, False, False,  True,  True,  True,  True,  True,
          True,  True, False, False, False, False,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True, False,  True,
          True, False,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False, False, False, False, False, False,  True,  True,  True,
          True,  True,  True,  True, False, False, False, False, False, False,
         False, False,  True,  True,  True,  True,  True,  True, False, False,
         False, False, False, False, False,  True,  True,  True,  True,  True,
          True,  True, False, False, False, False,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True, False, False, False,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False, False, False, False, False, False, False,  True,  True,
          True,  True,  True,  True,  True, False, False, False, False,  True,
         False, False, False,  True,  True,  True,  True, False, False, False,
         False, False,  True, False, False,  True,  True,  True,  True,  True,
          True,  True,  True, False,  True, False]], device='cuda:1')

# Image processing function
def apply_mask_with_white_shade(image_path, mask_tensor, opacity=0.4):
    mask = mask_tensor.to('cpu').numpy().flatten()

    # Load and resize image
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image_np = np.array(image).astype(np.float32)

    patch_size = 16
    output_image = image_np.copy()
    
    assert mask.shape[0] == 196, "Mask must have 196 elements (14x14 patches)"

    idx = 0
    for i in range(0, 224, patch_size):
        for j in range(0, 224, patch_size):
            if not mask[idx]:
                # Blend the patch with white using the specified opacity
                original_patch = output_image[i:i+patch_size, j:j+patch_size, :]
                white_patch = np.ones_like(original_patch) * 255.0
                blended_patch = (1 - opacity) * original_patch + opacity * white_patch
                output_image[i:i+patch_size, j:j+patch_size, :] = blended_patch
            idx += 1

    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    return Image.fromarray(output_image)

# ---- Example usage ----
# Replace with the path to your image (should be a color image)
image_path = "/home/viraj/my_project/val/val3/n01682714/ILSVRC2012_val_00005563.JPEG"  # <<-- CHANGE THIS to your actual image path
result_image = apply_mask_with_white_shade(image_path, mask_tensor, opacity=0.75)

# Display the result
plt.figure(figsize=(6, 6))

plt.imshow(result_image)
plt.axis('off')
plt.title("Masked Image with White Overlay")
plt.show()


# %% Original cell 2
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch

# Sample mask tensor from CUDA device (moved to CPU for processing)
mask_tensor = torch.tensor([[False, False, False, False, False, False, False,  True, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False,  True, False, False,  True, False, False, False, False,
         False, False,  True, False, False,  True,  True,  True,  True,  True,
          True,  True, False, False, False, False,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True, False,  True,
          True, False,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False, False, False, False, False, False,  True,  True,  True,
          True,  True,  True,  True, False, False, False, False, False, False,
         False, False,  True,  True,  True,  True,  True,  True, False, False,
         False, False, False, False, False,  True,  True,  True,  True,  True,
          True,  True, False, False, False, False,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True, False, False, False,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False, False, False, False, False, False, False,  True,  True,
          True,  True,  True,  True,  True, False, False, False, False,  True,
         False, False, False,  True,  True,  True,  True, False, False, False,
         False, False,  True, False, False,  True,  True,  True,  True,  True,
          True,  True,  True, False,  True, False]], device='cuda:1')  # Replace [...] with your actual mask data

# Image processing function
def apply_mask_with_white_shade(image_path, mask_tensor, opacity=0.4, border_size=10):
    mask = mask_tensor.to('cpu').numpy().flatten()

    # Load and resize image
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image_np = np.array(image).astype(np.float32)

    patch_size = 16
    output_image = image_np.copy()
    
    assert mask.shape[0] == 196, "Mask must have 196 elements (14x14 patches)"

    idx = 0
    for i in range(0, 224, patch_size):
        for j in range(0, 224, patch_size):
            if not mask[idx]:
                # Blend the patch with white using the specified opacity
                original_patch = output_image[i:i+patch_size, j:j+patch_size, :]
                white_patch = np.ones_like(original_patch) * 255.0
                blended_patch = (1 - opacity) * original_patch + opacity * white_patch
                output_image[i:i+patch_size, j:j+patch_size, :] = blended_patch
            idx += 1

    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    result_image = Image.fromarray(output_image)

    # Add black border
    result_image_with_border = ImageOps.expand(result_image, border=border_size, fill='black')
    return result_image_with_border

# ---- Example usage ----
image_path = "/home/viraj/my_project/val/val3/n01682714/ILSVRC2012_val_00005563.JPEG"  # Update with your actual image path
result_image = apply_mask_with_white_shade(image_path, mask_tensor, opacity=0.75, border_size=2)

# Display the result
plt.figure(figsize=(6, 6))
plt.imshow(result_image)
plt.axis('off')
plt.title("Masked Image with White Overlay and Black Border")
plt.show()
