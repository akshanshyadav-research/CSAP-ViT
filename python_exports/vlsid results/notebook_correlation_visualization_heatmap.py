# Exported from experiments/vlsid results/notebook_correlation_visualization_heatmap.ipynb
# Historical code: review paths and side effects before executing.


# %% Original cell 0
drop_rate_layer = 0.00
drop_rate= 0.000


# %% Original cell 1
import matplotlib.pyplot as plt
from time import time
import numpy as np
import os
import cv2
import json
import torch
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
import torch
import torch.nn as nn

import numpy as np
import os
import re
import timm


# %% Original cell 2
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
img_folder = "/home/viraj/my_project/val/val2/"
txt_path=f"similarity_score_with_cls_token_accuracy_drop_{drop_rate_layer}_percent.txt"
json_file = "/home/viraj/my_project/imagenet_class_index.json"
# drop_rate=average_percent_dro_embedding/100
# drop_rate_layers=0.60


# %% Original cell 3
# pretrained_vit = vit_large_patch16_224(pretrained=True).to(device)

pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)


# %% Original cell 4
# def plot(image):
#     plt.figure(figsize=(8, 8))
#     # plt.subplot(1, 2, 1)
#     plt.imshow(image)
#     plt.show()

def plot(image):
    # Convert to numpy in case it's a PIL Image
    img = np.array(image)

    # Add a 1-pixel black border
    img_with_border = np.pad(img, ((1,1),(1,1),(0,0)), mode='constant', constant_values=0)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_with_border)
    plt.axis("off")   # removes x and y axis values
    plt.show()


# %% Original cell 5
def copy_image(img):
    """
    Create a copy of the input image without using any libraries.

    Args:
    - img: Input image (list of lists representing pixel values)

    Returns:
    - img_copy: A copy of the input image
    """
    # Determine the dimensions of the image
    height = len(img)
    width = len(img[0])

    # Create an empty list to store the copied image
    img_copy = []

    # Iterate over each row of the image
    for i in range(height):
        # Create a copy of the current row
        row_copy = img[i][:]  # or row_copy = list(img[i]) for deep copy
        # Append the copied row to the copied image
        img_copy.append(row_copy)

    return img_copy


# %% Original cell 6
def bgr_to_rgb(img):
    """
    Convert an image from BGR to RGB color space.

    Args:
    - img: Input image in BGR format (numpy array)

    Returns:
    - img_rgb: Image converted to RGB color space
    """
    # Reverse the order of color channels
    img_rgb = img[:, :, ::-1]

    return img_rgb


# %% Original cell 7
def conv2d(img, kernel):
    """
    Perform a 2D convolution on an image using the provided kernel.

    Parameters:
    img (numpy.ndarray): Input image, a 2D array.
    kernel (numpy.ndarray): Convolution kernel, a 2D array.

    Returns:
    numpy.ndarray: The result of the convolution, with the same size as the input image.
    """
    # Flip the kernel (needed for convolution)
    kernel = np.flipud(np.fliplr(kernel))
    
    # Determine the padding size
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    
    # Pad the image with zeros on all sides
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Initialize the output image
    output = np.zeros_like(img)
    
    # Perform the convolution
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Extract the region of interest
            region = padded_img[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            # Perform element-wise multiplication and sum the result
            output[i, j] = np.sum(region * kernel)
    
    return output


# %% Original cell 8
def resize_nearest_neighbor(image, size):
    original_height = len(image)
    original_width = len(image[0])
    new_width, new_height = size

    if isinstance(image[0][0], list) or isinstance(image[0][0], np.ndarray):
        # For color images
        resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                # Calculate the nearest pixel in the original image
                x = int(j * original_width / new_width)
                y = int(i * original_height / new_height)
                resized_image[i, j] = image[y][x]
    else:
        # For grayscale images
        resized_image = np.zeros((new_height, new_width), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                # Calculate the nearest pixel in the original image
                x = int(j * original_width / new_width)
                y = int(i * original_height / new_height)
                resized_image[i, j] = image[y][x]
    
    return np.array(resized_image)


# %% Original cell 9
def save_average_percentage(file_path, average_percent, accuracy, processed_image, w):
    with open(file_path, 'a') as file:
        file.write(f"Epoch: Average percentage_patch_drop: {average_percent}, Accuracy: {accuracy}, processed_image: {processed_image}, scan: {w}\n")


# %% Original cell 10
def load_mapping_file(json_file):
    with open(json_file, 'r') as f:
        mapping_data = json.load(f)
    return mapping_data


# %% Original cell 11
import torchvision.transforms as transforms
def transform_image(image, device):

    image=transforms.ToTensor()(image)
    image= image.unsqueeze(0)
    image=image.to(device)    
    

    
    return image


# %% Original cell 12
mapping_data = load_mapping_file(json_file)

def get_key_by_value(predicted_index, class_name):
    
    for key, values in mapping_data.items():
        key = int(key)
        
        if class_name in values:
            # print(key, predicted_index, "key and predicted_index")
            # print(f"Found class_name '{class_name}' in values: {values}")
            if int(key) == predicted_index:
                return True
    return False


# %% Original cell 13
import numpy as np

def rgb_to_gray(image):
    """
    Convert an RGB image to grayscale.

    Parameters:
    image (numpy.ndarray): Input RGB image as a NumPy array.

    # Returns:
    numpy.ndarray: Grayscale image.
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an RGB image with 3 channels.")
    
    # Extract the R, G, B channels
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    
    # Apply the grayscale conversion formula
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    
    # Convert to uint8 (0-255 range)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    
    return gray


# %% Original cell 14
# def compute_inv_covariance(x):
#     # x: (batch_size, num_tokens, embedding_dim)
#     # Flatten batch and token dimensions
#     B, N, D = x.shape
#     x_flat = x.reshape(-1, D)  # (B*N, D)
    
#     # Compute mean and center
#     mean = x_flat.mean(dim=0, keepdim=True)
#     centered = x_flat - mean
    
#     # Compute covariance: (D, D)
#     cov = centered.T @ centered / (x_flat.shape[0] - 1)
    
#     # Regularization for numerical stability
#     eps = 1e-6
#     cov += eps * torch.eye(D, device=x.device)
    
#     # Invert covariance matrix
#     inv_cov = torch.inverse(cov)
#     return inv_cov  # (D, D)

# def mahalanobis_distance(x, y, inv_cov):
#     # x: (B, N, D), y: (B, D), inv_cov: (D, D)
#     diff = x - y.unsqueeze(1)  # (B, N, D)
#     left = torch.matmul(diff, inv_cov)  # (B, N, D)
#     mahal = (left * diff).sum(dim=-1).sqrt()  # (B, N)
#     return mahal


# %% Original cell 15
# import torch
# import numpy as np
# from sklearn.metrics import pairwise_distances  # ✅ Correct import

# def compute_sklearn_mahalanobis(x, y):
#     """
#     Compute Mahalanobis distances between each token and CLS token using sklearn.
    
#     x: (B, N, D) - tokens
#     y: (B, D)    - cls_token
#     Returns: (B, N) Mahalanobis distances
#     """
#     B, N, D = x.shape
#     distances = []

#     for b in range(B):
#         x_np = x[b].detach().cpu().numpy()               # (N, D)
#         y_np = y[b].detach().cpu().numpy().reshape(1, -1)  # (1, D)

#         # Covariance + small regularization
#         cov = np.cov(x_np.T)
#         cov += 1e-3 * np.eye(D)

#         # Inverse of covariance matrix
#         inv_cov = np.linalg.inv(cov)

#         # Mahalanobis distances using sklearn
#         dists = pairwise_distances(x_np, y_np, metric='mahalanobis', VI=inv_cov).flatten()
#         distances.append(torch.tensor(dists, device=x.device))

#     return torch.stack(distances)  # (B, N)


# %% Original cell 16
import torch

def pearson_corr(cls_token, tokens):
    """
    Compute Pearson correlation between CLS token and each patch token.
    cls_token: Tensor of shape (B, D)
    tokens: Tensor of shape (B, N, D)
    Returns: Tensor of shape (B, N)
    """
    B, N, D = tokens.shape

    # LayerNorm-style normalization (mean and variance)
    cls_mean = cls_token.mean(dim=1, keepdim=True)                    # (B, 1)
    cls_var = cls_token.var(dim=1, keepdim=True, unbiased=False)      # (B, 1)
    cls_norm = (cls_token - cls_mean) / torch.sqrt(cls_var + 1e-6)    # (B, D)

    tokens_mean = tokens.mean(dim=2, keepdim=True)                    # (B, N, 1)
    tokens_var = tokens.var(dim=2, keepdim=True, unbiased=False)      # (B, N, 1)
    tokens_norm = (tokens - tokens_mean) / torch.sqrt(tokens_var + 1e-6)  # (B, N, D)

    # Pearson correlation via dot product of normalized vectors
    corr = torch.einsum('bd,bnd->bn', cls_norm, tokens_norm) / D      # (B, N)

    return corr


# %% Original cell 17
mask2= []

img_path2 = "/home/viraj/my_project/val/val2/n01773157/ILSVRC2012_val_00004118.JPEG"
# Image processing function
def apply_mask_with_white_shade(image_path, mask_tensor, opacity=0.9, border_size=10):
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


# %% Original cell 18
def append_values_to_file(filename, single_value):
    """
    Appends a single value to the specified file.

    Args:
        filename (str): The path to the file to be modified.
        single_value (float): The single float value to append to the file.
    """
    try:
        # Open the file in 'a' (append) mode.
        # The 'with' statement ensures the file is properly closed.
        with open(filename, 'a') as f:
            # The error "TypeError: 'float' object is not iterable" occurs
            # if you try to use a "for" loop on a single number.
            # This corrected version writes the single value directly without a loop.
            f.write(f"{single_value:.4f} ")
            
    except IOError as e:
        print(f"An error occurred while writing to {filename}: {e}")


filename = "appended_values.txt"


# %% Original cell 19
# import torch
# import torch.nn as nn
# import timm
# from timm.models.vision_transformer import VisionTransformer, Block
# import numpy as np
# import torch.nn.functional as F
# import seaborn as sns
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import numpy as np
# from PIL import Image, ImageOps
# import matplotlib.pyplot as plt
# import torch
# from scipy.spatial.distance import mahalanobis
# import numpy as np
# import seaborn as sns

# class CustomAttention(nn.Module):
#     def __init__(self, attn, num_heads):
#         super(CustomAttention, self).__init__()
#         self.num_heads = num_heads
#         self.scale = attn.scale
#         self.qkv = attn.qkv
#         self.attn_drop = attn.attn_drop
#         self.proj = attn.proj
#         self.proj_drop = attn.proj_drop

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         mask_attention = None
#         avg_attn = attn.mean(dim=1)
#         cls_token_attn = avg_attn[:, 0, :] 
#         ###########################################################################################
#         threshold = torch.quantile(cls_token_attn, drop_rate_layer , dim=-1, keepdim=True)
#         mask_attention = cls_token_attn >= threshold
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x, mask_attention

# class PatchDropout(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.drop_rate = drop_rate  # Drop 20% of the tokens

#     def forward(self, x, mask=None):
#         """
#         Processes input tokens by grouping them into 2x2 patches, pruning the least 
#         important groups, and returning a mask of the kept tokens.
    
#         x: [B, 197, D] - Input tensor with batch size B, 197 tokens, and dimension D.
#         drop_rate: The fraction of groups to drop.
    
#         Returns: 
#             - x_filtered: Filtered tokens after dropping groups.
#             - full_mask: A binary mask of shape [B, 197] where 1 indicates a kept
#                          token and 0 indicates a dropped token.
#         """
#         B, seq_len, D = x.shape
#         assert seq_len == 197, f"Expected sequence length of 197, but got {seq_len}"
    
#         # Separate the CLS token from the patch tokens
#         cls_token = x[:, 0:1, :]  # Shape: [B, 1, D]
#         tokens = x[:, 1:, :]      # Shape: [B, 196, D]
    
#         # ---- Step 1: Calculate Pearson correlation between CLS and patch tokens ----
#         # corr = pearson_corr(cls_token.squeeze(1), tokens)  # Shape: [B, 196]

#             # --- Step 3: Calculate Pearson correlation ---
#     # --- Step 3: Calculate Pearson correlation ---
#         correlations = pearson_corr(cls_token.squeeze(1), tokens)  # Shape: [B, 196]
#         # print(f"Calculated correlations shape: {correlations.shape}")
#         print(f"Average correlation value: {torch.mean(correlations).item():.4f}")
#         values = round(torch.mean(correlations).item(), 4)
#         append_values_to_file(filename, values)
#         # --- Step 4: Visualize the correlation heatmap ---
#         # Get the correlations for the first item in the batch
#         corr_for_viz = correlations[0].detach().cpu().numpy()
#         # Reshape to the 14x14 grid for visualization
#         corr_grid = corr_for_viz.reshape(14, 14)
    
#         # Plot the heatmap using seaborn with custom styling
#         plt.figure(figsize=(8, 8))
#         # Create the heatmap with a yellow-orange-red color map, no annotations,
#         # no color bar, and a black boundary around each cell.
#         ax = sns.heatmap(
#             corr_grid, 
#             annot=False, 
#             cmap='YlOrRd', 
#             cbar=False, 
#             xticklabels=False, 
#             yticklabels=False,
#             linewidths=1.0,
#             linecolor='black',
#             square=True
#         )
#         # Remove axis ticks for a cleaner look
#         ax.tick_params(left=False, bottom=False)
#         # plt.show()
    

#         return x
#         # # ---- Step 2: Reshape tokens and scores into a 14x14 spatial grid ----
#         # tokens_grid = tokens.view(B, 14, 14, D)  # Shape: [B, 14, 14, D]
#         # scores_grid = corr.view(B, 14, 14)       # Shape: [B, 14, 14]
    
#         # # ---- Step 3: Group into 2x2 patches ----
#         # tokens_patches = tokens_grid.view(B, 7, 2, 7, 2, D)
#         # scores_patches = scores_grid.view(B, 7, 2, 7, 2)
    
#         # tokens_patches = tokens_patches.permute(0, 1, 3, 2, 4, 5).reshape(B, 49, 4, D)
#         # scores_patches = scores_patches.permute(0, 1, 3, 2, 4).reshape(B, 49, 4)
        
#         # num_groups = 49
    
#         # # ---- Step 4: Calculate a single score for each group by summing token scores ----
#         # group_scores = scores_patches.sum(dim=-1)  # Shape: [B, 49]
    
#         # # ---- Step 5: Keep the top-k scoring groups ----
#         # num_to_keep = round(num_groups * (1 - drop_rate))
#         # if num_to_keep == 0 and num_groups > 0:
#         #     num_to_keep = 1
        
#         # _, topk_idx = torch.topk(group_scores, num_to_keep, dim=1, largest=False)  # Shape: [B, num_to_keep]
    
#         # # ---- Step 6: Create the binary mask with correct token indices ----
#         # # Start with a mask of zeros for all 49 groups.
#         # group_mask = torch.zeros(B, num_groups, dtype=torch.int, device=x.device)
#         # # Use scatter_ to place a 1 at the indices of the groups we are keeping.
#         # group_mask.scatter_(1, topk_idx, 1)
    
#         # # To create a token mask with correct indices, we must reverse the grouping operations.
#         # # [B, 49] -> [B, 49, 4] -> [B, 7, 7, 2, 2]
#         # mask_patches = group_mask.unsqueeze(-1).expand(-1, -1, 4).view(B, 7, 7, 2, 2)
#         # # Reverse the permutation: [B, 7, 7, 2, 2] -> [B, 7, 2, 7, 2]
#         # mask_permuted = mask_patches.permute(0, 1, 3, 2, 4)
#         # # Reshape back to the 14x14 grid: [B, 7, 2, 7, 2] -> [B, 14, 14]
#         # token_mask_grid = mask_permuted.reshape(B, 14, 14)
#         # # Flatten to the final token mask: [B, 14, 14] -> [B, 196]
#         # token_mask = token_mask_grid.reshape(B, 196)
#         # print(token_mask)
    
#         # # Create the full mask by prepending a 1 for the CLS token, which is always kept.
#         # full_mask = torch.cat([torch.ones(B, 1, dtype=torch.int, device=x.device), token_mask], dim=1)
    
#         # # ---- Step 7: Gather the tokens from the kept groups ----
#         # idx_expanded = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, D)
#         # kept_groups = torch.gather(tokens_patches, 1, idx_expanded)
        
#         # kept_tokens = kept_groups.reshape(B, num_to_keep * 4, D)
    
#         # # ---- Step 8: Prepend the CLS token back to the sequence ----
#         # x_filtered = torch.cat([cls_token, kept_tokens], dim=1)
        
#         # print(f"Original shape: {x.shape}")
#         # print(f"Filtered shape: {x_filtered.shape}")
#         # print(f"Mask shape: {full_mask.shape}")
    
#         # # return x_filtered


#     # def forward(self, x, mask):
#     #     """
#     #     x: [B, 197, D]
#     #     Returns: filtered tokens after dropping 50% groups
#     #     """
#     #     image_path=mask
#     #     B, seq_len, D = x.shape
#     #     assert seq_len == 197, f"Expected sequence length of 197, but got {seq_len}"

#     #     cls_token = x[:, 0:1, :]  # [B,1,D]
#     #     tokens = x[:, 1:, :]      # [B,196,D]

#     #     # ---- Step 1: Pearson correlation ----
#     #     corr = pearson_corr(cls_token.squeeze(1), tokens)   # [B,196]

#     #     # ---- Step 2: reshape into 14×14 ----
#     #     tokens_grid = tokens.view(B, 14, 14, D)  # [B,14,14,D]
#     #     scores_grid = corr.view(B, 14, 14)       # [B,14,14]

#     #     # ---- Step 3: group into 2×2 patches ----
#     #     tokens_patches = tokens_grid.view(B, 7, 2, 7, 2, D)   # [B,7,2,7,2,D]
#     #     scores_patches = scores_grid.view(B, 7, 2, 7, 2)      # [B,7,2,7,2]

#     #     tokens_patches = tokens_patches.permute(0,1,3,2,4,5).reshape(B, 49, 4, D)  # [B,49,4,D]
#     #     scores_patches = scores_patches.permute(0,1,3,2,4).reshape(B, 49, 4)       # [B,49,4]

#     #     # ---- Step 4: sum scores per patch ----
#     #     group_scores = scores_patches.sum(dim=-1)  # [B,49]

#     #     # ---- Step 5: keep top-k groups ----
#     #     k = int(drop_rate * group_scores.shape[1])   # keep 50% groups
#     #     topk_vals, topk_idx = torch.topk(group_scores, k, dim=1, largest=False)  # [B,k]
#     #     # print(topk_idx)

#     #     group_mask = torch.zeros(B, 49, dtype=torch.int, device=x.device)
#     #     group_mask.scatter_(1, topk_idx, 1)  # mark kept groups as 1

#     #     # expand back to 196 tokens (4 per group)
#     #     token_mask = group_mask.unsqueeze(-1).expand(-1, -1, 4).reshape(B, 196)  # [B,196]
#     #     print(token_mask)

#     #     # add CLS (always kept)
#     #     full_mask = torch.cat([torch.ones(B,1, dtype=torch.int, device=x.device), token_mask], dim=1)  # [B,197]

#     #     # print("Mask (1=kept, 0=dropped):")
#     #     # print(full_mask)

#     #     # gather kept groups
#     #     idx_expanded = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, D)  # [B,k,4,D]
#     #     # print(idx_expanded)
#     #     kept_groups = torch.gather(tokens_patches, 1, idx_expanded)   # [B,k,4,D]
#     #     # print(kept_groups)

#     #     kept_tokens = kept_groups.reshape(B, k*4, D)  # [B, kept*4, D]

#     #     # ---- Step 6: prepend CLS ----
#     #     x_filtered = torch.cat([cls_token, kept_tokens], dim=1)  # [B, 1+kept*4, D]
#     #     print(x_filtered.shape)

#         # return x_filtered
#         # #####################################################################################################################################
#         # drop_rate=0.999
#         # # print(x)
#         # # print(mask,"image_path")



#         # image_path=mask

        
#         # # img_path2=mask
#         # # batch_size, seq_len, embedding_dim = x.shape
#         # # assert seq_len == 197, "Expected sequence length of 197, but got {}".format(seq_len)

#         # # # Extract CLS token (first token)
#         # # cls_token = x[:, 0, :]  # Shape: (batch_size, embedding_dim)

#         # # # Compute cosine similarity with all other tokens
#         # # similarity_scores = F.cosine_similarity(x[:, 1:, :], cls_token.unsqueeze(1), dim=-1)  # (batch_size, 196)
#         # # # print(similarity_scores.shape)

#         # #                     # Convert tensor to numpy
        
#         # # # Convert tensor to numpy
#         # # similarity_grid = similarity_scores[0].reshape(14, 14).detach().cpu().numpy()


        
#         # # # Determine threshold to drop top 20% most similar tokens
#         # # k = int(seq_len * drop_rate)  # 20% of 196 tokens (~39 tokens)
#         # # threshold_values, _ = torch.kthvalue(similarity_scores, seq_len - k-1 , dim=1, keepdim=True)  # (batch_size, 1)
#         # # # print(threshold_values)

#         # # # Create a mask to **remove top 20% (high similarity scores)**
#         # # keep_mask = similarity_scores <= threshold_values  # (batch_size, 196)
    
#         # # # Sample mask tensor from CUDA device (moved to CPU for processing)
#         # # mask_tensor = keep_mask

#         # batch_size, seq_len, embedding_dim = x.shape
#         # assert seq_len == 197, f"Expected sequence length of 197, but got {seq_len}"
        
#         # # Extract CLS token
#         # cls_token = x[:, 0, :]  # (B, D)
#         # tokens = x[:, 1:, :]    # (B, 196, D)
    
#         # # Recompute inverse covariance matrix from all tokens (excluding CLS)
#         # # inv_cov = compute_inv_covariance(tokens)  # (D, D)
    
#         # # # Compute Mahalanobis distance between each token and CLS
#         # # distances = mahalanobis_distance(tokens, cls_token, inv_cov)  # (B, 196)
#         # # print(distances)
#         # distances = pearson_corr(cls_token, tokens)
    
#         # # Keep bottom 80% farthest tokens (drop closest 20%)
#         # k = int((seq_len - 1) * drop_rate)  # 20% of 196 tokens
#         # # print(k)
#         # threshold_values, _ = torch.kthvalue(distances, k + 1, dim=1, keepdim=True)  # (B, 1)
    
#         # keep_mask = distances <= threshold_values  # (B, 196)
#         # # print(keep_mask)
#         # # print(keep_mask)
#         # # keep_mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=x.device), keep_mask], dim=1)  # (B, 197)
    
#         # # Apply mask to x
#         # # x_filtered = x[keep_mask].view(batch_size, -1, embedding_dim)
#         # # print(x_filtered.shape)
#         # # token_counts_per_image.append(x_filtered.shape[1])
#         # mask_tensor = keep_mask


                
#         # # Main logic
#         # batch_size, seq_len, embedding_dim = x.shape
#         # assert seq_len == 197, f"Expected sequence length of 197, but got {seq_len}"
        
#         # # Extract CLS token and patch tokens
#         # cls_token = x[:, 0, :]       # (B, D)
#         # tokens = x[:, 1:, :]         # (B, 196, D)
        
#         # # Compute Mahalanobis distances using sklearn
#         # distances = compute_sklearn_mahalanobis(tokens, cls_token)  # (B, 196)
#         # # print(distances)
        
#         # # Drop closest 20%, keep farthest 80%
#         # k = int((seq_len - 1) * drop_rate)  # drop_rate e.g., 0.2
#         # # print(k)
        
#         # # Get the (k+1)-th smallest value in each row to create threshold
#         # threshold_values, _ = torch.kthvalue(distances, k + 1, dim=1, keepdim=True)  # (B, 1)
#         # keep_mask = distances > threshold_values  # (B, 196)
#         # mask_tensor = keep_mask

#         # image_path=mask
#         # mask_tensor = token_mask
#         # Image processing function
#         # def apply_mask_with_white_shade(image_path, mask_tensor, opacity=0.9, border_size=10):
#         #     mask = mask_tensor.to('cpu').numpy().flatten()
    
#         #     # Load and resize image
#         #     image = Image.open(image_path).convert("RGB").resize((224, 224))
#         #     image_np = np.array(image).astype(np.float32)
        
#         #     patch_size = 16
#         #     output_image = image_np.copy()
            
#         #     assert mask.shape[0] == 196, "Mask must have 196 elements (14x14 patches)"
        
#         #     idx = 0
#         #     for i in range(0, 224, patch_size):
#         #         for j in range(0, 224, patch_size):
#         #             if not mask[idx]:
#         #                 # Blend the patch with white using the specified opacity
#         #                 original_patch = output_image[i:i+patch_size, j:j+patch_size, :]
#         #                 white_patch = np.ones_like(original_patch) * 255.0
#         #                 blended_patch = (1 - opacity) * original_patch + opacity * white_patch
#         #                 output_image[i:i+patch_size, j:j+patch_size, :] = blended_patch
#         #             idx += 1
        
#         #     output_image = np.clip(output_image, 0, 255).astype(np.uint8)
#         #     result_image = Image.fromarray(output_image)
        
#         #     # Add black border
#         #     result_image_with_border = ImageOps.expand(result_image, border=border_size, fill='black')
#         #     return result_image_with_border
        
#         # ---- Example usage ----
#         # image_path =   # Update with your actual image path
#         # result_image = apply_mask_with_white_shade(image_path, mask_tensor, opacity=0.90, border_size=1)
        
#         # # Display the result
#         # plt.figure(figsize=(6, 6))
#         # plt.imshow(result_image)
#         # plt.axis('off')
#         # # plt.title("Masked Image with White Overlay and Black Border")
#         # plt.show()
        
        
    

















        


        
#         # print(keep_mask.size, "keep_mask")
        
#         # # Append True for CLS token (always keep it)
#         # keep_mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=x.device), keep_mask], dim=1)  # (batch_size, 197)

#         # # Apply mask and return filtered tensor
#         # x_filtered = x[keep_mask].view(batch_size, -1, embedding_dim)  # Adjust shape after filtering
#         # print(x_filtered.shape)
#         # return x_filtered

# class CustomBlock(nn.Module):
#     def __init__(self, block):
#         super(CustomBlock, self).__init__()
#         self.norm1 = block.norm1
#         self.attn = CustomAttention(block.attn, block.attn.num_heads)
#         self.ls1 = block.ls1
#         self.drop_path1 = block.drop_path1
#         self.norm2 = block.norm2
#         self.mlp = block.mlp
#         self.ls2 = block.ls2
#         self.drop_path2 = block.drop_path2

#     def forward(self, x, apply_patch_drop=False, img_path=None):

        
#         x_residual = self.norm1(x)
        
#         x_residual, mask_attention = self.attn(x_residual)
#         x = x + self.drop_path1(self.ls1(x_residual))
#         x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
#         if apply_patch_drop:
            
            
#             # mask2= mask_attention
#             # image_path = mask  # Update with your actual image path
#             # print(mask_attention.shape)
#             mask_attention2 = mask_attention[:, 1:]
#             # print(mask_attention)
#             img_path222=img_path
#             # result_image = apply_mask_with_white_shade(img_path222,  mask_attention2, opacity=0.90, border_size=1)
#                     # Display the result
#             # plt.figure(figsize=(6, 6))
#             # plt.imshow(result_image)
#             # plt.axis('off')
#             # # plt.title("Masked Image with White Overlay and Black Border")
#             # plt.show()
            
            
#             # batch_size, seq_len, embedding_dim = x.shape
#             # # assert seq_len == 197, "Expected sequence length of 197, but got {}".format(seq_len)
    
#             # # Extract CLS token (first token)
#             # cls_token = x[:, 0, :]  # Shape: (batch_size, embedding_dim)
    
#             # # Compute cosine similarity with all other tokens
#             # similarity_scores = F.cosine_similarity(x[:, 1:, :], cls_token.unsqueeze(1), dim=-1)  # (batch_size, 196)

#             # # Determine threshold to drop top 20% most similar tokens
#             # k = int(seq_len *  drop_rate_layers)  # 20% of 196 tokens (~39 tokens)
#             # # print(k, "k_value")
#             # threshold_values, _ = torch.kthvalue(similarity_scores, seq_len - k-1 , dim=1, keepdim=True)  # (batch_size, 1)
#             # # print(threshold_values,"thresholdssss")
    
#             # # Create a mask to **remove top 20% (high similarity scores)**
#             # keep_mask = similarity_scores <= threshold_values  # (batch_size, 196)
#             # # print(keep_mask.shape, "keep_mask")
            
#             # # Append True for CLS token (always keep it)
#             # keep_mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=x.device), keep_mask], dim=1)  # (batch_size, 197)
    
#             # # Apply mask and return filtered tensor
#             # x_filtered = x[keep_mask].view(batch_size, -1, embedding_dim)  # Adjust shape after filtering
#             # print(x_filtered.shape)

#             cls_token = x[:, 0].unsqueeze(1)  # Keep the class token
#             other_tokens = x[:, 1:]
#             # selected_tokens = other_tokens[mask_attention[:, 1:]] 
#             # selected_tokens = selected_tokens.unsqueeze(0)
#             # x = torch.cat((cls_token, selected_tokens), dim=1)


#             correlations = pearson_corr(cls_token.squeeze(1),  other_tokens)  # Shape: [B, 196]
#             # print(f"Calculated correlations shape: {correlations.shape}")
#             print(f"Average correlation value: {torch.mean(correlations).item():.4f}")
            
#             values = round(torch.mean(correlations).item(), 4)
#             append_values_to_file(filename, values)
#              # append_value_to_file(filename, value)

            
#             # --- Step 4: Visualize the correlation heatmap ---
#             # Get the correlations for the first item in the batch
#             corr_for_viz = correlations[0].detach().cpu().numpy()
#             # Reshape to the 14x14 grid for visualization
#             corr_grid = corr_for_viz.reshape(14, 14)
        
#             # Plot the heatmap using seaborn with custom styling

#             plt.figure(figsize=(8, 8))
#             ax = sns.heatmap(
#                 corr_grid, 
#                 annot=False, 
#                 cmap='YlOrRd', 
#                 cbar=True,            # ensure colorbar
#                 vmin=-0.2, vmax=1.0,
#                 xticklabels=False, 
#                 yticklabels=False,
#                 linewidths=1.0,
#                 linecolor='black',
#                 square=True
#             )
            
#             # Get the figure’s colorbar instead of ax.collections[0]
#             cbar = ax.figure.axes[-1]   # the last axis is the colorbar
#             cbar.set_ylabel("Correlation Value", fontsize=12)
#             cbar.tick_params(labelsize=10)
            
#             # Remove axis ticks
#             ax.tick_params(left=False, bottom=False)
            
#             plt.show()
            

            
#             # plt.figure(figsize=(8, 8))
#             # # Create the heatmap with a yellow-orange-red color map, no annotations,
#             # # no color bar, and a black boundary around each cell.
#             # ax = sns.heatmap(
#             #     corr_grid, 
#             #     annot=False, 
#             #     cmap='YlOrRd', 
#             #     cbar=False, 
#             #     xticklabels=False, 
#             #     yticklabels=False,
#             #     linewidths=1.0,
#             #     linecolor='black',
#             #     square=True
#             # )
            
#             # # Customize the colorbar
#             # cbar = ax.collections[0].colorbar
#             # cbar.set_label("Correlation Value", fontsize=12)   # label for the scale
#             # cbar.ax.tick_params(labelsize=10) 
#             # # Remove axis ticks for a cleaner look
#             # ax.tick_params(left=False, bottom=False)
#             # # plt.show()
        
    
#             return x     
#         return x

# class CustomVisionTransformer(VisionTransformer):
#     def __init__(self, patch_drop_blocks=None, **kwargs):
#         super(CustomVisionTransformer, self).__init__(**kwargs)
#         self.patch_drop_blocks = patch_drop_blocks if patch_drop_blocks is not None else []
#         self.patch_drop = PatchDropout()
#         self.blocks = nn.Sequential(*[
#             CustomBlock(block) if i in self.patch_drop_blocks else block
#             for i, block in enumerate(self.blocks)
#         ])

#     def forward(self, x, mask=None):
#         B = x.shape[0]
#         x = self.patch_embed(x)
#         cls_tokens = self.cls_token.expand(B, -1, -1)  # class token
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = self.pos_drop(x + self.pos_embed)
#         if mask is not None:
#             x = self.patch_drop(x, mask)

#         for i, block in enumerate(self.blocks):
#             if isinstance(block, CustomBlock):
#                 x = block(x, apply_patch_drop=True, img_path=mask)
#                 # print(i, x.shape)
#             else:
#                 x = block(x)
#         x = self.norm(x)
#         x = self.head(x[:, 0])
#         return x

# # Example usage
# # my_array = np.random.randint(1, 2, size=197)  # Updated to have values 0 or 1
# my_array="/home/viraj/my_project/val/val5/vallll/ILSVRC2012_val_00006306.JPEG"
# custom_vit = CustomVisionTransformer(
#     img_size=224,
#     patch_size=16,
#     embed_dim=768,         # ViT-Large uses 1024
#     depth=12,               # ViT-Large has 24 transformer blocks
#     num_heads=12,           # ViT-Large has 16 attention heads
#     mlp_ratio=4,
#     qkv_bias=True,
#     norm_layer=nn.LayerNorm,
#     num_classes=1000,
#     patch_drop_blocks=[0,1,2,3,4,5,6,7,8,9,10,11]  # Adjusted to drop patches uniformly across deeper layers
# ).to(device)



# # Load pretrained weights
# # pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True)
# custom_vit.load_state_dict(pretrained_vit.state_dict(), strict=True)

# # Forward pass with input tensor and masks
# input_tensor = torch.randn(1, 3, 224, 224).to(device)
# output = custom_vit(input_tensor, mask=my_array)
# print(output.shape)  # Should be [1, 1000]


# %% Original cell 20
import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer, Block
import numpy as np
import torch.nn.functional as F

drop_rate_layer = 0.00001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pearson_corr(x, y):
    """Calculates Pearson correlation between CLS token (x) and patch tokens (y)."""
    x_mean = x.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)
    
    x_sub = x - x_mean
    y_sub = y - y_mean
    
    cov = (x_sub.unsqueeze(1) * y_sub).sum(dim=-1)
    var = torch.sqrt((x_sub.unsqueeze(1) ** 2).sum(dim=-1) * (y_sub ** 2).sum(dim=-1))
    
    return cov / (var + 1e-8)

def append_values_to_file(block_idx, values):
    """Appends raw values to a block-specific file (e.g., appended_values_block_-1.txt)."""
    filename = f"appended_values_block_{block_idx}.txt"
    with open(filename, "a") as f:
        if isinstance(values, (list, np.ndarray, torch.Tensor)):
            formatted_vals = " ".join([f"{v:.4f}" for v in values])
            f.write(formatted_vals + " ")
        else:
            f.write(f"{values:.4f} ")

class CustomAttention(nn.Module):
    def __init__(self, attn, num_heads):
        super(CustomAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = attn.scale
        self.qkv = attn.qkv
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        avg_attn = attn.mean(dim=1)
        cls_token_attn = avg_attn[:, 0, :] 
        threshold = torch.quantile(cls_token_attn, drop_rate_layer, dim=-1, keepdim=True)
        mask_attention = cls_token_attn >= threshold
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, mask_attention

class PatchDropout(nn.Module):
    def __init__(self, block_idx=-1):
        super().__init__()
        self.block_idx = block_idx

    def forward(self, x, mask=None):
        B, seq_len, D = x.shape
        assert seq_len == 197, f"Expected sequence length of 197, but got {seq_len}"
        
        cls_token = x[:, 0:1, :]
        tokens = x[:, 1:, :]
        
        correlations = pearson_corr(cls_token.squeeze(1), tokens)
        
        # Save raw values for block -1
        raw_values = correlations.detach().cpu().numpy().flatten()
        append_values_to_file(self.block_idx, raw_values)

        return x

class CustomBlock(nn.Module):
    def __init__(self, block, block_idx):
        super(CustomBlock, self).__init__()
        self.block_idx = block_idx
        self.norm1 = block.norm1
        self.attn = CustomAttention(block.attn, block.attn.num_heads)
        
        self.ls1 = getattr(block, 'ls1', nn.Identity())
        self.drop_path1 = getattr(block, 'drop_path1', nn.Identity())
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.ls2 = getattr(block, 'ls2', nn.Identity())
        self.drop_path2 = getattr(block, 'drop_path2', nn.Identity())

    def forward(self, x, apply_patch_drop=False, img_path=None):
        x_residual = self.norm1(x)
        x_residual, mask_attention = self.attn(x_residual)
        x = x + self.drop_path1(self.ls1(x_residual))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        if apply_patch_drop:
            cls_token = x[:, 0].unsqueeze(1)
            other_tokens = x[:, 1:]

            correlations = pearson_corr(cls_token.squeeze(1), other_tokens)
            
            # Save raw values for block 0 through 11
            raw_values = correlations.detach().cpu().numpy().flatten()
            append_values_to_file(self.block_idx, raw_values)

        return x

class CustomVisionTransformer(VisionTransformer):
    def __init__(self, patch_drop_blocks=None, **kwargs):
        super(CustomVisionTransformer, self).__init__(**kwargs)
        self.patch_drop_blocks = patch_drop_blocks if patch_drop_blocks is not None else []
        self.patch_drop = PatchDropout(block_idx=-1)
        self.blocks = nn.Sequential(*[
            CustomBlock(block, block_idx=i) if i in self.patch_drop_blocks else block
            for i, block in enumerate(self.blocks)
        ])

    def forward(self, x, mask=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        if mask is not None:
            x = self.patch_drop(x, mask)

        for i, block in enumerate(self.blocks):
            if isinstance(block, CustomBlock):
                x = block(x, apply_patch_drop=True, img_path=mask)
            else:
                x = block(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

# Model Setup
pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True)

custom_vit = CustomVisionTransformer(
    img_size=224,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer=nn.LayerNorm,
    num_classes=1000,
    patch_drop_blocks=list(range(12))
).to(device)

custom_vit.load_state_dict(pretrained_vit.state_dict(), strict=True)

# Process multiple images to continuously append raw values per block
image_paths = "/home/viraj/my_project/val/val5/vallll/ILSVRC2012_val_00006306.JPEG"

for img_path in image_paths:
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    output = custom_vit(input_tensor, mask=img_path)
    print(f"Processed {img_path}, Output shape: {output.shape}")


# %% Original cell 21



# %% Original cell 22
from qtorch.quant import Quantizer, quantizer
from qtorch import FixedPoint
# Define the fixed-point quantizer
weight_quantizer = quantizer(forward_number=FixedPoint(16, 8), forward_rounding="nearest")
act_quantizer = quantizer(forward_number=FixedPoint(16, 8), forward_rounding="nearest")


# %% Original cell 23
import torch
import torch.nn as nn

# Define the QuantizedVisionTransformer class
class QuantizedVisionTransformer(nn.Module):
    def __init__(self, model, weight_quantizer, act_quantizer):
        super(QuantizedVisionTransformer, self).__init__()
        self.model = model
        self.weight_quantizer = weight_quantizer
        self.act_quantizer = act_quantizer
        self.apply(self._quantize_weights)

    def _quantize_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data = self.weight_quantizer(module.weight.data)
            if module.bias is not None:
                module.bias.data = self.weight_quantizer(module.bias.data)
        elif isinstance(module, nn.Conv2d):
            module.weight.data = self.weight_quantizer(module.weight.data)
            if module.bias is not None:
                module.bias.data = self.weight_quantizer(module.bias.data)

    def forward(self, x, mask):
        x = self.act_quantizer(x)
        # for name, module in self.model.named_children():
        #     x = module(x, mask) if 'block' in name else module(x)
        #     if isinstance(module, (nn.Linear, nn.Conv2d)):
        #         x = self.act_quantizer(x)
        return self.model(x, mask)

# Example usage:
# Assume `vision_transformer_model` is your pretrained model
# and `weight_quantizer` and `act_quantizer` are your quantization functions
quantized_model = QuantizedVisionTransformer(custom_vit, weight_quantizer, act_quantizer)


# %% Original cell 24
def check_quantization(model):
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Data type: {param.dtype}")
        print(f"Values: {param.view(-1)[:10]}")  # Print the first 10 values
        print(f"Storage size (bytes): {param.numel() * param.element_size()}")
        print()



# Check quantized weights and biases
# check_quantization(quantized_model)


# %% Original cell 25
# output = quantized_model(input_tensor, mask=my_array)


# %% Original cell 26
def apply_mask_with_white_shade2(image_input, mask_tensor, opacity=0.9, border_size=10):
    # Ensure mask is a numpy array
    if isinstance(mask_tensor, torch.Tensor):
        mask = mask_tensor.to('cpu').numpy().flatten()
    else:
        mask = np.array(mask_tensor).flatten()

    # Handle image input (can be file path or numpy array)
    if isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input).convert("RGB").resize((224, 224))
    else:  # assume it's a path
        image = Image.open(image_input).convert("RGB").resize((224, 224))

    image_np = np.array(image).astype(np.float32)

    patch_size = 16
    output_image = image_np.copy()

    assert mask.shape[0] == 196, "Mask must have 196 elements (14x14 patches)"

    idx = 0
    for i in range(0, 224, patch_size):
        for j in range(0, 224, patch_size):
            if not mask[idx]:
                # Blend patch with white
                original_patch = output_image[i:i+patch_size, j:j+patch_size, :]
                white_patch = np.ones_like(original_patch) * 255.0
                blended_patch = (1 - opacity) * original_patch + opacity * white_patch
                output_image[i:i+patch_size, j:j+patch_size, :] = blended_patch
            idx += 1

    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    result_image = Image.fromarray(output_image)

    # Add border
    result_image_with_border = ImageOps.expand(result_image, border=border_size, fill='black')
    return result_image_with_border


# %% Original cell 27
def plot_result(image):
    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.axis("off")
    plt.show()


# %% Original cell 28
drop_rate_layer = 0.00001


# %% Original cell 29
def main_processing(image):
    resized_image =  resize_nearest_neighbor(image, (224, 224))
    resized_image = bgr_to_rgb(resized_image)
    resized_image = resized_image.copy()
    # plot(resized_image)
    # mask2 = torch.randint(1, 2, (196,))
    # imageeee =apply_mask_with_white_shade2(image, mask2, opacity=0.9, border_size=1)
    # plot_result(imageeee)
    

    
    image_tensor= transform_image(resized_image, device)
    mask = np.random.randint(1, 2, size=197) 
    return image_tensor, mask
filename = "appended_valueseeeeeeeeeeeeeeeeeesssssssssssssssssssssssssssssssssssss222222222222.txt"


# %% Original cell 30
def save_percentage(file_path, average_percent1, average_percent2, average_percent3 ,average_percent4, average, count):
    with open(file_path, 'a') as file:
        file.write(f"Epoch: Average percentage_patch_drop: { average_percent1}, { average_percent2}, { average_percent3}, { average_percent4}, average: {average}, count{count}\n")


# %% Original cell 31
def convert_to_canvas(img_folder, model, device):
    correct_predictions = 0
    total_images = 0
    processed_images = 0
    
    for subfolder_name in os.listdir(img_folder):
        subfolder_path = os.path.join(img_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            # Process each image in the subfolder
            for img_name in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                
                    # Load the image
                    image = cv2.imread(img_path)
                    print(img_path)
                    img_path2=  img_path
                    if image is None:
                        print(f"Failed to load image: {img_path}")
                        continue

                    image_tensor, fast_mask = main_processing(image)

                    # print(img_path)
                    
                    # outputs = custom_vit( image_tensor, mask= fast_mask)
                    # img_path2= img_path
                    outputs= quantized_model(image_tensor, mask= img_path)

                    probs = F.softmax(outputs, dim=1)
                    predicted_idx = torch.argmax(outputs, dim=1).item()
                    predicted_prob = probs[0, predicted_idx].item()
                    print(f"Predicted class index: {predicted_idx}")
                    print(f"Probability: {predicted_prob:.4f}")
                    class72_prob = probs[0, 42].item()
                    print(f"Probability of belonging to class 72: {class72_prob:.4f}")

                    
                    match = re.search(r'val2/([^/]+)/', img_path)
                    if match:
                        class_name = match.group(1)
                    else:
                        class_name = None
                    
                    prediction_result = get_key_by_value(predicted_idx, class_name)
                    
                    if prediction_result is True:
                        correct_predictions += 1    
                    processed_images += 1
                    
                    # Print accuracy after processing every 50 images
                    if processed_images % 1 == 0:
                        # w= ( sum(scan)/ len(scan))
                        w=0
                        accuracy_so_far = (correct_predictions / processed_images) * 100
                        # average_percent = sum(patch_drop_percentage) / len(patch_drop_percentage)
                        
                        # print("average percentage embedding_drop " , average_percent , len(patch_drop_percentage))
                        print(f"Accuracy after processing {processed_images} images: {accuracy_so_far:.2f}%")
                        # save_average_percentage(txt_path,  average_percent_dro_embedding, accuracy_so_far, processed_images, w )
                    
                    


# %% Original cell 32
print(device)


# %% Original cell 33
convert_to_canvas(img_folder, pretrained_vit, device)
