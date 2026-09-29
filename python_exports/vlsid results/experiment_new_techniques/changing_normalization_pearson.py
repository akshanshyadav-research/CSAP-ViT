# Exported from experiments/vlsid results/experiment_new_techniques/changing_normalization_pearson.ipynb
# Historical code: review paths and side effects before executing.


# %% Original cell 0
patch_drop_blocks=[]
drop_rate=0.41
attention_pruning= 0.00


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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
img_folder = "/home/viraj/my_project/val/val2/"
txt_path = f"vit_base_similarity_dropblock{patch_drop_blocks}_droprate{drop_rate:.2f}_attnprune{attention_pruning:.2f}.txt"
json_file = "/home/viraj/my_project/imagenet_class_index.json"
# line_r=0
 


# %% Original cell 3
# pretrained_vit = vit_large_patch16_224(pretrained=True).to(device)

pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)


# %% Original cell 4
def plot(image):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
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


# %% Original cell 8
def save_average_percentage(file_path, average_percent, accuracy, processed_image, w):
    with open(file_path, 'a') as file:
        file.write(f"Epoch: Average percentage_patch_drop: {average_percent}, Accuracy: {accuracy}, processed_image: {processed_image}, scan: {w}\n")


# %% Original cell 9
def load_mapping_file(json_file):
    with open(json_file, 'r') as f:
        mapping_data = json.load(f)
    return mapping_data


# %% Original cell 10
import torchvision.transforms as transforms
def transform_image(image, device):

    image=transforms.ToTensor()(image)
    image= image.unsqueeze(0)
    image=image.to(device)    
    

    
    return image


# %% Original cell 11
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


# %% Original cell 12
patch_drop_percentage = []
def average_patch_drop(mask):
    count_ones = mask.count(1)
    count_ones=count_ones-1
    y = (((196 - count_ones ) / 196) * 100)
    patch_drop_percentage.append(y)
    


# %% Original cell 13
import torch

def pearson_corr(cls_token, tokens):
    """
    Compute Pearson correlation between CLS token and each patch token.
    cls_token: Tensor of shape (B, D)
    tokens: Tensor of shape (B, N, D)
    Returns: Tensor of shape (B, N)
    """
    B, N, D = tokens.shape

    # Normalize CLS and tokens by subtracting their mean and dividing by std deviation
    cls_mean = cls_token.mean(dim=1, keepdim=True)        # (B, 1)
    cls_std = cls_token.std(dim=1, keepdim=True) + 1e-6   # (B, 1)

    token_mean = tokens.mean(dim=2, keepdim=True)         # (B, N, 1)
    token_std = tokens.std(dim=2, keepdim=True) + 1e-6     # (B, N, 1)

    # Standardize
    cls_norm = (cls_token - cls_mean) / cls_std           # (B, D)
    tokens_norm = (tokens - token_mean) / token_std       # (B, N, D)

    corr = torch.einsum('bd,bnd->bn', cls_norm, tokens_norm) / D  # (B, N)

    return corr


# %% Original cell 14
token_counts_per_image = []


# %% Original cell 15
import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer, Block
import numpy as np
import torch.nn.functional as F
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
        mask_attention = None
        avg_attn = attn.mean(dim=1)
        cls_token_attn = avg_attn[:, 0, :] 
        threshold = torch.quantile(cls_token_attn, attention_pruning , dim=-1, keepdim=True)
        mask_attention = cls_token_attn >= threshold
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, mask_attention




class PatchDropout(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x, mask=None):
        """
        x: [B, 197, D]
        Returns: filtered tokens after dropping 50% groups
 """

        B, seq_len, D = x.shape
        assert seq_len == 197, f"Expected sequence length of 197, but got {seq_len}"

        # Separate the [CLS] token from the patch tokens
        cls_token = x[:, 0:1, :]  # Shape: [B, 1, D]
        tokens = x[:, 1:, :]      # Shape: [B, 196, D]

        # ---- Step 1: Calculate Pearson correlation between [CLS] and each patch token ----
        corr = pearson_corr(cls_token.squeeze(1), tokens)  # Shape: [B, 196]

        # ---- Step 2: Reshape tokens and scores into a 14x14 grid ----
        tokens_grid = tokens.view(B, 14, 14, D)  # Shape: [B, 14, 14, D]
        scores_grid = corr.view(B, 14, 14)      # Shape: [B, 14, 14]

        # ---- Step 3: Group the grid into 49 non-overlapping 2x2 patches ----
        # This creates 49 groups, each containing 4 tokens.
        tokens_patches = tokens_grid.unfold(1, 2, 2).unfold(2, 2, 2) # Shape: [B, 7, 7, D, 2, 2]
        tokens_patches = tokens_patches.reshape(B, 49, D, 4).permute(0, 1, 3, 2) # Shape: [B, 49, 4, D]

        scores_patches = scores_grid.unfold(1, 2, 2).unfold(2, 2, 2) # Shape: [B, 7, 7, 2, 2]
        scores_patches = scores_patches.reshape(B, 49, 4) # Shape: [B, 49, 4]

        # ---- Step 4: Calculate a single score for each group by summing patch scores ----
        group_scores = scores_patches.sum(dim=-1)  # Shape: [B, 49]
        # print(group_scores)
        mean = group_scores.mean(dim=1, keepdim=True)  # [B,1]
        std = group_scores.std(dim=1, keepdim=True)    # [B,1]
        alpha = 0.0  # can adjust: 0 = mean, >0 = more selective
        threshold = mean + alpha * std                 # [B,1]

        # ---- Step 6: create boolean mask of groups to keep ----
        keep_mask = group_scores >= threshold         # [B,49], True = keep

# ---- Step 7: indices of groups to keep (per batch) ----
        keep_idx_list = [keep_mask[b].nonzero(as_tuple=False).squeeze(-1) for b in range(group_scores.shape[0])]

        # mean = group_scores.mean(dim=1, keepdim=True)  # [B,1]
        # std = group_scores.std(dim=1, keepdim=True)    # [B,1]
        # threshold = mean * std  
        
        # # ---- Step 5: Identify groups to keep and groups to drop/average ----
        # num_groups = 49
        # k_to_keep = num_groups - int(num_groups * threshold)

        # # Get the indices of the groups to KEEP (those with the highest scores)
        # _, keep_idx = torch.topk(group_scores, k_to_keep, dim=1, largest=False)

        # # Get the indices of the groups to DROP (those with the lowest scores)
        # k_to_drop = num_groups - k_to_keep
        # if k_to_drop > 0:
        #     _, drop_idx = torch.topk(group_scores, k_to_drop, dim=1, largest=True)

        # ---- Step 6: Process the groups that are being kept ----
        # Expand indices to gather all 4 tokens from each kept group
        keep_idx_expanded = keep_idx_list.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, D)
        # print(keep_idx_expanded)
        kept_groups = torch.gather(tokens_patches, 1, keep_idx_expanded)
        # print(kept_groups)
        kept_tokens = kept_groups.reshape(B, k_to_keep * 4, D) # Flatten into a sequence

        # # ---- Step 7: Process the groups that are being dropped ----
        # if k_to_drop > 0:
        #     # Expand indices to gather all 4 tokens from each dropped group
        #     drop_idx_expanded = drop_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, D)
        #     dropped_groups = torch.gather(tokens_patches, 1, drop_idx_expanded)

        #     # Create a single summary embedding for each dropped group by averaging
        #     dropped_embedding = torch.mean(dropped_groups, dim=2) # Average across the 4 tokens

        #     # ---- Step 8: Combine [CLS], kept tokens, and averaged dropped embeddings ----
        #     x_filtered = torch.cat([cls_token, kept_tokens, dropped_embedding], dim=1)
        # else:
            # If no groups were dropped, just combine [CLS] and kept tokens
        x_filtered = torch.cat([cls_token, kept_tokens], dim=1)

        print(x_filtered.shape)
        # token_counts_per_image = token_counts_per_image.append(x_filtered[1])

        return x_filtered




 #    def forward(self, x, mask=None):
 #        """
 #        x: [B, 197, D]
 #        Returns: filtered tokens after dropping 50% groups
 # """

 #        B, seq_len, D = x.shape
 #        assert seq_len == 197, f"Expected sequence length of 197, but got {seq_len}"

 #        # Separate the [CLS] token from the patch tokens
 #        cls_token = x[:, 0:1, :]  # Shape: [B, 1, D]
 #        tokens = x[:, 1:, :]      # Shape: [B, 196, D]

 #        # ---- Step 1: Calculate Pearson correlation between [CLS] and each patch token ----
 #        corr = pearson_corr(cls_token.squeeze(1), tokens)  # Shape: [B, 196]

 #        # ---- Step 2: Reshape tokens and scores into a 14x14 grid ----
 #        tokens_grid = tokens.view(B, 14, 14, D)  # Shape: [B, 14, 14, D]
 #        scores_grid = corr.view(B, 14, 14)      # Shape: [B, 14, 14]

 #        # ---- Step 3: Group the grid into 49 non-overlapping 2x2 patches ----
 #        # This creates 49 groups, each containing 4 tokens.
 #        tokens_patches = tokens_grid.unfold(1, 2, 2).unfold(2, 2, 2) # Shape: [B, 7, 7, D, 2, 2]
 #        tokens_patches = tokens_patches.reshape(B, 49, D, 4).permute(0, 1, 3, 2) # Shape: [B, 49, 4, D]

 #        scores_patches = scores_grid.unfold(1, 2, 2).unfold(2, 2, 2) # Shape: [B, 7, 7, 2, 2]
 #        scores_patches = scores_patches.reshape(B, 49, 4) # Shape: [B, 49, 4]

 #        # ---- Step 4: Calculate a single score for each group by summing patch scores ----
 #        group_scores = scores_patches.sum(dim=-1)  # Shape: [B, 49]
 #        # print(group_scores)

 #        # ---- Step 5: Identify groups to keep and groups to drop/average ----
 #        num_groups = 49
 #        k_to_keep = num_groups - int(num_groups * drop_rate)

 #        # Get the indices of the groups to KEEP (those with the highest scores)
 #        _, keep_idx = torch.topk(group_scores, k_to_keep, dim=1, largest=False)

 #        # Get the indices of the groups to DROP (those with the lowest scores)
 #        k_to_drop = num_groups - k_to_keep
 #        if k_to_drop > 0:
 #            _, drop_idx = torch.topk(group_scores, k_to_drop, dim=1, largest=True)

 #        # ---- Step 6: Process the groups that are being kept ----
 #        # Expand indices to gather all 4 tokens from each kept group
 #        keep_idx_expanded = keep_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, D)
 #        # print(keep_idx_expanded)
 #        kept_groups = torch.gather(tokens_patches, 1, keep_idx_expanded)
 #        # print(kept_groups)
 #        kept_tokens = kept_groups.reshape(B, k_to_keep * 4, D) # Flatten into a sequence

 #        # # ---- Step 7: Process the groups that are being dropped ----
 #        # if k_to_drop > 0:
 #        #     # Expand indices to gather all 4 tokens from each dropped group
 #        #     drop_idx_expanded = drop_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, D)
 #        #     dropped_groups = torch.gather(tokens_patches, 1, drop_idx_expanded)

 #        #     # Create a single summary embedding for each dropped group by averaging
 #        #     dropped_embedding = torch.mean(dropped_groups, dim=2) # Average across the 4 tokens

 #        #     # ---- Step 8: Combine [CLS], kept tokens, and averaged dropped embeddings ----
 #        #     x_filtered = torch.cat([cls_token, kept_tokens, dropped_embedding], dim=1)
 #        # else:
 #            # If no groups were dropped, just combine [CLS] and kept tokens
 #        x_filtered = torch.cat([cls_token, kept_tokens], dim=1)

 #        print(x_filtered.shape)
 #        token_counts_per_image = token_counts_per_image.append(x_filtered[1])

 #        return x_filtered


    #     B, seq_len, D = x.shape
    #     assert seq_len == 197, f"Expected sequence length of 197, but got {seq_len}"

    #     cls_token = x[:, 0:1, :]  # [B,1,D]
    #     tokens = x[:, 1:, :]      # [B,196,D]

    #     # ---- Step 1: Pearson correlation ----
    #     corr = pearson_corr(cls_token.squeeze(1), tokens)   # [B,196]

    #     # ---- Step 2: reshape into 14×14 ----
    #     tokens_grid = tokens.view(B, 14, 14, D)  # [B,14,14,D]
    #     scores_grid = corr.view(B, 14, 14)       # [B,14,14]

    #     # ---- Step 3: group into 2×2 patches ----
    #     tokens_patches = tokens_grid.view(B, 7, 2, 7, 2, D)   # [B,7,2,7,2,D]
    #     scores_patches = scores_grid.view(B, 7, 2, 7, 2)      # [B,7,2,7,2]

    #     tokens_patches = tokens_patches.permute(0,1,3,2,4,5).reshape(B, 49, 4, D)  # [B,49,4,D]
    #     scores_patches = scores_patches.permute(0,1,3,2,4).reshape(B, 49, 4)       # [B,49,4]

    #     # ---- Step 4: sum scores per patch ----
    #     group_scores = scores_patches.sum(dim=-1)  # [B,49]

    #     # ---- Step 5: keep top-k groups ----
    #     k = int(drop_rate * group_scores.shape[1])   # keep 50% groups
    #     topk_vals, topk_idx = torch.topk(group_scores, k, dim=1, largest=False)  # [B,k]
    #     # print(topk_idx)

    #     group_mask = torch.zeros(B, 49, dtype=torch.int, device=x.device)
    #     group_mask.scatter_(1, topk_idx, 1)  # mark kept groups as 1

    #     # expand back to 196 tokens (4 per group)
    #     token_mask = group_mask.unsqueeze(-1).expand(-1, -1, 4).reshape(B, 196)  # [B,196]

    #     # add CLS (always kept)
    #     full_mask = torch.cat([torch.ones(B,1, dtype=torch.int, device=x.device), token_mask], dim=1)  # [B,197]

    #     # print("Mask (1=kept, 0=dropped):")
    #     # print(full_mask)

    #     # gather kept groups
    #     idx_expanded = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, D)  # [B,k,4,D]
    #     # print(idx_expanded)
    #     kept_groups = torch.gather(tokens_patches, 1, idx_expanded)   # [B,k,4,D]
    #     # print(kept_groups)

    #     kept_tokens = kept_groups.reshape(B, k*4, D)  # [B, kept*4, D]

    # # ---- Step 8: prepend CLS and append dropped avg ----
    #     # x_filtered = torch.cat([cls_token, kept_tokens, dropped_avg], dim=1) 

    #     # ---- Step 6: prepend CLS ----
    #     x_filtered = torch.cat([cls_token, kept_tokens], dim=1)  # [B, 1+kept*4, D]
    #     print(x_filtered.shape)

    #     return x_filtered

class CustomBlock(nn.Module):
    def __init__(self, block):
        super(CustomBlock, self).__init__()
        self.norm1 = block.norm1
        self.attn = CustomAttention(block.attn, block.attn.num_heads)
        self.ls1 = block.ls1
        self.drop_path1 = block.drop_path1
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.ls2 = block.ls2
        self.drop_path2 = block.drop_path2

    def forward(self, x, apply_patch_drop=False):

        
        x_residual = self.norm1(x)
        x_residual, mask_attention = self.attn(x_residual)
        x = x + self.drop_path1(self.ls1(x_residual))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        if apply_patch_drop:
            
            cls_token = x[:, 0].unsqueeze(1)  # Keep the class token
            other_tokens = x[:, 1:]
            selected_tokens = other_tokens[mask_attention[:, 1:]] 
            selected_tokens = selected_tokens.unsqueeze(0)
            x = torch.cat((cls_token, selected_tokens), dim=1)
            
        return x

class CustomVisionTransformer(VisionTransformer):
    def __init__(self, patch_drop_blocks=None, **kwargs):
        super(CustomVisionTransformer, self).__init__(**kwargs)
        self.patch_drop_blocks = patch_drop_blocks if patch_drop_blocks is not None else []
        self.patch_drop = PatchDropout()
        self.blocks = nn.Sequential(*[
            CustomBlock(block) if i in self.patch_drop_blocks else block
            for i, block in enumerate(self.blocks)
        ])

    def forward(self, x, mask=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # class token
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        if mask is not None:
            x = self.patch_drop(x, mask)

        for i, block in enumerate(self.blocks):
            if isinstance(block, CustomBlock):
                x = block(x, apply_patch_drop=True)
                # token_counts_per_image.append(x.shape[1])
            else:
                x = block(x)
                # token_counts_per_image.append(x.shape[1])
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

# Example usage
my_array = np.random.randint(1, 2, size=197)  # Updated to have values 0 or 1
custom_vit = CustomVisionTransformer(
    img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    norm_layer=nn.LayerNorm, num_classes=1000, patch_drop_blocks= patch_drop_blocks
).to(device)


# Load pretrained weights
# pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True)
custom_vit.load_state_dict(pretrained_vit.state_dict(), strict=True)

# Forward pass with input tensor and masks
input_tensor = torch.randn(1, 3, 224, 224).to(device)
output = custom_vit(input_tensor, mask=my_array)
print(output.shape)  # Should be [1, 1000]


# %% Original cell 16
from qtorch.quant import Quantizer, quantizer
from qtorch import FixedPoint
# Define the fixed-point quantizer
weight_quantizer = quantizer(forward_number=FixedPoint(16, 8), forward_rounding="nearest")
act_quantizer = quantizer(forward_number=FixedPoint(16, 8), forward_rounding="nearest")


# %% Original cell 17
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



# %% Original cell 18
def check_quantization(model):
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Data type: {param.dtype}")
        print(f"Values: {param.view(-1)[:10]}")  # Print the first 10 values
        print(f"Storage size (bytes): {param.numel() * param.element_size()}")
        print()



# Check quantized weights and biases
# check_quantization(quantized_model)


# %% Original cell 19
output = quantized_model(input_tensor, mask=my_array)


# %% Original cell 20
def main_processing(image):
    resized_image =  resize_nearest_neighbor(image, (224, 224))
    resized_image = bgr_to_rgb(resized_image)
    # gray_image = rgb_to_gray(resized_image)
    resized_image = resized_image.copy()                                        
    image_tensor= transform_image(resized_image, device)
    mask = np.random.randint(1, 2, size=197) 
    
    return image_tensor, mask


# %% Original cell 21



# %% Original cell 22
def save_percentage(file_path, average_percent1, average_percent2, average_percent3 ,average_percent4, average, count):
    with open(file_path, 'a') as file:
        file.write(f"Epoch: Average percentage_patch_drop: { average_percent1}, { average_percent2}, { average_percent3}, { average_percent4}, average: {average}, count{count}\n")


# %% Original cell 23
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
                    if image is None:
                        print(f"Failed to load image: {img_path}")
                        continue

                    image_tensor, fast_mask = main_processing(image)
                    
                    # outputs = custom_vit( image_tensor, mask= fast_mask)
                    outputs= quantized_model(image_tensor, mask= fast_mask)
                    predicted_idx = torch.argmax(outputs, dim=1).item()
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
                    if processed_images % 10 == 0:
                        yes= ( sum(token_counts_per_image)/ len(token_counts_per_image))
                        print(len(token_counts_per_image))
                        print(yes)
                        # w= ( sum(scan)/ len(scan))
                        accuracy_so_far = (correct_predictions / processed_images) * 100
                        # average_percent= 15_15_15_15
                        w=0
                        # average_percent = sum(patch_drop_percentage) / len(patch_drop_percentage)
                        # # print("average percentage embedding_drop " , average_percent , len(patch_drop_percentage))
                        print(f"Accuracy after processing {processed_images} images: {accuracy_so_far:.2f}%")
                        save_average_percentage(txt_path, attention_pruning, accuracy_so_far, processed_images, w )
                    
                    
                    


# %% Original cell 24
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# Example usage
print(device)


# %% Original cell 25
convert_to_canvas(img_folder, pretrained_vit, device)


# %% Original cell 26

