# Exported from experiments/vlsid results/drop_1_15/output/notebook_87.ipynb
# Historical code: review paths and side effects before executing.


# %% Original cell 0
keep_rate = 87


# %% Original cell 1
patch_drop_blocks=[]
attention_pruning= 0.00


# %% Original cell 2
keep_rate= keep_rate/100


# %% Original cell 3
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


# %% Original cell 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
img_folder = "/home/viraj/my_project/val/val2/"
txt_path = f"vit_base_similarity_dropblock{patch_drop_blocks}_droprate{keep_rate:.2f}_attnprune{attention_pruning:.2f}.txt"
json_file = "/home/viraj/my_project/imagenet_class_index.json"
# line_r=0
 


# %% Original cell 5
# pretrained_vit = vit_large_patch16_224(pretrained=True).to(device)

pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)


# %% Original cell 6
def plot(image):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.show()


# %% Original cell 7
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


# %% Original cell 8
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


# %% Original cell 9
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


# %% Original cell 10
def save_average_percentage(file_path, average_percent, accuracy, processed_image, w):
    with open(file_path, 'a') as file:
        file.write(f"Epoch: Average percentage_patch_drop: {average_percent}, Accuracy: {accuracy}, processed_image: {processed_image}, scan: {w}\n")


# %% Original cell 11
def load_mapping_file(json_file):
    with open(json_file, 'r') as f:
        mapping_data = json.load(f)
    return mapping_data


# %% Original cell 12
import torchvision.transforms as transforms
def transform_image(image, device):

    image=transforms.ToTensor()(image)
    image= image.unsqueeze(0)
    image=image.to(device)    
    

    
    return image


# %% Original cell 13
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


# %% Original cell 14
patch_drop_percentage = []
def average_patch_drop(mask):
    count_ones = mask.count(1)
    count_ones=count_ones-1
    y = (((196 - count_ones ) / 196) * 100)
    patch_drop_percentage.append(y)
    


# %% Original cell 15
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


# %% Original cell 16
# token_counts_per_image = []


# %% Original cell 17
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

    # Compute Pearson correlation (dot product of standardized vectors)
    corr = torch.einsum('bd,bnd->bn', cls_norm, tokens_norm) / D  # (B, N)

    return corr


# %% Original cell 18
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
        # self.drop_rate = drop_rate  # Drop 20% of the tokens

    def forward(self, x, mask):
        batch_size, seq_len, embedding_dim = x.shape
        assert seq_len == 197, f"Expected sequence length of 197, but got {seq_len}"
        
        # Extract CLS token
        cls_token = x[:, 0, :]  # (B, D)
        tokens = x[:, 1:, :]    # (B, 196, D)
    
        # Recompute inverse covariance matrix from all tokens (excluding CLS)
        # inv_cov = compute_inv_covariance(tokens)  # (D, D)
    
        # Compute Mahalanobis distance between each token and CLS
        distances = pearson_corr(cls_token, tokens)  # (B, 196)
        
        # Keep bottom 80% farthest tokens (drop closest 20%)
        k = int((seq_len - 1) * keep_rate)  # 20% of 196 tokens
        threshold_values, _ = torch.kthvalue(distances, k + 1, dim=1, keepdim=True)  # (B, 1)
    
        keep_mask = distances <= threshold_values  # (B, 196)
        keep_mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=x.device), keep_mask], dim=1)  # (B, 197)
    
        # Apply mask to x
        x_filtered = x[keep_mask].view(batch_size, -1, embedding_dim)
        # print(x_filtered.shape)
        # token_counts_per_image.append(x_filtered.shape[1])
        
        return x_filtered

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


# %% Original cell 19
from qtorch.quant import Quantizer, quantizer
from qtorch import FixedPoint
# Define the fixed-point quantizer
weight_quantizer = quantizer(forward_number=FixedPoint(16, 8), forward_rounding="nearest")
act_quantizer = quantizer(forward_number=FixedPoint(16, 8), forward_rounding="nearest")


# %% Original cell 20
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



# %% Original cell 21
def check_quantization(model):
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Data type: {param.dtype}")
        print(f"Values: {param.view(-1)[:10]}")  # Print the first 10 values
        print(f"Storage size (bytes): {param.numel() * param.element_size()}")
        print()



# Check quantized weights and biases
# check_quantization(quantized_model)


# %% Original cell 22
output = quantized_model(input_tensor, mask=my_array)


# %% Original cell 23
def main_processing(image):
    resized_image =  resize_nearest_neighbor(image, (224, 224))
    resized_image = bgr_to_rgb(resized_image)
    # gray_image = rgb_to_gray(resized_image)
    resized_image = resized_image.copy()                                        
    image_tensor= transform_image(resized_image, device)
    mask = np.random.randint(1, 2, size=197) 
    
    return image_tensor, mask


# %% Original cell 24
token_counts_per_image = []


# %% Original cell 25
def save_percentage(file_path, average_percent1, average_percent2, average_percent3 ,average_percent4, average, count):
    with open(file_path, 'a') as file:
        file.write(f"Epoch: Average percentage_patch_drop: { average_percent1}, { average_percent2}, { average_percent3}, { average_percent4}, average: {average}, count{count}\n")


# %% Original cell 26
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
                    if processed_images % 1000 == 0:
                        # yes= ( sum(token_counts_per_image)/ len(token_counts_per_image))
                        # print(len(token_counts_per_image))
                        # print(yes)
                        # w= ( sum(scan)/ len(scan))
                        accuracy_so_far = (correct_predictions / processed_images) * 100
                        # average_percent= 15_15_15_15
                        w=0
                        # average_percent = sum(patch_drop_percentage) / len(patch_drop_percentage)
                        # # print("average percentage embedding_drop " , average_percent , len(patch_drop_percentage))
                        print(f"Accuracy after processing {processed_images} images: {accuracy_so_far:.2f}%")
                        save_average_percentage(txt_path, attention_pruning, accuracy_so_far, processed_images, w )
                    
                    
                    


# %% Original cell 27
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# Example usage
print(device)


# %% Original cell 28
convert_to_canvas(img_folder, pretrained_vit, device)


# %% Original cell 29

