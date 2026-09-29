# Exported from experiments/vlsid results/Untitled.ipynb
# Historical code: review paths and side effects before executing.


# %% Original cell 0
import torch
import timm
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ========== 1. Load Pretrained ViT ==========
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()
model.cuda()

# ========== 2. Preprocess the Input Image ==========
img_path = '/home/viraj/my_project/val/val2/n01440764/ILSVRC2012_val_00000293.JPEG'  # 🔁 Replace with your image path
image = Image.open(img_path).convert('RGB')

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

img_tensor = transform(image).unsqueeze(0).cuda()
img_tensor.requires_grad = True

# ========== 3. Capture Patch Embeddings + Retain Gradient ==========
tokens = {}

def hook_tokens(module, input, output):
    output.retain_grad()
    tokens['embeddings'] = output

hook_handle = model.patch_embed.register_forward_hook(hook_tokens)

# ========== 4. Forward and Backward ==========
output = model(img_tensor)
hook_handle.remove()

target_class = output.argmax(dim=1).item()
logit = output[0, target_class]

model.zero_grad()
logit.backward()

# ========== 5. Compute Gradient × Input ==========
embeddings = tokens['embeddings']        # [1, num_tokens, dim]
grads = embeddings.grad                  # [1, num_tokens, dim]

with torch.no_grad():
    attribution = (grads * embeddings).sum(dim=-1).squeeze(0).cpu().numpy()  # [num_tokens]

# ========== 6. Prepare Heatmap ==========
patch_attribution = attribution[1:]  # Remove CLS token (at index 0)

# Normalize
patch_attribution -= patch_attribution.min()
patch_attribution /= patch_attribution.max()

# Reshape to 14x14 patch grid
heatmap = patch_attribution.reshape(14, 14)

# Upsample heatmap to match 224x224 image
heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
heatmap_tensor = torch.nn.functional.interpolate(heatmap_tensor, size=(224, 224), mode='bilinear', align_corners=False)
heatmap = heatmap_tensor.squeeze().numpy()

# ========== 7. Plot Original and Heatmap ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original image
axes[0].imshow(image.resize((224, 224)))
axes[0].set_title("Original Image")
axes[0].axis("off")

# Heatmap overlay
axes[1].imshow(image.resize((224, 224)))
axes[1].imshow(heatmap, cmap='hot', alpha=0.5)
axes[1].set_title("Token Importance Heatmap")
axes[1].axis("off")

plt.tight_layout()
plt.show()


# %% Original cell 1
import torch
import timm
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ========= 1. Load Pretrained ViT Model from timm ==========
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval().cuda()

# ========= 2. Load and Preprocess Input Image ==========
img_path = '/home/viraj/my_project/val/val2/n01440764/ILSVRC2012_val_00000293.JPEG'  # 🔁 Replace this with your image path
image = Image.open(img_path).convert('RGB')

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

img_tensor = transform(image).unsqueeze(0).cuda()
img_tensor.requires_grad = True

# ========= 3. Register Hook to Capture Patch Embeddings and Retain Grad ==========
tokens = {}

def hook_tokens(module, input, output):
    output.retain_grad()
    tokens['embeddings'] = output

hook_handle = model.patch_embed.register_forward_hook(hook_tokens)

# ========= 4. Forward and Backward Pass ==========
output = model(img_tensor)
hook_handle.remove()

target_class = output.argmax(dim=1).item()
logit = output[0, target_class]

model.zero_grad()
logit.backward()

# ========= 5. Compute Gradient × Input Attribution ==========
embeddings = tokens['embeddings']        # shape: [1, num_tokens, dim]
grads = embeddings.grad                  # shape: [1, num_tokens, dim]

with torch.no_grad():
    attribution = (grads * embeddings).sum(dim=-1).squeeze(0).cpu().numpy()  # [num_tokens]

# ========= 6. Extract Patch Token Attribution (Exclude CLS Token) ==========
num_tokens = len(attribution)
if num_tokens == 197:
    patch_attribution = attribution[1:]  # [1:197] → 196 tokens
elif num_tokens == 196:
    patch_attribution = attribution      # already CLS removed
elif num_tokens == 195:
    print("⚠️ Warning: Only 195 tokens found — check model/token slicing.")
    patch_attribution = attribution
else:
    raise ValueError(f"❌ Unexpected number of tokens: {num_tokens}")

# ========= 7. Normalize and Reshape Heatmap ==========
patch_attribution -= patch_attribution.min()
patch_attribution /= patch_attribution.max()

grid_size = int(np.sqrt(patch_attribution.shape[0]))
assert grid_size * grid_size == patch_attribution.shape[0], \
    f"Cannot reshape {patch_attribution.shape[0]} tokens into square grid"

heatmap = patch_attribution.reshape(grid_size, grid_size)

# Upsample to 224x224
heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
heatmap_tensor = torch.nn.functional.interpolate(
    heatmap_tensor, size=(224, 224), mode='bilinear', align_corners=False
)
heatmap = heatmap_tensor.squeeze().numpy()

# ========= 8. Plot Input and Overlayed Heatmap ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Left: Original image
axes[0].imshow(image.resize((224, 224)))
axes[0].set_title("Original Image")
axes[0].axis("off")

# Right: Heatmap overlay
axes[1].imshow(image.resize((224, 224)))
axes[1].imshow(heatmap, cmap='hot', alpha=0.5)
axes[1].set_title("Token Importance (Grad × Input)")
axes[1].axis("off")

plt.tight_layout()
plt.show()


# %% Original cell 2
import torch
import timm
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ========= 1. Load Pretrained ViT Model ==========
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval().cuda()

# ========= 2. Load and Preprocess Input Image ==========
img_path = '/home/viraj/my_project/val/val2/n01440764/ILSVRC2012_val_00000293.JPEG'  # 🔁 Replace with your image path
image = Image.open(img_path).convert('RGB')

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

img_tensor = transform(image).unsqueeze(0).cuda()
img_tensor.requires_grad = True

# ========= 3. Hook to Capture Patch Embeddings ==========
tokens = {}

def hook_tokens(module, input, output):
    output.retain_grad()
    tokens['embeddings'] = output

hook_handle = model.patch_embed.register_forward_hook(hook_tokens)

# ========= 4. Forward + Backward ==========
output = model(img_tensor)
hook_handle.remove()

target_class = output.argmax(dim=1).item()
logit = output[0, target_class]

model.zero_grad()
logit.backward()

# ========= 5. Gradient × Input Attribution ==========
embeddings = tokens['embeddings']        # [1, num_tokens, dim]
grads = embeddings.grad                  # [1, num_tokens, dim]

with torch.no_grad():
    attribution = (grads * embeddings).sum(dim=-1).squeeze(0).cpu().numpy()

# ========= 6. Extract Patch Attribution (exclude CLS token) ==========
if len(attribution) == 197:
    patch_attribution = attribution[1:]
else:
    raise ValueError(f"Expected 197 tokens, got {len(attribution)}")

# Normalize attribution
patch_attribution -= patch_attribution.min()
patch_attribution /= patch_attribution.max()

# Reshape into grid
grid_size = 14
heatmap = patch_attribution.reshape(grid_size, grid_size)

# ========= 7. Plot with Text on Patches ==========
fig, ax = plt.subplots(figsize=(8, 8))
image_resized = image.resize((224, 224))
ax.imshow(image_resized)

patch_size = 16  # 224 / 14 = 16 pixels

for i in range(grid_size):
    for j in range(grid_size):
        # Patch position
        x = j * patch_size
        y = i * patch_size

        score = heatmap[i, j]
        color = 'white' if score < 0.5 else 'black'

        # Rectangle border (optional)
        rect = plt.Rectangle((x, y), patch_size, patch_size, fill=False, edgecolor='red', linewidth=0.5)
        ax.add_patch(rect)

        # Importance text
        ax.text(x + patch_size // 2, y + patch_size // 2, f"{score:.2f}",
                ha='center', va='center', fontsize=7, color=color)

ax.set_title("Patch-wise Token Importance (Gradient × Input)")
ax.axis('off')
plt.tight_layout()
plt.show()


# %% Original cell 3
import torch
import timm
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ========= 1. Load Pretrained ViT Model ==========
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval().cuda()

# ========= 2. Load and Preprocess Input Image ==========
img_path = '/home/viraj/my_project/val/val2/n01580077/ILSVRC2012_val_00003386.JPEG'  # 🔁 Replace this with your image path
image = Image.open(img_path).convert('RGB')

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

img_tensor = transform(image).unsqueeze(0).cuda()
img_tensor.requires_grad = True

# ========= 3. Hook to Capture Patch Embeddings ==========
tokens = {}

def hook_tokens(module, input, output):
    output.retain_grad()
    tokens['embeddings'] = output

hook_handle = model.patch_embed.register_forward_hook(hook_tokens)

# ========= 4. Forward + Backward ==========
output = model(img_tensor)
hook_handle.remove()

target_class = output.argmax(dim=1).item()
logit = output[0, target_class]

model.zero_grad()
logit.backward()

# ========= 5. Gradient × Input Attribution ==========
embeddings = tokens['embeddings']        # [1, num_tokens, dim]
grads = embeddings.grad                  # [1, num_tokens, dim]

with torch.no_grad():
    attribution = (grads * embeddings).sum(dim=-1).squeeze(0).cpu().numpy()

# ========= 6. Extract Patch Attribution (exclude CLS token if needed) ==========
if len(attribution) == 197:
    patch_attribution = attribution[1:]
elif len(attribution) == 196:
    patch_attribution = attribution
else:
    raise ValueError(f"Expected 196 or 197 tokens, got {len(attribution)}")

# Normalize attribution scores
patch_attribution -= patch_attribution.min()
patch_attribution /= patch_attribution.max()

# Reshape to grid
grid_size = int(np.sqrt(patch_attribution.shape[0]))
assert grid_size * grid_size == patch_attribution.shape[0], \
    f"Cannot reshape {patch_attribution.shape[0]} tokens into square grid"

heatmap = patch_attribution.reshape(grid_size, grid_size)

# ========= 7A. Heatmap Overlay Visualization ==========
# Upsample to 224x224
heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
heatmap_tensor = torch.nn.functional.interpolate(
    heatmap_tensor, size=(224, 224), mode='bilinear', align_corners=False
)
heatmap_resized = heatmap_tensor.squeeze().numpy()

# Show side-by-side: Original + Heatmap Overlay
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Left: Original image
axes[0].imshow(image.resize((224, 224)))
axes[0].set_title("Original Image")
axes[0].axis("off")

# Right: Heatmap over image
axes[1].imshow(image.resize((224, 224)))
axes[1].imshow(heatmap_resized, cmap='hot', alpha=0.5)
axes[1].set_title("Token Importance (Heatmap)")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# ========= 7B. Patch-wise Importance Visualization (Text Overlay) ==========
fig, ax = plt.subplots(figsize=(8, 8))
image_resized = image.resize((224, 224))
ax.imshow(image_resized)

patch_size = 224 // grid_size  # Should be 16

for i in range(grid_size):
    for j in range(grid_size):
        x = j * patch_size
        y = i * patch_size
        score = heatmap[i, j]
        color = 'white' if score < 0.5 else 'black'

        # Draw red rectangle
        rect = plt.Rectangle((x, y), patch_size, patch_size, fill=False, edgecolor='red', linewidth=0.5)
        ax.add_patch(rect)

        # Draw importance score
        ax.text(x + patch_size // 2, y + patch_size // 2, f"{score:.2f}",
                ha='center', va='center', fontsize=7, color=color)

ax.set_title("Patch-wise Token Importance (Text Overlay)")
ax.axis('off')
plt.tight_layout()
plt.show()
