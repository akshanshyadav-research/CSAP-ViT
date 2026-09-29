# Exported from experiments/deit_tiny_similarity/Untitled.ipynb
# Historical code: review paths and side effects before executing.


# %% Original cell 0
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained MobileViT-XXS
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.to(device)
model.eval()

# ImageNet mean and std
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Data transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Load ImageNet validation dataset
imagenet_val_path = "/home/viraj/my_project/val/val2/"  # Change this to your path
val_dataset = datasets.ImageFolder(imagenet_val_path, transform=transform)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Evaluation
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"\n✅ Top-1 Accuracy of MobileViT-XXS on ImageNet validation set: {accuracy:.2f}%")
