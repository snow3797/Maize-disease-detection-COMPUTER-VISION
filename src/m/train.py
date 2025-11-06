import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np, os

# Paths
DATA_DIR = 'data/maize'
SAVE_PATH = 'saved_models/maize_resnet50.pth'

# Transformations
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_tfms)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=test_tfms)
test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=test_tfms)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_classes = len(train_ds.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

train_losses, val_losses = [], []

for epoch in range(8):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            val_loss += criterion(model(imgs), labels).item()
    val_losses.append(val_loss / len(val_loader))
    print(f"Epoch {epoch+1}: Train loss {train_losses[-1]:.4f}, Val loss {val_losses[-1]:.4f}")

# Save model
torch.save({'model_state_dict': model.state_dict(), 'classes': train_ds.classes}, SAVE_PATH)

# Plot losses
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Training Curve')
plt.savefig('loss_curve.png')
plt.show()
