import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

SAVE_PATH = 'saved_models/maize_resnet50.pth'
DATA_DIR = 'data/maize/test'

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

test_ds = datasets.ImageFolder(DATA_DIR, transform)
test_loader = DataLoader(test_ds, batch_size=32)
checkpoint = torch.load(SAVE_PATH, map_location='cpu')
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint['classes']))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = model(imgs)
        preds = outputs.argmax(1)
        y_true += labels.tolist()
        y_pred += preds.tolist()

print(classification_report(y_true, y_pred, target_names=test_ds.classes))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, xticklabels=test_ds.classes, yticklabels=test_ds.classes)
plt.title('Confusion Matrix')
plt.show()


