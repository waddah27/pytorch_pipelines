import sys
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from mri_dataset import MRI_Dataset
from torchvision.models import mobilenet_v2

from model_train import model_trainer
data_dir = 'D:\Job\Other\pytorch\Alzheimer-MRI'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
lr = 0.001
# Image preprocessing modules
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])
train_data = MRI_Dataset(root_dir=data_dir, train=True, transform=transform)
test_data = MRI_Dataset(root_dir=data_dir, train=False, transform=transform)

train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
print(len(test_data_loader))
for img, label in test_data_loader:
    print(img.shape)
    break

model = mobilenet_v2(pretrained=True).to(DEVICE)
for param in model.features[:17]:
    param.requires_grad = False
print(model)
model.classifier[1] = torch.nn.Sequential(
    torch.nn.Linear(1280, 640),
    torch.nn.ReLU6(),
    torch.nn.Linear(640, 4)
    ).to(DEVICE)

print(model)

# model training
trainer = model_trainer(model, model_name="mobilenet", n_epochs=1, lr=0.001, is_rnn=False, device=DEVICE)
trainer.train(loaders={"train": train_data_loader, "test": test_data_loader}, load_checkpoint=False)