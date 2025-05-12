import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset import get_dataloaders
from tqdm import tqdm
import os

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, val_loader, class_names = get_dataloaders()

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

        acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss:.4f} - Train Acc: {acc:.4f}")

    os.makedirs("model", exist_ok=True)
    torch.save(model, "model/plant_classifier.pth")
    print("✅ Model saved to model/plant_classifier.pth")
    return class_names
    
if __name__ == "__main__":
    train_model()
