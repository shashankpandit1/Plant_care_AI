import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ----------------------------
# Load and preprocess dataset
# ----------------------------
def get_dataloaders(data_dir="Medicinal_Plants", batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_ds.classes

# ----------------------------
# Train the model
# ----------------------------
def train_model(data_dir="Medicinal_Plants", epochs=10, batch_size=32, lr=1e-4, save_path="model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Training on: {device}")

    train_loader, val_loader, class_names = get_dataloaders(data_dir, batch_size)
    num_classes = len(class_names)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        acc = correct / len(train_loader.dataset)
        print(f"✅ Epoch {epoch+1}: Loss = {running_loss:.4f} | Accuracy = {acc:.4f}")

    # Save model
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "plant_classifier.pth"))

    # Save class names
    with open(os.path.join(save_path, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    print(f"🎉 Model and class names saved to '{save_path}/'")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    train_model()
