from torchvision import datasets, transforms
import torch 

def get_dataloaders(data_dir="data", batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
    val_ds   = datasets.ImageFolder(f"{data_dir}/val", transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_ds.classes
