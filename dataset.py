from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_dataloaders(
    train_dir=r"C:\Users\kirolosAnwar\Desktop\AMIT\data\train", 
    test_dir=r"C:\Users\kirolosAnwar\Desktop\AMIT\data\test", 
    batch_size=16, 
    fraction=1 # 👈 نسبة الداتا اللي تستخدمها (1.0 يعني 100%)
):
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ⬅️ هنا كان الخطأ
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset  = datasets.ImageFolder(test_dir, transform=transform)

    # 👇 ناخد عينة من الداتا لو fraction < 1.0
    if fraction < 1.0:
        n_samples = int(len(train_dataset) * fraction)
        indices = np.random.choice(len(train_dataset), n_samples, replace=False)
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # لو dataset متغّير لـ Subset، لازم نجيب الكلاسات من الـ dataset الأصلي
    classes = train_dataset.dataset.classes if isinstance(train_dataset, Subset) else train_dataset.classes

    return train_loader, test_loader, classes
