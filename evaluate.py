import torch
from dataset import get_dataloaders
from model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model_path=r"C:\Users\kirolosAnwar\Desktop\AMIT\models\sign_model_epoch3.pth"):
    train_loader, test_loader, classes = get_dataloaders()
    checkpoint = torch.load(model_path, map_location=device)

    model = build_model(len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Accuracy: {acc:.2f}%")
    return acc

if __name__ == "__main__":
    evaluate()
