import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_dataloaders
from model import build_model
from tqdm import tqdm   # ✅ progress bar
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, test_loader, classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    return precision, recall, f1, acc


def train_model(epochs=1, save_dir="models"):
    train_loader, test_loader, classes = get_dataloaders()
    model = build_model(len(classes)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        # ✅ Evaluate
        precision, recall, f1, acc = evaluate_model(model, test_loader, classes)
        print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        # ✅ Save model
        save_path = os.path.join(save_dir, f"sign_model_epoch{epoch+1}.pth")
        torch.save({"model_state": model.state_dict(), "classes": classes}, save_path)
        print(f"Model saved to {save_path}\n")


if __name__ == "__main__":
    train_model(epochs=5, save_dir="models")
