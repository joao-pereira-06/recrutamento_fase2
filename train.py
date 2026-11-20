import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Importa a arquitetura diretamente do pacote ROS 2 para garantir 100% de compatibilidade
sys.path.append(os.path.join(os.path.dirname(__file__), "ros2_ws/src/tsb_cv_brain"))
from tsb_cv_brain.vision_node import TurtleVisionCNN


def main():
    data_dir = "archive/Direction"
    save_path = "models/arrow_cnn.pth"
    batch_size = 32
    epochs = 12
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A treinar no dispositivo: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print("Mapeamento de classes:", full_dataset.class_to_idx)

    # Divisão 80% treino / 20% validação
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = TurtleVisionCNN(num_classes=len(full_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- TREINO ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        # --- VALIDAÇÃO ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Loss: {train_loss / train_total:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": full_dataset.class_to_idx,
                },
                save_path,
            )

    print(f"\nTreino concluído! Melhor Val Acc: {best_val_acc:.2f}% -> Guardado em {save_path}")


if __name__ == "__main__":
    main()
