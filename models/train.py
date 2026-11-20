import os
import sys
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# 1. Caminhos do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "archive"  # Ajusta se as pastas estiverem numa subpasta de archive/
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "arrow_cnn.pth"
PLOT_SAVE_PATH = PROJECT_ROOT / "models" / "training_metrics.png"

# Importar a TurtleVisionCNN do teu pacote ROS 2 para garantir compatibilidade total de pesos
sys.path.insert(0, str(PROJECT_ROOT / "ros2_ws" / "src" / "tsb_cv_brain"))
from tsb_cv_brain.vision_node import TurtleVisionCNN  # Se a classe estiver noutro ficheiro (ex: model.py), ajusta aqui

# Ordem alfabética padrão: 0: Down (BACKWARD), 1: Left (LEFT), 2: Right (RIGHT), 3: Up (FORWARD)
CLASSES = ["Down", "Left", "Right", "Up"]


# 2. Custom Dataset
class ArrowDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def collect_samples(root_dir: Path):
    samples = []
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = root_dir / class_name
        if not class_dir.exists():
            # Procura recursivamente caso exista uma pasta intermédia dentro de archive/
            matches = list(root_dir.rglob(class_name))
            class_dir = matches[0] if matches else class_dir

        if not class_dir.exists():
            raise FileNotFoundError(f"Pasta da classe '{class_name}' não encontrada em {root_dir}")

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in valid_exts:
                samples.append((img_path, label_idx))
    return samples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A treinar no dispositivo: {device}")

    # 3. Recolher amostras e dividir 80% treino / 20% validação
    all_samples = collect_samples(DATASET_DIR)
    print(f"Total de imagens encontradas: {len(all_samples)}")

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(all_samples), generator=generator).tolist()
    split = int(0.8 * len(all_samples))

    train_samples = [all_samples[i] for i in indices[:split]]
    val_samples = [all_samples[i] for i in indices[split:]]

    # 4. Transformações (Data Augmentation apenas no treino; NUNCA usar Flip!)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = ArrowDataset(train_samples, transform=train_transform)
    val_dataset = ArrowDataset(val_samples, transform=val_transform)

    # 5. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # 6. Modelo, Loss e Otimizador
    model = TurtleVisionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # 7. Loop de Treino e Validação
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = 100.0 * correct / total

        # Validação
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%"
        )

    # 8. Guardar os pesos treinados (.pth)
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nPesos guardados com sucesso em: {MODEL_SAVE_PATH}")

    # 9. Bónus: Gerar e guardar gráficos de Loss e Accuracy
    epoch_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 4.5))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, train_accs, marker="o", label="Train Accuracy")
    plt.plot(epoch_range, val_accs, marker="s", label="Validation Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epoch_range, train_losses, marker="o", label="Train Loss")
    plt.plot(epoch_range, val_losses, marker="s", label="Validation Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=300)
    print(f"Gráficos de métricas guardados em: {PLOT_SAVE_PATH}")


if __name__ == "__main__":
    main()