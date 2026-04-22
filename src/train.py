import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import Config
from src.data import get_dataloaders
from src.model import SimpleCNN

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None: #Make training runs more repeatable.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _step( #Handle one training or evaluation pass.
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
) -> tuple[float, float]:
    total_loss, correct, total = 0.0, 0, 0
    model.train() if optimizer else model.eval()
    with torch.set_grad_enabled(optimizer is not None): #Skip gradient tracking during eval.
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def run_training(cfg: Config) -> None: #Train the model and save the best checkpoint.
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | Epochs: %d | LR: %.4f", device, cfg.epochs, cfg.lr)

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    train_ld, val_ld, _ = get_dataloaders(cfg.data_dir, cfg.batch_size, cfg.num_workers, cfg.seed) #Training ignores the test loader.

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = _step(model, train_ld, criterion, optimizer, device)
        vl_loss, vl_acc = _step(model, val_ld, criterion, None, device)
        logger.info(
            f"Epoch {epoch}/{cfg.epochs} | Tr: {tr_loss:.4f}({tr_acc:.3f}) | Vl: {vl_loss:.4f}({vl_acc:.3f})"
        )

        if vl_acc > best_val_acc: #Only write improved checkpoints.
            best_val_acc = vl_acc
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": vl_acc},
                Path(cfg.checkpoint_dir) / "best.pt",
            )
            logger.info("Best checkpoint saved. GOOD.")