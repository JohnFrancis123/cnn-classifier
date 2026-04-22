import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage

from src import data as data_module


class FakeCIFAR10(Dataset): #Tiny in-memory stand-in for CIFAR-10.
    def __init__(
        self,
        root: str,
        train: bool,
        download: bool,
        transform=None,
    ) -> None:
        del root, download #Unused in the test double.
        self.length = 16 if train else 8
        self.transform = transform
        self.to_pil = ToPILImage()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = torch.full((3, 32, 32), index % 255, dtype=torch.uint8)
        label = index % 10
        sample = self.to_pil(image)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label


def test_loader_shapes(monkeypatch, tmp_path): #Loaders should emit the expected batch shapes.
    monkeypatch.setattr(data_module.datasets, "CIFAR10", FakeCIFAR10)

    train_loader, val_loader, test_loader = data_module.get_dataloaders(
        str(tmp_path),
        4,
        0,
        42,
    )

    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))
    test_images, test_labels = next(iter(test_loader))

    assert train_images.shape == (4, 3, 32, 32)
    assert train_labels.shape == (4,)
    assert val_images.shape == (4, 3, 32, 32)
    assert val_labels.shape == (4,)
    assert test_images.shape == (4, 3, 32, 32)
    assert test_labels.shape == (4,)


def test_loader_split_is_deterministic(monkeypatch, tmp_path): #The split should stay stable for a fixed seed.
    monkeypatch.setattr(data_module.datasets, "CIFAR10", FakeCIFAR10)

    train_loader_a, val_loader_a, _ = data_module.get_dataloaders(
        str(tmp_path),
        4,
        0,
        42,
    )
    train_loader_b, val_loader_b, _ = data_module.get_dataloaders(
        str(tmp_path),
        4,
        0,
        42,
    )

    assert train_loader_a.dataset.indices == train_loader_b.dataset.indices
    assert val_loader_a.dataset.indices == val_loader_b.dataset.indices