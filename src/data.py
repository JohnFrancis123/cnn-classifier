import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _build_transforms() -> tuple[transforms.Compose, transforms.Compose]: #Keep augmentation on training only.
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])
    return train_transform, eval_transform


def _split_indices(dataset_size: int, seed: int) -> tuple[list[int], list[int]]: #Use a reproducible 80/20 split.
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_size = int(0.8 * dataset_size)
    return indices[:train_size], indices[train_size:]


def get_dataloaders( #Build CIFAR-10 train, validation, and test loaders.
    data_dir: str,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform, eval_transform = _build_transforms()
    train_dataset = datasets.CIFAR10( #Training split keeps augmentation enabled.
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    val_dataset = datasets.CIFAR10( #Validation reuses the training split with eval transforms.
        root=data_dir,
        train=True,
        download=True,
        transform=eval_transform,
    )
    test_dataset = datasets.CIFAR10( #Test data always uses eval transforms.
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    train_indices, val_indices = _split_indices(len(train_dataset), seed)
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    return (
        DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True, #Shuffle only the training loader.
            num_workers=num_workers,
        ),
        DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )