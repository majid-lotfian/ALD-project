from __future__ import annotations

from torch.utils.data import DataLoader


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
