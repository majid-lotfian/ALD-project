from __future__ import annotations

from torch.utils.data import DataLoader, IterableDataset


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
):
    is_iterable = isinstance(dataset, IterableDataset)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    if is_iterable:
        if shuffle:
            # Shuffling must be handled inside the IterableDataset itself.
            # Do not pass shuffle=True to DataLoader for streaming datasets.
            pass
        return DataLoader(
            dataset,
            **loader_kwargs,
        )

    return DataLoader(
        dataset,
        shuffle=shuffle,
        **loader_kwargs,
    )