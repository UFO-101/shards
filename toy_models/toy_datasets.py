#%%
import random
from dataclasses import dataclass
from typing import Literal

import torch as t
from rich.pretty import pretty_repr
from torch.utils.data import Dataset, DataLoader

@dataclass
class FeatureSet:
    features: list[tuple[int, int]]
    sparsity: float = 0.0
    type: Literal["correlated", "anti-correlated"] = "correlated"

    @staticmethod
    def within_range(
        feature_count: int,
        sparsity: float = 0.0,
        feature_range: tuple[int, int] = (0, 1),
        type: Literal["correlated", "anti-correlated"] = "correlated",
    ):
        return FeatureSet([feature_range for _ in range(feature_count)], sparsity, type)

    def generate(self):
        if not random.random() > self.sparsity:
            return [0 for _ in self.features]
        if self.type == "correlated":
            return [random.random() * (b - a) + a for a, b in self.features]
        else:
            active_feature = random.choice(list(range(len(self.features))))
            return [
                random.random() * (b - a) + a if idx == active_feature else 0
                for idx, (a, b) in enumerate(self.features)
            ]

class CombinationDataset(Dataset):
    def __init__(
        self,
        feature_sets: list[FeatureSet],
        size: int = 100,
    ):
        data = [
            [feature for fs in feature_sets for feature in fs.generate()]
            for _ in range(size)
        ]
        self.feature_sets = feature_sets
        self.data = t.tensor(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    # def __str__(self):
    #     return (
    #         f"CombinationDataset with {len(self)} samples, and features:"
    #         + pretty_repr(self.feature_sets)
    #     )

    def describe(self):
        description = " × ".join(
            [
                ("F" if len(fs.features) == 1 else f"({len(fs.features)} {fs.type})")
                for fs in self.feature_sets
            ]
        )
        return f"[{description}]"
