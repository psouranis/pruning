import torch
import torch.nn as nn

from tqdm import tqdm
from torch import Tensor
from collections.abc import Mapping
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader


def finetune(
    model: nn.Module,
    dataloader: DataLoader[Mapping[str, Tensor]],
    device: str | torch.device | None = None,
    epochs: int = 10,
    lr: float = 2e-5,
    weight_decay: float = 1e-4,
    masks: dict[str, Tensor] | None = None,
) -> nn.Module:
    """Finetune the model

    Args:
        model (torch.nn.Module): Model to finetune
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset
        device (str): Device to run the model on
        epochs (int): Number of epochs to finetune
        lr (float): Learning rate
        weight_decay (float): Weight decay
        masks (dict): Masks for the model

    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        for batch in tqdm(dataloader, total=len(dataloader)):
            images, labels = (
                batch["pixel_values"].to(device),
                batch["labels"].to(device),
            )

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output.logits, labels)
            loss.backward()
            optimizer.step()

        if masks:
            with torch.no_grad():
                for name, module in model.named_modules():
                    if name in masks:
                        module.weight.data.mul_(masks[name])

    model.eval()
    return model


def evaluate(
    model: nn.Module,
    dataloader: DataLoader[Mapping[str, Tensor]],
    device: str | torch.device | None = None,
) -> float:
    """Evaluate the model on the dataset

    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset
        device (str): Device to run the model on

    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    preds, actual = [], []

    for batch in tqdm(dataloader, total=len(dataloader)):
        images, labels = batch["pixel_values"].to(device), batch["labels"].to(device)

        actual.extend(labels.tolist())

        with torch.no_grad():
            p = model(images).logits.argmax(dim=1)
            preds.extend(p.tolist())

    acc = accuracy_score(actual, preds)
    print(f"Accuracy score achieved: {acc * 100}\n", "=" * 50, sep="")

    return acc
