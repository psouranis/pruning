import torch
import argparse
import os
import torch.nn as nn
import numpy as np

from functools import partial
from inspect import getfullargspec
from transformers import AutoModelForImageClassification
from torchvision import transforms
from typing import Any
from torchprofile import profile_macs
from torch.utils.data import DataLoader
from collections.abc import Mapping, Iterable
from classes import IMAGENET2012_CLASSES, classes_to_idx
from torch import Tensor
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from torch.nn.utils.prune import (
    random_structured,
    random_unstructured,
    l1_unstructured,
    ln_structured,
    remove,
)


def print_namespace(namespace: argparse.Namespace):
    """
    Print the namespace of the arguments

    Args:
        namespace: Namespace of the arguments
    """

    for key, value in vars(namespace).items():
        print(f'{key}: {value}')


def create_hf_dataset_collate_fn(transform: transforms.Compose):
    """
    Wrapper function to create a collate_fn for Hugging Face Datasets that return dictionaries.

    Args:
        transform: Transform to apply to the images

    Returns:
        hf_dataset_collate_fn: Collate_fn for Hugging Face Datasets
    """

    def hf_dataset_collate_fn(batch: Iterable[Mapping[str, Any]]):
        """
        Collate_fn for Hugging Face Datasets that return dictionaries.
        Assumes each item in the batch is a dictionary with "image" and "label" keys.
        """

        images = []
        labels = []

        for item in batch:
            try:
                image = item["image"]
                label = classes_to_idx[IMAGENET2012_CLASSES[os.path.splitext(image.filename)[0].rsplit('_', 1)[1]]]

                if image.mode != 'RGB':
                    continue
                else:
                    image = transform(image, return_tensors="pt")
                    images.append(image["pixel_values"].squeeze(0))
                    labels.append(label)
            except:
                continue

        collated_batch = {
            "pixel_values": torch.stack(images),
            "labels": torch.tensor(labels),
        }

        return collated_batch

    return hf_dataset_collate_fn


def create_model(model_name: str, device: str = 'cuda', half: bool = False):
    """
    Create a model from timm

    Args:
        model_name: Name of the model in timm
        device: Device to run the model on
        half: Put model in bfloat16

    Returns:
        model: Created model
    """
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    if half:
        model = model.half()

    model.eval()
    return model


def get_model_dtype(model):
    """
    Get the dtype of the model

    Args:
        model: Model to get the dtype of

    Returns:
        dtype: dtype of the model
    """

    return next(model.parameters()).dtype.itemsize


def get_torch_autocast_dtype(t: str):
    """
    Get the torch type from the string

    Args:
        t: Type in string format

    Returns:
        torch_type: Torch type
    """
    match t:
        case 'float32':
            return torch.float32
        case 'float16':
            return torch.float16
        case 'bfloat16':
            return torch.bfloat16
        case _:
            raise ValueError(f"Select one of 'float32', 'float16', 'bfloat16'. Got {t}")


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    Calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements

    Args:
        tensor (torch.Tensor): input tensor

    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    Calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements

    model (nn.Module): input model
    """

    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements


def get_num_parameters(model: nn.Module, count_nonzero_only: bool = False) -> int:
    """
    Calculate the total number of parameters of model
    Args:
         model (nn.Module): input model
         count_nonzero_only (bool): only count nonzero weights

    Returns:
        num_counted_elements (int): number of counted elements
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width: int = 32, count_nonzero_only: bool = False) -> int:
    """
    Calculate the model size in bits

    Args:
        model (nn.Module): input model
        data_width (int): #bits per element
        count_nonzero_only (bool): only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width


def collect_stats(model: nn.Module, inputs: torch.Tensor):
    """
    Collect model statistics

    Args:
        model (nn.Module): input model
        inputs (torch.Tensor): input tensor
    """

    model_size = get_model_size(model)
    model_sparsity = get_model_sparsity(model)
    n_params = get_num_parameters(model)
    macs = profile_macs(model, inputs)

    return macs, n_params, model_size, model_sparsity


def evaluate(model: nn.Module, dataloader: DataLoader[Mapping[str, Tensor]], device: str | torch.device) -> float:
    """
    Evaluate the model on the dataset

    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset
        device (str): Device to run the model on

    """

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


def measure_inference_time(model: nn.Module, input_example: Tensor):
    """
    Measure the inference time of the model

    Args:
        model (torch.nn.Module): Model to measure inference time
        input_example (torch.Tensor): Example input tensor

    Returns:
        times (List[float]): List of inference times
    """

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]

    # measure model inference time and size
    for i in range(100):
        start_events[i].record()
        with torch.no_grad():
            model(input_example)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    return times


def report_inference_and_size(
    base_macs: int,
    base_params: int,
    base_size: int,
    model_params_dtype: int,
    model_sparsity: float,
    times: Iterable[float],
    description: str = "",
):
    """
    Report the inference time and size of the model

    Args:
        base_macs (int): MACs of the model
        base_params: Total parameters of the model
        base_size: Total size of the model
        model_params_dtype: dtype of the model
        model_sparsity (float): Sparsity of the model
        times: List of inference times
        description: Description of the stage (e.g. Pruned, Base)
    """

    print(
        f"\nResults {description}\n",
        "-" * 180,
        f"\nBase MACs: {base_macs}",
        " | ",
        f"Total Params: {base_params}",
        " | ",
        f"Total Size: {base_size / 1e6:.4f} MB",
        " | ",
        f"Base dtype: {model_params_dtype} (Bytes)",
        " | ",
        f"Base Sparsity: {model_sparsity:.4f}",
        " | ",
        f"Base Average Inference Time: {sum(times) / 100:.4f} ms +- {np.std(times):.4f} ms\n",
        "-" * 180,
        sep="",
    )


def prune(
    model: nn.Module, pruning_ratio: float, method: str, return_masks: bool = False, dim: int = 0, n: int | float = 1
) -> tuple[nn.Module, dict] | tuple[nn.Module, None]:
    """
    Prune the model with the given pruning ratio and method

    Args:
        model (torch.nn.Module): Model to prune
        pruning_ratio (float): Percentage of Params to keep
        method (str): Pruning method to use
        return_masks (bool): Return the masks
        dim (int): Dimension to prune
        n (int | float): n for ln_structured pruning (L_n norm)

    Returns:
        - tuple (nn.Module, dict): Pruned model and masks
        - nn.Module: Pruned model
    """

    if method == "random_structured":
        pruner = random_structured
    elif method == "random_unstructured":
        pruner = random_unstructured
    elif method == "l1_unstructured":
        pruner = l1_unstructured
    elif method == "ln_structured":
        pruner = ln_structured
    else:
        raise ValueError("Invalid pruning method")

    masks = dict()

    for name, module in model.named_modules():
        if hasattr(module, 'weight') and not isinstance(module, nn.LayerNorm):  # Skip LayerNorm
            if 'dim' in getfullargspec(pruner).args:
                if method == "ln_structured":
                    pruner(module, name="weight", amount=pruning_ratio, dim=dim, n=n)
                else:
                    pruner(module, name="weight", amount=pruning_ratio, dim=dim)
            else:
                pruner(module, name="weight", amount=pruning_ratio)

            if hasattr(module, 'weight_mask'):
                masks[name] = module.weight_mask

            remove(module, name="weight")  # The parameter named name+'_orig' is removed from the parameter list.

    if return_masks:
        return model, masks

    return model, None


def finetune(
    model: nn.Module,
    dataloader: DataLoader[Mapping[str, Tensor]],
    device: str | torch.device,
    epochs: int = 10,
    lr: float = 2e-5,
    weight_decay: float = 1e-4,
    masks: dict[str, Tensor] | None = None,
):
    """
    Finetune the model

    Args:
        model (torch.nn.Module): Model to finetune
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset
        device (str): Device to run the model on
        epochs (int): Number of epochs to finetune
        lr (float): Learning rate
        weight_decay (float): Weight decay
        masks (dict): Masks for the model
    """
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):

        for batch in tqdm(dataloader, total=len(dataloader)):
            images, labels = batch["pixel_values"].to(device), batch["labels"].to(device)

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
