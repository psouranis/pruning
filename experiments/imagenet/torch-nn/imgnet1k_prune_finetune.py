import os
import numpy as np
import torch
import argparse
import torch.nn as nn

from classes import create_hf_dataset_collate_fn
from datasets import load_dataset
from transformers import AutoImageProcessor
from finetune_eval import finetune
from imgnet1k_utils import IGNORE, full_report
from torch.utils.data import DataLoader

from pruning.utils import (
    set_seed,
    create_model,
    print_namespace,
)
from pruning.pruners import Pruner


def parse_args() -> argparse.Namespace:
    """Parses the arguments given to the script

    Returns:
        args: parsed arguments

    """
    parser = argparse.ArgumentParser(description="Pruning of Torch Image Models")
    parser.add_argument(
        "--model",
        default="google/vit-base-patch16-224",
        type=str,
        help="Model name in timm",
    )
    parser.add_argument(
        "--dataset", default="../data/ilsvrc", type=str, help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--sample_size",
        default=5000,
        type=int,
        help="Number of samples to use for evaluation",
    )
    parser.add_argument(
        "--prune_method",
        default="random_unstructured",
        type=str,
        help="Pruning method to use",
    )
    parser.add_argument(
        "--pruning_ratio", default=None, type=float, help="Percentage of Params to keep"
    )
    parser.add_argument(
        "--compile", action="store_true", help="Compile model after pruning"
    )
    parser.add_argument(
        "--finetune", action="store_true", help="Finetune model after pruning"
    )
    parser.add_argument(
        "--epochs_finetune", default=1, type=int, help="Number of epochs to finetune"
    )
    parser.add_argument(
        "--finetune_lr", default=2e-5, type=float, help="Learning rate for finetuning"
    )
    parser.add_argument(
        "--compile_mode",
        default="reduce-overhead",
        type=str,
        help="Compile mode for torch",
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Device to run the model on"
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="Batch size for evaluation"
    )
    parser.add_argument("--store_model", action="store_true", help="Store the model")

    args = parser.parse_args()
    return args


def main() -> None:
    set_seed(42)

    args = parse_args()
    print_namespace(args)

    os.makedirs("./output", exist_ok=True)

    assert not args.compile or args.compile_mode, (
        "Compile mode should be provided if compile is True"
    )
    assert not args.finetune or args.epochs_finetune, (
        "Number of epochs to finetune should be provided if finetune is True"
    )

    device = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = create_model(args.model, device)

    transforms = AutoImageProcessor.from_pretrained(args.model, use_fast=True)
    random_indices = np.random.randint(
        0, 50000, args.sample_size
    )  # 50000 is the size of the validation set
    hf_dataset_collate_fn = create_hf_dataset_collate_fn(transform=transforms)

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)

    sample = DataLoader(
        load_dataset(args.dataset, split="validation", num_proc=12).select(
            random_indices
        ),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4,
        collate_fn=hf_dataset_collate_fn,
    )

    input_example = torch.randn(1, 3, 224, 224, device=device)
    full_report(model, input_example, sample, description="base")

    pruner = Pruner(args.prune_method, args.pruning_ratio)
    model, _ = pruner.prune(
        model=model, skip_modules_by_name=IGNORE, skip_modules_by_type=(nn.LayerNorm)
    )

    dataset = load_dataset(
        args.dataset, split="validation", num_proc=12
    ).train_test_split(test_size=args.sample_size)

    finetune_dataloader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4,
        collate_fn=hf_dataset_collate_fn,
    )
    sample = DataLoader(
        dataset["test"],
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4,
        collate_fn=hf_dataset_collate_fn,
    )

    args_finetune = {"lr": args.finetune_lr, "epochs": args.epochs_finetune}
    model = finetune(model, finetune_dataloader, **args_finetune)
    model = pruner.remove(model)  # Remove pruning masks after finetuning

    full_report(
        model,
        input_example,
        sample,
        description=f"Pruned Finetuned {args.pruning_ratio} ratio",
    )

    if args.store_model:
        torch.save(model, f"./output/{args.model.split('/')[1]}.pt")


if __name__ == "__main__":
    main()
