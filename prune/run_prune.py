import argparse
import numpy as np
import os
import torch

from datasets import load_dataset
from per_layer_pruning_ratio import IGNORE, PER_LAYER
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from utils import (
    collect_stats,
    create_hf_dataset_collate_fn,
    create_model,
    evaluate,
    finetune,
    get_model_dtype,
    measure_inference_time,
    print_namespace,
    prune,
    report_inference_and_size,
    sensitivity_scan,
    set_seed,
)
from visualize import plot_pruning_results, plot_weight_histogram


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
        "--dataset", default="./data/ilsvrc", type=str, help="Dataset to evaluate on"
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
        "--search", action="store_true", help="Search for best pruning ratio"
    )
    parser.add_argument(
        "--sensitivity_scan_per_layer",
        action="store_true",
        help="Scan sensitivity for pruning",
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
        "--finetune_lr", default=1e-4, type=float, help="Learning rate for finetuning"
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

    assert args.model or args.list_models, (
        "Model name or list_models should be provided"
    )
    assert (
        sum(
            [
                bool(args.search),
                bool(args.pruning_ratio),
                bool(args.sensitivity_scan_per_layer),
            ]
        )
        == 1
    ), "Choose either search OR pruning_ratio OR sensitivity_scan_per_layer"
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
    model_params_dtype = get_model_dtype(model)

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

    input_size = (3, 224, 224)
    input_example = torch.randn(1, *input_size, device=device)

    # warmup steps
    for _ in range(5):
        model(input_example)

    times = measure_inference_time(model, input_example)
    base_macs, base_params, model_size, model_sparsity = collect_stats(
        model, input_example
    )

    report_inference_and_size(
        base_macs,
        base_params,
        model_size,
        model_params_dtype,
        model_sparsity,
        times,
        description="Base",
    )
    init_acc = evaluate(model, sample, device)

    plot_weight_histogram(
        model,
        layers=["classifier"],
        save=f"./output/cls_histogram_base.png",
        add_to_title="unpruned",
    )

    if args.search:
        model = create_model(args.model, device)
        pruning_ratios = np.arange(0.1, 0.9, 0.1)
        diff_accs = []

        for pr in pruning_ratios:
            model, _ = prune(
                model,
                pr,
                args.prune_method,
                ignore=IGNORE,
                per_layer_pruning_ratio=PER_LAYER,
            )

            times = measure_inference_time(model, input_example)
            base_macs, base_params, model_size, model_sparsity = collect_stats(
                model, input_example
            )

            report_inference_and_size(
                base_macs,
                base_params,
                model_size,
                model_params_dtype,
                model_sparsity,
                times,
                description=f"Pruned {pr} ratio",
            )

            acc = evaluate(model, sample, device)
            diff_accs.append((acc - init_acc) / init_acc)

        plot_pruning_results(
            pruning_ratios,
            np.array(diff_accs),
            init_acc,
            save_as=f"./output/pruning_results_{args.prune_method}.png",
            legend_name=args.prune_method,
        )

    elif args.sensitivity_scan_per_layer:
        pruning_ratios = np.arange(0.1, 0.9, 0.1)
        sensitivity_scan(
            model,
            pruning_ratios,
            args.prune_method,
            sample,
            device=device,
            save="./output/sensitivity_results.json",
        )
    else:
        return_masks = bool(args.finetune)
        model = create_model(args.model, device)
        model, masks = prune(
            model,
            args.pruning_ratio,
            args.prune_method,
            return_masks=return_masks,
            ignore=IGNORE,
            per_layer_pruning_ratio=PER_LAYER,
        )

        if args.compile:
            model = torch.compile(model, mode=args.compile_mode)

        times = measure_inference_time(model, input_example)
        base_macs, base_params, model_size, model_sparsity = collect_stats(
            model, input_example
        )

        report_inference_and_size(
            base_macs,
            base_params,
            model_size,
            model_params_dtype,
            model_sparsity,
            times,
            description=f"Pruned {args.pruning_ratio} ratio",
        )
        evaluate(model, sample, device)

        plot_weight_histogram(
            model,
            layers=["classifier"],
            add_to_title=f"{args.prune_method}",
            save=f"./output/cls_histogram_{args.prune_method}.png",
        )

        if args.finetune:
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

            model = finetune(
                model,
                finetune_dataloader,
                device,
                epochs=args.epochs_finetune,
                lr=args.finetune_lr,
                masks=masks,
            )

            base_macs, base_params, model_size, model_sparsity = collect_stats(
                model, input_example
            )

            report_inference_and_size(
                base_macs,
                base_params,
                model_size,
                model_params_dtype,
                model_sparsity,
                times,
                description=f"Pruned {args.pruning_ratio} ratio",
            )

            evaluate(model, sample, device)

        if args.store_model:
            torch.save(model, f"./output/{args.model.split('/')[1]}.pt")


if __name__ == "__main__":
    main()
