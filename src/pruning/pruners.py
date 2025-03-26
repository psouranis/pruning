import json
import torch
import torch.nn as nn

from torch import Tensor
from typing import Any, TypeVar, ParamSpec
from collections.abc import Mapping, Callable, Iterable
from torch.utils.data import DataLoader
from torch.nn.utils.prune import (
    remove,
    ln_structured,
    l1_unstructured,
    random_structured,
    random_unstructured,
)

from .utils import get_model_sparsity, contains_any_substring_loop

P = ParamSpec("P")
T = TypeVar("T")

PRUNE_METHODS = {
    "random_unstructured": random_unstructured,
    "random_structured": random_structured,
    "ln_structured": ln_structured,
    "l1_unstructured": l1_unstructured,
}


def _select_pruning_method(prune_method: str) -> Callable[[str], Any]:
    """Get the pruning method"""

    if prune_method not in PRUNE_METHODS:
        raise KeyError("Not available pruning method.")

    return PRUNE_METHODS[prune_method]


class Pruner:
    def __init__(self, prune_method: str, ratio: float = 0.0):
        """Initializes the pruner with the specified pruning method and ratio.

        Args:
            prune_method (str): The method to use for pruning.
            ratio (float): The ratio of pruning to apply.
        """

        self.ratio = ratio
        self.pruner = _select_pruning_method(prune_method)

    def prune(
        self,
        model: nn.Module,
        skip_modules_by_type: tuple[nn.Module, ...] | None = None,
        skip_modules_by_name: list[str] | None = None,
        per_layer_ratio: dict[str, float] | None = None,
        return_masks: bool = False,
        remove_original_weights: bool = False,
        **kwargs: dict[str, Any],
    ) -> tuple[nn.Module, dict] | tuple[nn.Module, None]:
        """Prune the model with the given pruning ratio and method

        Args:
            model (torch.nn.Module): Model to prune
            skip_modules_by_type (tuple[nn.Module], optional): Skip modules by type (e.g. nn.LayerNorm)
            skip_modules_by_name (list[str], optional): List of layers by names or (substrings) to ignore
            per_layer_ratio (dict[str, float], optional): Define different pruning ratios for each layer.
            return_masks (bool): Return the masks

        Returns:
            - tuple (nn.Module, dict): Pruned model and masks
            - nn.Module: Pruned model

        """

        per_layer_ratio = per_layer_ratio or {}
        skip_modules_by_name = skip_modules_by_name or []
        skip_modules_by_type = skip_modules_by_type or ()
        masks = {}

        for name, module in model.named_modules():
            if (
                hasattr(module, "weight")
                and not isinstance(module, skip_modules_by_type)
                and not contains_any_substring_loop(name, skip_modules_by_name)
            ):
                amount = per_layer_ratio.get(name, self.ratio)
                print(f"Pruning layer {name} with amount {amount}")
                self.pruner(module, name="weight", amount=amount, **kwargs)

            if hasattr(module, "weight_mask"):
                masks[name] = module.weight_mask

        if remove_original_weights:
            self.remove(model)

        if return_masks:
            return model, masks

        return model, None

    def iterative_prune(
        self,
        model: nn.Module,
        iterations: int = 1,
        *,
        dataloader: DataLoader[torch.Tensor],
        eval_dataloader: DataLoader[torch.Tensor],
        finetune: Callable[P, T],
        evaluate: Callable[P, T],
        stop_on_target_sparsity: bool = True,
        skip_modules_by_name: list[str] | None = None,
        skip_modules_by_type: tuple[nn.Module, ...] | None = None,
        per_layer_ratio: dict[str, float] | None = None,
        return_masks: bool = False,
        args_evaluate: dict[str, Any] | None = None,
        args_finetune: dict[str, Any] | None = None,
    ) -> nn.Module:
        """Iterate through the pruning process for a given number of iterations, finetune the model, and evaluate it.

        Iterate with pruning starts with a small pruning ratio - prunes and evaluates the network and graudally
        increases it to reach the desired ratio ~self.ratio.

        Args:
            model (nn.Module): The neural network model to be pruned.
            dataloader (DataLoader[torch.Tensor]): The dataloader providing the data for finetuning and evaluation.
            iterations (int): The number of pruning iterations to perform.
            evaluate (Callable[P, T]): A callable function to evaluate the model after each pruning iteration.
            skip_modules_by_name (list[str]): List of module names to skip during pruning.
            skip_modules_by_type (tuple[nn.Module, ...]): Tuple of module types to skip during pruning.
            per_layer_ratio (dict[str, float]): Dictionary specifying the pruning ratio for each layer by name.
            return_masks (bool, optional): Whether to return the pruning masks. Defaults to False.

        Returns:
            nn.Module: The pruned and finetuned model.
        """
        args_evaluate = args_evaluate or {}
        args_finetune = args_finetune or {}

        step_ratio = self.ratio / iterations
        self.ratio = step_ratio

        for _ in range(iterations):
            model, _ = self.prune(
                model=model,
                skip_modules_by_name=skip_modules_by_name,
                skip_modules_by_type=skip_modules_by_type,
                return_masks=return_masks,
                per_layer_ratio=per_layer_ratio,
            )

            finetune(model, dataloader, **args_finetune)
            evaluate(model, eval_dataloader, **args_evaluate)
            sparsity = get_model_sparsity(model)
            print(f"Sparsity: {sparsity}")

            self.remove(model)  # remove the masks only after finetuning

            if stop_on_target_sparsity and get_model_sparsity(model) >= self.ratio:
                print(f"Target sparsity {self.ratio} reached. Stopping the iterations.")
                break

            self.ratio += step_ratio

        return model

    def scan(
        self,
        model: nn.Module,
        ratio_range: Iterable[float],
        dataloader: DataLoader[Mapping[str, Tensor]],
        evaluate: Callable[[nn.Module, DataLoader[Mapping[str, Tensor]]], float],
        *,
        skip_modules_by_type: tuple[nn.Module] | None = None,
        skip_modules_by_name: list[str] | None = None,
        modules: list[str] | None = None,
        save_to: str | None = None,
        **kwargs: dict[str, Any],
    ) -> dict[str, dict[str, float]]:
        """
        Scans the sensitivity of a model's modules to pruning by evaluating the model's performance
        over a range of pruning ratios.

        Args:
            model (nn.Module): The neural network model to be pruned.
            range (Iterable[float]): An iterable of pruning ratios to evaluate.
            dataloader (DataLoader[Mapping[str, Tensor]]): DataLoader for the evaluation dataset.
            evaluate (Callable[[nn.Module, DataLoader[Mapping[str, Tensor]]], float]):
                A function that evaluates the model and returns a performance metric.
            skip_modules_by_type (tuple[nn.Module] | None, optional): Tuple of module types to skip during pruning. Defaults to None.
            skip_modules_by_name (list[str] | None, optional): List of module names to skip during pruning. Defaults to None.
            modules (list[str] | None, optional): List of specific module names to prune. If None, all modules with weights are considered. Defaults to None.
            save_to (str | None, optional): File path to save the sensitivities dictionary as a JSON file. Defaults to None.
            **kwargs (dict[str, Any]): Additional keyword arguments for the pruner.

        Returns:
            dict[str, dict[str, float]]: A dictionary where keys are module names and values are lists of dictionaries
            containing 'ratio', 'metric', and 'sparsity' for each pruning ratio.
        """

        skip_modules_by_type = skip_modules_by_type or ()
        skip_modules_by_name = skip_modules_by_name or []

        modules = modules or [
            name
            for name, module in model.named_modules()
            if hasattr(module, "weight")
            and not isinstance(module, skip_modules_by_type)
            and not contains_any_substring_loop(name, skip_modules_by_name)
        ]

        sensitivities = {}
        for name, module in model.named_modules():
            if name not in modules:
                continue
            original_state = module.weight.clone()
            print(f"Running sensitivty scan for module: {name}")

            for ratio in ratio_range:
                print(f"Pruning ratio: {ratio}")
                self.pruner(module, name="weight", amount=ratio, **kwargs)

                remove(module, name="weight")

                metric = evaluate(model, dataloader)
                sparsity = get_model_sparsity(model)
                sensitivities.setdefault(name, []).append(
                    {"ratio": ratio, "metric": metric, "sparsity": sparsity}
                )

            module.weight.data = original_state

        if save_to:
            with open(save_to, "w") as f:
                json.dump(sensitivities, f)

    def remove(self, model: nn.Module) -> nn.Module:
        """Remove pruning masks from the model.

        Args:
            model (nn.Module): The model to remove the masks from.
            skip_modules_by_name (list[str]): List of module names to skip during mask removal.
        """

        for name, module in model.named_modules():
            if hasattr(module, "weight_mask"):
                print(f"Removing weights from {name}\n")
                remove(module, name="weight")

        return model
