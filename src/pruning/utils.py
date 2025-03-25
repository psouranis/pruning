import numpy as np
import torch
import random
import argparse
import torch.nn as nn

from torch import Tensor
from torchprofile import profile_macs
from transformers import AutoModelForImageClassification
from collections.abc import Iterable


def contains_any_substring_loop(
    main_string: str, list_of_substrings: list[str]
) -> bool:
    """
    Checks if a main string contains any of the substrings in a given list.

    Args:
      main_string: The string to search within.
      list_of_substrings: A list of strings to search for.

    Returns:
      True if any of the substrings are found in the main string, False otherwise.
    """
    return any(main_string.find(substring) != -1 for substring in list_of_substrings)


def set_seed(seed):
    """Sets the random seed for PyTorch, NumPy, and Python's random module.

    Args:
        seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Optional: Ensure deterministic behavior on CUDA (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_namespace(namespace: argparse.Namespace) -> None:
    """Print the namespace of the arguments

    Args:
        namespace: Namespace of the arguments

    """
    for key, value in vars(namespace).items():
        print(f"{key}: {value}")


def create_model(
    model_name: str, device: str = "cuda", half: bool = False
) -> nn.Module:
    """Create a model from timm

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


def get_model_dtype(model: nn.Module) -> int:
    """Get the dtype of the model

    Args:
        model: Model to get the dtype of

    Returns:
        dtype: dtype of the model

    """
    return next(model.parameters()).dtype.itemsize


def get_torch_autocast_dtype(t: str) -> torch.dtype:
    """Get the torch type from the string

    Args:
        t: Type in string format

    Returns:
        torch_type: Torch type

    """
    match t:
        case "float32":
            return torch.float32
        case "float16":
            return torch.float16
        case "bfloat16":
            return torch.bfloat16
        case _:
            raise ValueError(f"Select one of 'float32', 'float16', 'bfloat16'. Got {t}")


def get_sparsity(tensor: torch.Tensor) -> float:
    """Calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements

    Args:
        tensor (torch.Tensor): input tensor

    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """Calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements

        Accounts for the case that the model contains pruned weights (stored in weight_orig)

    Args:
        model (nn.Module): input model
    """
    num_nonzeros, num_elements = 0, 0

    for name, param in model.named_parameters():
        if name.endswith("weight_orig"):
            submodule = name.split(".weight_orig")[0]
            num_nonzeros += model.get_submodule(submodule).weight.count_nonzero()
            num_elements += model.get_submodule(submodule).weight.numel()
        else:
            num_nonzeros += param.count_nonzero()
            num_elements += param.numel()

    return 1 - float(num_nonzeros) / num_elements


def get_num_parameters(model: nn.Module, count_nonzero_only: bool = False) -> int:
    """Calculate the total number of parameters of model
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


def get_model_size(
    model: nn.Module, data_width: int = 32, count_nonzero_only: bool = False
) -> int:
    """Calculate the model size in bits

    Args:
        model (nn.Module): input model
        data_width (int): #bits per element
        count_nonzero_only (bool): only count nonzero weights

    """
    return get_num_parameters(model, count_nonzero_only) * data_width


def collect_stats(
    model: nn.Module, inputs: torch.Tensor
) -> tuple[int, int, int, float]:
    """Collect model statistics

    Args:
        model (nn.Module): input model
        inputs (torch.Tensor): input tensor

    """
    model_size = get_model_size(model)
    model_sparsity = get_model_sparsity(model)
    n_params = get_num_parameters(model)
    macs = profile_macs(model, inputs)

    return macs, n_params, model_size, model_sparsity


def measure_inference_time(model: nn.Module, input_example: Tensor) -> list[float]:
    """Measure the inference time of the model

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
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events, strict=True)]

    return times


def report_inference_and_size(
    base_macs: int,
    base_params: int,
    base_size: int,
    model_params_dtype: int,
    model_sparsity: float,
    times: Iterable[float],
    description: str = "",
) -> None:
    """Report the inference time and size of the model

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
        f"\nMACs: {base_macs}",
        " | ",
        f"Total Params: {base_params}",
        " | ",
        f"Total Size: {base_size / 1e6:.4f} MB",
        " | ",
        f"dtype: {model_params_dtype} (Bytes)",
        " | ",
        f"Sparsity: {model_sparsity:.4f}",
        " | ",
        f"Average Inference Time: {sum(times) / 100:.4f} ms +- {np.std(times):.4f} ms\n",
        "-" * 180,
        sep="",
    )
