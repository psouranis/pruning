import torch
import torch.nn as nn

from finetune_eval import evaluate
from torch.utils.data import DataLoader

from pruning.utils import (
    collect_stats,
    get_model_dtype,
    measure_inference_time,
    report_inference_and_size,
)

IGNORE = [
    "vit.embeddings.patch_embeddings.projection",
    "classifier",
    "attention",
    "layer.0",
    "layer.11",
]

PER_LAYER = {
    # "vit.encoder.layer.0.attention.attention.key": 0.0,
    # "vit.encoder.layer.0.attention.attention.query": 0.0,
    # "vit.encoder.layer.0.attention.attention.value": 0.0,
    # "vit.encoder.layer.0.attention.output.dense": 0.0,
    # "vit.encoder.layer.0.intermediate.dense": 0.0,
    # "vit.encoder.layer.0.output.dense": 0.0,
    # "vit.encoder.layer.11.attention.attention.key": 0.0,
    # "vit.encoder.layer.11.attention.attention.query": 0.0,
    # "vit.encoder.layer.11.attention.attention.value": 0.0,
    # "vit.encoder.layer.11.attention.output.dense": 0.0,
    # "vit.encoder.layer.11.intermediate.dense": 0.0,
    # "vit.encoder.layer.11.output.dense": 0.0,
}


def full_report(
    model: nn.Module,
    input_example: torch.Tensor,
    sample: DataLoader[torch.Tensor],
    description: str = "base",
) -> None:
    """
    Generates a full report on the given model's performance and characteristics.

    This function measures the inference time, collects various statistics about the model,
    and evaluates the model on a given sample. It then reports the inference time, model size,
    number of parameters, model sparsity, and other relevant metrics.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        input_example (torch.Tensor): An example input tensor to be used for measuring inference time and collecting stats.
        sample (Any): A sample dataset or input to be used for evaluating the model.

    Returns:
        None
    """
    # warmup steps
    model.eval()

    for _ in range(5):
        model(input_example)

    times = measure_inference_time(model, input_example)
    base_macs, base_params, model_size, model_sparsity = collect_stats(
        model, input_example
    )
    model_params_dtype = get_model_dtype(model)

    report_inference_and_size(
        base_macs,
        base_params,
        model_size,
        model_params_dtype,
        model_sparsity,
        times,
        description=description,
    )
    evaluate(model, sample)
