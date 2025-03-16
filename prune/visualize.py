import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray


def plot_pruning_results(
    ratios: NDArray[float],
    accuracies: NDArray[float],
    base_accuracy: float,
    title: str = "Accuracy vs Pruning ratio",
    x_label: str = "Pruning Ratio (Parameters Pruned Away)",
    y_label: str = "Accuracy Loss",
    legend_name: str = "Random Unstructured Pruning",
    save_as: str | None = None,
    show: bool = False,
) -> None:
    """Plot the pruning results

    Args:
        ratios (np.array): array of pruning ratios
        accuracies (np.array): array of accuracies
        title (str): Title of the plot
        x_label (str): Label for the x-axis
        y_label (str): Label for the y-axis
        legend_name (str): Name of
        save_as (str): Path to save the plot
        show (bool): Show the plot

    """
    plt.style.use("default")

    _, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed

    ax.plot(
        ratios * 100,
        accuracies * 100,
        label=legend_name,
        color="#c43c35",  # Red-ish color
        linestyle="-",
        marker="D",
        markersize=6,
        fillstyle="full",
    )  # Filled diamond

    # Horizontal line at zero accuracy loss
    ax.axhline(y=0, color="black", linestyle="--")

    # Axes labels and title
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(
        ylabel=f"{y_label} (Base Accuracy : {base_accuracy:.4f})", fontsize=12
    )
    ax.set_title(title, fontsize=14)

    # Set x and y-axis limits and ticks to match the image
    ax.set_xlim([-1, 101])
    ax.set_xticks(np.arange(0, 100, 10))  # Ticks every 10 from 1 to 100
    ax.set_xticklabels(
        [f"{x}%" for x in np.arange(0, 100, 10)]
    )  # Format x ticks with '%'

    ax.set_ylim([-101, 10])
    ax.set_yticks(np.arange(-100, 5, 10))
    ax.set_yticklabels(
        [f"{y:.1f}%" for y in np.arange(-100, 5, 10)]
    )  # Format y ticks with '%'

    # Grid - horizontal lines only
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="lightgray")
    ax.xaxis.grid(False)  # No vertical grid

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend customization
    _ = ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=10,
        handlelength=1.5,
        handletextpad=0.5,
        markerscale=0.8,
        columnspacing=1,
    )

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches="tight")

    # Show plot
    if show:
        plt.show()


def plot_weight_histogram(
    model,
    layers=None,
    bins=50,
    figsize=(10, 6),
    title="Neural Network Weights Histogram",
    alpha=0.7,
    color=None,
    edgecolor: str = "black",
    save: str | None = None,
    show: bool = False,
    add_to_title: str | None = None,
):
    """
    Generates a beautiful histogram of the weights of a PyTorch neural network.

    Args:
        model (torch.nn.Module): The PyTorch neural network model.
        layers (list, optional): A list of layer names (strings) to include in the histogram.
                                 If None, all weight parameters will be included. Defaults to None.
                                 You can specify full parameter names (e.g., 'conv1.weight') or
                                 just the base layer name (e.g., 'conv1' to include all weights
                                 associated with that layer).
        bins (int, optional): The number of bins in the histogram. Defaults to 50.
        figsize (tuple, optional): The size of the figure (width, height). Defaults to (10, 6).
        title (str, optional): The title of the plot. Defaults to "Neural Network Weights Histogram".
        alpha (float, optional): The transparency of the histogram bars. Defaults to 0.7.
        color (str or list, optional): The color(s) of the histogram bars. If a list is provided,
                                       it should have the same length as the number of layers being plotted.
                                       Defaults to None (uses Matplotlib's default).
        edgecolor (str, optional): The color of the edges of the histogram bars. Defaults to 'black'.
    """
    weight_data = []
    layer_names = []

    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name:
            if layers is None:
                weight_data.append(param.detach().cpu().numpy().flatten())
                layer_names.append(name)
            elif isinstance(layers, list):
                include_layer = False
                for layer_name in layers:
                    if name.startswith(layer_name + ".") and "weight" in name:
                        include_layer = True
                        break
                    elif name == layer_name + ".weight":
                        include_layer = True
                        break
                    elif (
                        name.split(".")[0] == layer_name and "weight" in name
                    ):  # Handle base layer name
                        include_layer = True
                        break
                if include_layer:
                    weight_data.append(param.detach().cpu().numpy().flatten())
                    layer_names.append(name)

    if not weight_data:
        print("No weight parameters found for the specified layers.")
        return

    num_layers = len(weight_data)
    if num_layers == 1:
        plt.figure(figsize=figsize)
        if color:
            plt.hist(
                weight_data[0],
                bins=bins,
                alpha=alpha,
                color=color
                if isinstance(color, str)
                else color[0]
                if isinstance(color, list) and len(color) > 0
                else None,
                edgecolor=edgecolor,
            )
        else:
            plt.hist(weight_data[0], bins=bins, alpha=alpha, edgecolor=edgecolor)
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        if bool(add_to_title):
            plt.title(f"Histogram of {layer_names[0]} : {add_to_title}")
        else:
            plt.title(f"Histogram of {layer_names[0]}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        if show:
            plt.show()
    else:
        rows = int(np.ceil(np.sqrt(num_layers)))
        cols = int(np.ceil(num_layers / rows))
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(title, fontsize=16)

        for i, (data, name) in enumerate(zip(weight_data, layer_names)):
            ax = axes.flatten()[i]
            if color:
                ax.hist(
                    data,
                    bins=bins,
                    alpha=alpha,
                    color=color[i]
                    if isinstance(color, list) and len(color) == num_layers
                    else None,
                    edgecolor=edgecolor,
                )
            else:
                ax.hist(data, bins=bins, alpha=alpha, edgecolor=edgecolor)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("Weight Value", fontsize=8)
            ax.set_ylabel("Frequency", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(True, linestyle="--", alpha=0.6)

        # Remove any unused subplots
        if num_layers < rows * cols:
            for i in range(num_layers, rows * cols):
                fig.delaxes(axes.flatten()[i])

        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Adjust layout to prevent title overlap

    if bool(save):
        plt.savefig(save)

    if show:
        plt.show()
