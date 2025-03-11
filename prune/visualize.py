import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def plot_pruning_results(
    ratios: NDArray[float],
    accuracies: NDArray[float],
    title: str = 'Accuracy vs Pruning ratio',
    x_label: str = 'Pruning Ratio (Parameters Pruned Away)',
    y_label: str = 'Accuracy Loss',
    legend_name: str = 'Random Unstructured Pruning',
    save_as: str | None = None,
    show: bool = False,
) -> None:
    """
    Plot the pruning results

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

    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed

    ax.plot(
        ratios * 100,
        accuracies * 100,
        label=legend_name,
        color='#c43c35',  # Red-ish color
        linestyle='-',
        marker='D',
        markersize=6,
        fillstyle='full',
    )  # Filled diamond

    # Horizontal line at zero accuracy loss
    ax.axhline(y=0, color='black', linestyle='--')

    # Axes labels and title
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)

    # Set x and y-axis limits and ticks to match the image
    ax.set_xlim([-1, 101])
    ax.set_xticks(np.arange(0, 100, 10))  # Ticks every 10 from 1 to 100
    ax.set_xticklabels([f'{x}%' for x in np.arange(0, 100, 10)])  # Format x ticks with '%'

    ax.set_ylim([-101, 10])
    ax.set_yticks(np.arange(-100, 5, 10))
    ax.set_yticklabels([f'{y:.1f}%' for y in np.arange(-100, 5, 10)])  # Format y ticks with '%'

    # Grid - horizontal lines only
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
    ax.xaxis.grid(False)  # No vertical grid

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend customization
    legend = ax.legend(
        loc='upper left',
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
        plt.savefig(save_as, dpi=300, bbox_inches='tight')

    # Show plot
    if show:
        plt.show()
