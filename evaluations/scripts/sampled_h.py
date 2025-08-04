import pandas as pd
import matplotlib.pyplot as plt
import re
import json
import seaborn as sns
import numpy as np


def sample_geometric_bounded(p, max_value):
    while True:
        sample = np.random.geometric(p) - 1
        sample = sample + 1
        if sample <= max_value + 1:
            return sample


def plot_h_distribution(p=0.3, max_value=4, n_samples=1000000):
    """
    Plot the distribution of h values sampled from the geometric distribution.
    
    Args:
        p (float): Probability parameter for geometric distribution
        max_value (int): Maximum allowed value
        n_samples (int): Number of samples to generate
    """
    # Generate samples
    samples = [sample_geometric_bounded(p, max_value) for _ in range(n_samples)]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.histplot(samples, discrete=True, stat='probability')
    plt.xlabel('h', fontsize=24)
    plt.ylabel('P(h)', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.savefig('h_distribution.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # Example usage
    plot_h_distribution()

