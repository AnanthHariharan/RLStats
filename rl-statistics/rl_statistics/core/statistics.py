import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional
import warnings

class RLStatistics:
    """
    Core statistical functions for RL evaluation
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize with significance level

        Args:
            alpha: Significance level (0.05 = 95% confidence)
        """
        self.alpha = alpha
        self.confidence_level = 1 - alpha

    def bootstrap_ci(self, data: np.ndarray, n_bootstrap: int = 10000) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for mean

        Args:
            data: 1D array of performance scores
            n_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(data) < 2:
            warnings.warn("Need at least 2 data points for meaningful CI")
            return np.mean(data), np.mean(data)

        bootstrap_means = []
        n = len(data)

        for _ in range(n_bootstrap):
            # Sample with replacement
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        # Calculate percentiles
        lower_percentile = 100 * self.alpha / 2
        upper_percentile = 100 * (1 - self.alpha / 2)

        lower = np.percentile(bootstrap_means, lower_percentile)
        upper = np.percentile(bootstrap_means, upper_percentile)

        return lower, upper

    def permutation_test(self, data_a: np.ndarray, data_b: np.ndarray,
                        n_permutations: int = 10000) -> float:
        """
        Permutation test for comparing two algorithms

        Args:
            data_a: Performance scores for algorithm A
            data_b: Performance scores for algorithm B
            n_permutations: Number of permutations

        Returns:
            p-value
        """
        observed_diff = np.mean(data_a) - np.mean(data_b)
        combined_data = np.concatenate([data_a, data_b])

        count_extreme = 0
        for _ in range(n_permutations):
            # Randomly permute the combined data
            np.random.shuffle(combined_data)

            # Split back into two groups
            perm_a = combined_data[:len(data_a)]
            perm_b = combined_data[len(data_a):]

            # Calculate difference for this permutation
            perm_diff = np.mean(perm_a) - np.mean(perm_b)

            # Count if more extreme than observed
            if abs(perm_diff) >= abs(observed_diff):
                count_extreme += 1

        return count_extreme / n_permutations

    def interquartile_mean(self, data: np.ndarray) -> float:
        """
        Interquartile mean - more robust than regular mean
        Uses middle 50% of data

        Args:
            data: 1D array of scores

        Returns:
            IQM value
        """
        q25, q75 = np.percentile(data, [25, 75])
        filtered_data = data[(data >= q25) & (data <= q75)]

        if len(filtered_data) == 0:
            return np.mean(data)  # Fallback

        return np.mean(filtered_data)

    def summary_stats(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive summary statistics

        Args:
            data: 1D array of performance scores

        Returns:
            Dictionary of statistics
        """
        ci_lower, ci_upper = self.bootstrap_ci(data)

        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'median': np.median(data),
            'iqm': self.interquartile_mean(data),
            'min': np.min(data),
            'max': np.max(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_samples': len(data)
        }

# Convenience functions for quick usage
def bootstrap_ci(data: np.ndarray, alpha: float = 0.05, n_bootstrap: int = 10000) -> Tuple[float, float]:
    """Quick bootstrap CI calculation"""
    stats = RLStatistics(alpha=alpha)
    return stats.bootstrap_ci(data, n_bootstrap)

def permutation_test(data_a: np.ndarray, data_b: np.ndarray, n_permutations: int = 10000) -> float:
    """Quick permutation test"""
    stats = RLStatistics()
    return stats.permutation_test(data_a, data_b, n_permutations)

def compare_algorithms(results_dict: Dict[str, np.ndarray], alpha: float = 0.05) -> Dict:
    """
    Compare multiple algorithms with statistical tests

    Args:
        results_dict: Dictionary mapping algorithm names to performance arrays
        alpha: Significance level

    Returns:
        Dictionary with summary statistics and pairwise comparisons
    """
    stats = RLStatistics(alpha=alpha)

    # Calculate summary stats for each algorithm
    summary = {}
    for alg_name, scores in results_dict.items():
        summary[alg_name] = stats.summary_stats(scores)

    # Pairwise comparisons
    algorithms = list(results_dict.keys())
    comparisons = {}

    for i, alg1 in enumerate(algorithms):
        for j, alg2 in enumerate(algorithms[i+1:], i+1):
            p_value = stats.permutation_test(results_dict[alg1], results_dict[alg2])
            mean_diff = np.mean(results_dict[alg1]) - np.mean(results_dict[alg2])

            comparisons[f"{alg1}_vs_{alg2}"] = {
                'p_value': p_value,
                'mean_difference': mean_diff,
                'significant': p_value < alpha
            }

    return {
        'summary': summary,
        'comparisons': comparisons,
        'alpha': alpha
    }

if __name__ == "__main__":
    # Quick test
    np.random.seed(42)

    # Generate sample data
    alg1_scores = np.random.normal(100, 15, 10)
    alg2_scores = np.random.normal(110, 12, 10)

    # Test bootstrap CI
    ci_lower, ci_upper = bootstrap_ci(alg1_scores)
    print(f"Algorithm 1 CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

    # Test permutation test
    p_val = permutation_test(alg1_scores, alg2_scores)
    print(f"Permutation test p-value: {p_val:.4f}")

    # Test comparison function
    results = compare_algorithms({
        'Algorithm1': alg1_scores,
        'Algorithm2': alg2_scores
    })

    print("\nComparison Results:")
    for alg, stats in results['summary'].items():
        print(f"{alg}: {stats['mean']:.2f} ± {stats['std']:.2f}")

    for comparison, result in results['comparisons'].items():
        print(f"{comparison}: p={result['p_value']:.4f}, significant={result['significant']}")
