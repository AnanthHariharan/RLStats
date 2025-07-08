#!/usr/bin/env python3
"""
Quickstart example for RL Statistics package
This shows how to use the statistical tools on realistic RL data
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from rl_statistics.core.statistics import compare_algorithms, RLStatistics

def generate_realistic_rl_data():
    """
    Generate realistic RL performance data
    Simulates what you might get from actual RL experiments
    """
    np.random.seed(42)

    # Simulate PPO results - good average performance, moderate variance
    ppo_base = 850
    ppo_scores = np.random.normal(ppo_base, 120, 10)

    # Simulate SAC results - slightly better, lower variance
    sac_base = 920
    sac_scores = np.random.normal(sac_base, 80, 10)

    # Simulate TD3 results - similar to SAC but higher variance
    td3_base = 910
    td3_scores = np.random.normal(td3_base, 150, 10)

    # Simulate A2C results - worse performance, higher variance
    a2c_base = 750
    a2c_scores = np.random.normal(a2c_base, 180, 10)

    return {
        'PPO': ppo_scores,
        'SAC': sac_scores,
        'TD3': td3_scores,
        'A2C': a2c_scores
    }

def demonstrate_statistical_analysis():
    """
    Demonstrate comprehensive statistical analysis
    """
    print("=" * 60)
    print("RL STATISTICAL ANALYSIS DEMONSTRATION")
    print("=" * 60)

    # Generate data
    results = generate_realistic_rl_data()

    print("Analyzing performance of 4 RL algorithms:")
    print("- PPO: Proximal Policy Optimization")
    print("- SAC: Soft Actor-Critic")
    print("- TD3: Twin Delayed Deep Deterministic Policy Gradient")
    print("- A2C: Advantage Actor-Critic")
    print()

    # Run statistical analysis
    analysis = compare_algorithms(results)

    # Display results
    print("SUMMARY STATISTICS:")
    print("-" * 40)
    print(f"{'Algorithm':<10} {'Mean':<8} {'Std':<8} {'Median':<8} {'IQM':<8} {'95% CI':<20}")
    print("-" * 40)

    for alg, stats in analysis['summary'].items():
        ci_str = f"[{stats['ci_lower']:.1f}, {stats['ci_upper']:.1f}]"
        print(f"{alg:<10} {stats['mean']:<8.1f} {stats['std']:<8.1f} {stats['median']:<8.1f} {stats['iqm']:<8.1f} {ci_str:<20}")

    print("\nPAIRWISE COMPARISONS:")
    print("-" * 40)
    print(f"{'Comparison':<15} {'Mean Diff':<12} {'P-value':<10} {'Significant':<12}")
    print("-" * 40)

    for comparison, result in analysis['comparisons'].items():
        alg1, alg2 = comparison.split('_vs_')
        comparison_str = f"{alg1} vs {alg2}"
        significant = "Yes ***" if result['significant'] else "No"

        print(f"{comparison_str:<15} {result['mean_difference']:<12.1f} {result['p_value']:<10.4f} {significant:<12}")

    print("\n*** p < 0.05 indicates statistically significant difference")

    return analysis

def demonstrate_confidence_intervals():
    """
    Show importance of confidence intervals
    """
    print("\n" + "=" * 60)
    print("CONFIDENCE INTERVALS DEMONSTRATION")
    print("=" * 60)

    results = generate_realistic_rl_data()
    stats = RLStatistics()

    print("Why confidence intervals matter:")
    print("Comparing SAC vs TD3 (similar performance):")
    print()

    sac_stats = stats.summary_stats(results['SAC'])
    td3_stats = stats.summary_stats(results['TD3'])

    print(f"SAC: {sac_stats['mean']:.1f} ± {sac_stats['std']:.1f}")
    print(f"     95% CI: [{sac_stats['ci_lower']:.1f}, {sac_stats['ci_upper']:.1f}]")
    print()
    print(f"TD3: {td3_stats['mean']:.1f} ± {td3_stats['std']:.1f}")
    print(f"     95% CI: [{td3_stats['ci_lower']:.1f}, {td3_stats['ci_upper']:.1f}]")
    print()

    # Check if CIs overlap
    sac_ci = (sac_stats['ci_lower'], sac_stats['ci_upper'])
    td3_ci = (td3_stats['ci_lower'], td3_stats['ci_upper'])

    overlap = not (sac_ci[1] < td3_ci[0] or td3_ci[1] < sac_ci[0])

    if overlap:
        print("✓ Confidence intervals overlap -> No clear winner")
    else:
        print("✗ Confidence intervals don't overlap -> Clear difference")

    # Statistical test
    p_val = stats.permutation_test(results['SAC'], results['TD3'])
    print(f"Permutation test p-value: {p_val:.4f}")

    if p_val < 0.05:
        print("→ Statistically significant difference")
    else:
        print("→ No statistically significant difference")

def create_visualization():
    """
    Create a simple visualization
    """
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    results = generate_realistic_rl_data()

    # Create box plot
    fig, ax = plt.subplots(figsize=(10, 6))

    algorithms = list(results.keys())
    data_to_plot = [results[alg] for alg in algorithms]

    box_plot = ax.boxplot(data_to_plot, labels=algorithms, patch_artist=True)

    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Episode Return')
    ax.set_title('RL Algorithm Performance Comparison')
    ax.grid(True, alpha=0.3)

    # Add statistical annotations
    stats = RLStatistics()
    for i, alg in enumerate(algorithms):
        alg_stats = stats.summary_stats(results[alg])
        # Add mean as a point
        ax.plot(i+1, alg_stats['mean'], 'ro', markersize=8)

    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved as 'algorithm_comparison.png'")

    return fig

def main():
    """
    Run the complete demonstration
    """
    print("Starting RL Statistics Quickstart Demo...\n")

    # Run statistical analysis
    analysis = demonstrate_statistical_analysis()

    # Show confidence intervals
    demonstrate_confidence_intervals()

    # Create visualization
    fig = create_visualization()

    print("\n" + "=" * 60)
    print("SUMMARY AND INTERPRETATION")
    print("=" * 60)

    # Find best performing algorithm
    best_alg = max(analysis['summary'].keys(),
                   key=lambda x: analysis['summary'][x]['mean'])

    print(f"Best performing algorithm: {best_alg}")
    print(f"Mean performance: {analysis['summary'][best_alg]['mean']:.1f}")

    # Count significant differences
    significant_comparisons = sum(1 for comp in analysis['comparisons'].values()
                                if comp['significant'])
    total_comparisons = len(analysis['comparisons'])

    print(f"Significant differences: {significant_comparisons}/{total_comparisons}")

    print("\nKey takeaways:")
    print("• Always report confidence intervals, not just mean ± std")
    print("• Use statistical tests for comparisons, don't just eyeball means")
    print("• Consider using robust metrics like IQM for outlier resistance")
    print("• Visualize your results to spot patterns and outliers")

    print(f"\nFiles created:")
    print("• algorithm_comparison.png - Performance comparison plot")

    return analysis

if __name__ == "__main__":
    analysis = main()
