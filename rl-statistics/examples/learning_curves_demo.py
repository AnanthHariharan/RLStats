#!/usr/bin/env python3
"""
Advanced learning curve analysis demonstration
Shows how to analyze learning curves with proper statistical treatment
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from rl_statistics.core.curves import LearningCurveAnalyzer, analyze_curves
from rl_statistics.core.statistics import compare_algorithms

def generate_realistic_learning_curves():
    """
    Generate realistic learning curves for different RL algorithms
    Each algorithm has different characteristics:
    - PPO: Steady improvement, moderate variance
    - SAC: Fast early learning, then plateau
    - TD3: Slow start, then rapid improvement
    - A2C: High variance, unstable learning
    """
    np.random.seed(42)

    n_runs = 8
    n_timesteps = 2000
    timesteps = np.arange(n_timesteps)

    curves_dict = {}

    # PPO: Steady learner
    ppo_curves = []
    for run in range(n_runs):
        # Logistic growth curve
        base_curve = 900 / (1 + np.exp(-(timesteps - 800) / 300))
        noise = np.random.normal(0, 20, n_timesteps)
        # Add some correlation in noise (realistic training noise)
        for i in range(1, n_timesteps):
            noise[i] = 0.8 * noise[i-1] + 0.2 * noise[i]

        curve = base_curve + noise
        ppo_curves.append(curve)

    curves_dict['PPO'] = np.array(ppo_curves)

    # SAC: Fast early learner, then plateaus
    sac_curves = []
    for run in range(n_runs):
        # Exponential approach to asymptote
        base_curve = 850 * (1 - np.exp(-timesteps / 400))
        noise = np.random.normal(0, 15, n_timesteps)
        # Correlated noise
        for i in range(1, n_timesteps):
            noise[i] = 0.9 * noise[i-1] + 0.1 * noise[i]

        curve = base_curve + noise
        sac_curves.append(curve)

    curves_dict['SAC'] = np.array(sac_curves)

    # TD3: Slow start, then rapid improvement
    td3_curves = []
    for run in range(n_runs):
        # Delayed sigmoid
        base_curve = 880 / (1 + np.exp(-(timesteps - 1200) / 200))
        noise = np.random.normal(0, 25, n_timesteps)
        # Correlated noise
        for i in range(1, n_timesteps):
            noise[i] = 0.7 * noise[i-1] + 0.3 * noise[i]

        curve = base_curve + noise
        td3_curves.append(curve)

    curves_dict['TD3'] = np.array(td3_curves)

    # A2C: High variance, unstable
    a2c_curves = []
    for run in range(n_runs):
        # Lower performance with high variance
        base_curve = 650 / (1 + np.exp(-(timesteps - 1000) / 400))
        noise = np.random.normal(0, 50, n_timesteps)
        # Less correlated noise (more chaotic)
        for i in range(1, n_timesteps):
            noise[i] = 0.5 * noise[i-1] + 0.5 * noise[i]

        curve = base_curve + noise
        a2c_curves.append(curve)

    curves_dict['A2C'] = np.array(a2c_curves)

    return curves_dict, timesteps

def plot_learning_curves_with_bands(curves_dict, timesteps, save_path='learning_curves_comparison.png'):
    """
    Create publication-quality learning curve plots with confidence bands
    """
    analyzer = LearningCurveAnalyzer()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Professional colors

    # Plot 1: Raw curves with confidence bands
    for i, (alg_name, curves) in enumerate(curves_dict.items()):
        color = colors[i % len(colors)]

        # Calculate confidence bands
        mean_curve, lower_band, upper_band = analyzer.bootstrap_confidence_bands(curves)

        # Plot mean line
        ax1.plot(timesteps, mean_curve, label=alg_name, color=color, linewidth=2)

        # Plot confidence band
        ax1.fill_between(timesteps, lower_band, upper_band, alpha=0.2, color=color)

        # Plot individual runs (light)
        for curve in curves:
            ax1.plot(timesteps, curve, color=color, alpha=0.1, linewidth=0.5)

    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Episode Return')
    ax1.set_title('Learning Curves with 95% Confidence Bands')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Smoothed curves only
    for i, (alg_name, curves) in enumerate(curves_dict.items()):
        color = colors[i % len(colors)]

        # Smooth curves
        smoothed_curves = analyzer.smooth_curves(curves, method='gaussian', window_size=50)
        mean_curve, lower_band, upper_band = analyzer.bootstrap_confidence_bands(smoothed_curves)

        ax2.plot(timesteps, mean_curve, label=alg_name, color=color, linewidth=2)
        ax2.fill_between(timesteps, lower_band, upper_band, alpha=0.2, color=color)

    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Episode Return')
    ax2.set_title('Smoothed Learning Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Learning curves plot saved as '{save_path}'")

    return fig

def analyze_sample_efficiency(curves_dict, timesteps):
    """
    Analyze how quickly each algorithm reaches different performance thresholds
    """
    print("\n" + "=" * 60)
    print("SAMPLE EFFICIENCY ANALYSIS")
    print("=" * 60)

    analyzer = LearningCurveAnalyzer()

    # Define performance thresholds to analyze
    thresholds = [500, 600, 700, 800]

    print("Performance Thresholds Analysis:")
    print("-" * 40)
    print(f"{'Algorithm':<10} {'Threshold':<10} {'Success Rate':<12} {'Mean Steps':<12} {'Median Steps':<12}")
    print("-" * 40)

    efficiency_results = {}

    for threshold in thresholds:
        print(f"\nTarget: {threshold}")
        efficiency = analyzer.sample_efficiency_analysis(curves_dict, threshold)
        efficiency_results[threshold] = efficiency

        for alg_name, stats in efficiency.items():
            success_rate = stats['success_rate'] * 100
            mean_steps = stats['mean_samples_to_target']
            median_steps = stats['median_samples_to_target']

            if mean_steps == np.inf:
                mean_str = "Never"
                median_str = "Never"
            else:
                mean_str = f"{mean_steps:.0f}"
                median_str = f"{median_steps:.0f}"

            print(f"{alg_name:<10} {threshold:<10} {success_rate:<12.1f}% {mean_str:<12} {median_str:<12}")

    return efficiency_results

def compare_learning_characteristics(curves_dict):
    """
    Compare different aspects of learning: speed, stability, final performance
    """
    print("\n" + "=" * 60)
    print("LEARNING CHARACTERISTICS COMPARISON")
    print("=" * 60)

    analyzer = LearningCurveAnalyzer()
    comparison = analyzer.compare_learning_efficiency(curves_dict)

    print("Final Performance:")
    print("-" * 30)
    print(f"{'Algorithm':<10} {'Mean':<10} {'Std':<10} {'Median':<10}")
    print("-" * 30)

    for alg_name in comparison['algorithms']:
        final_perf = comparison['final_performance'][alg_name]
        print(f"{alg_name:<10} {final_perf['mean']:<10.1f} {final_perf['std']:<10.1f} {final_perf['median']:<10.1f}")

    print("\nLearning Speed (slope of first half):")
    print("-" * 30)
    print(f"{'Algorithm':<10} {'Mean':<10} {'Std':<10}")
    print("-" * 30)

    for alg_name in comparison['algorithms']:
        learning_speed = comparison['learning_speed'][alg_name]
        print(f"{alg_name:<10} {learning_speed['mean']:<10.3f} {learning_speed['std']:<10.3f}")

    print("\nStability (coefficient of variation in second half):")
    print("-" * 30)
    print(f"{'Algorithm':<10} {'Mean CV':<10} {'Std CV':<10}")
    print("-" * 30)

    for alg_name in comparison['algorithms']:
        stability = comparison['stability'][alg_name]
        print(f"{alg_name:<10} {stability['mean_cv']:<10.3f} {stability['std_cv']:<10.3f}")

    return comparison

def create_sample_efficiency_plot(efficiency_results, save_path='sample_efficiency.png'):
    """
    Create visualization of sample efficiency across thresholds
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    thresholds = sorted(efficiency_results.keys())
    algorithms = list(efficiency_results[thresholds[0]].keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Plot 1: Success rates
    for i, alg in enumerate(algorithms):
        success_rates = [efficiency_results[t][alg]['success_rate'] * 100 for t in thresholds]
        ax1.plot(thresholds, success_rates, 'o-', label=alg, color=colors[i], linewidth=2, markersize=8)

    ax1.set_xlabel('Performance Threshold')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rate vs Performance Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    # Plot 2: Mean steps to threshold (for successful runs only)
    width = 15
    x_positions = np.arange(len(thresholds))

    for i, alg in enumerate(algorithms):
        mean_steps = []
        for t in thresholds:
            steps = efficiency_results[t][alg]['mean_samples_to_target']
            if steps == np.inf:
                mean_steps.append(0)  # Never achieved
            else:
                mean_steps.append(steps)

        x_pos = x_positions + i * width
        bars = ax2.bar(x_pos, mean_steps, width, label=alg, color=colors[i], alpha=0.7)

    ax2.set_xlabel('Performance Threshold')
    ax2.set_ylabel('Mean Steps to Threshold')
    ax2.set_title('Sample Efficiency (Successful Runs Only)')
    ax2.set_xticks(x_positions + width * 1.5)
    ax2.set_xticklabels(thresholds)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sample efficiency plot saved as '{save_path}'")

    return fig

def statistical_significance_test(curves_dict):
    """
    Test statistical significance of differences in final performance
    """
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 60)

    # Extract final performance
    final_performance = {}
    for alg_name, curves in curves_dict.items():
        final_performance[alg_name] = curves[:, -1]

    # Use our existing comparison function
    comparison = compare_algorithms(final_performance)

    print("Final Performance Statistical Comparison:")
    print("-" * 50)
    print(f"{'Algorithm':<10} {'Mean':<10} {'95% CI':<20}")
    print("-" * 50)

    for alg, stats in comparison['summary'].items():
        ci_str = f"[{stats['ci_lower']:.1f}, {stats['ci_upper']:.1f}]"
        print(f"{alg:<10} {stats['mean']:<10.1f} {ci_str:<20}")

    print("\nPairwise Comparisons:")
    print("-" * 50)
    print(f"{'Comparison':<20} {'Mean Diff':<12} {'P-value':<10} {'Significant':<12}")
    print("-" * 50)

    for comp_name, result in comparison['comparisons'].items():
        alg1, alg2 = comp_name.split('_vs_')
        comp_str = f"{alg1} vs {alg2}"
        significant = "Yes ***" if result['significant'] else "No"

        print(f"{comp_str:<20} {result['mean_difference']:<12.1f} {result['p_value']:<10.4f} {significant:<12}")

    return comparison

def main():
    """
    Run the complete learning curve analysis demonstration
    """
    print("Starting Advanced Learning Curve Analysis...")
    print("=" * 60)

    # Generate realistic learning curves
    curves_dict, timesteps = generate_realistic_learning_curves()

    print(f"Generated learning curves for {len(curves_dict)} algorithms")
    print(f"Each algorithm: {curves_dict['PPO'].shape[0]} runs × {curves_dict['PPO'].shape[1]} timesteps")

    # Create learning curve plots
    plot_learning_curves_with_bands(curves_dict, timesteps)

    # Analyze sample efficiency
    efficiency_results = analyze_sample_efficiency(curves_dict, timesteps)

    # Compare learning characteristics
    learning_comparison = compare_learning_characteristics(curves_dict)

    # Create sample efficiency plot
    create_sample_efficiency_plot(efficiency_results)

    # Statistical significance testing
    statistical_comparison = statistical_significance_test(curves_dict)

    # Summary and insights
    print("\n" + "=" * 60)
    print("SUMMARY AND INSIGHTS")
    print("=" * 60)

    # Find best performing algorithm
    best_alg = max(statistical_comparison['summary'].keys(),
                   key=lambda x: statistical_comparison['summary'][x]['mean'])

    print(f"Best final performance: {best_alg}")
    print(f"Mean final performance: {statistical_comparison['summary'][best_alg]['mean']:.1f}")

    # Find fastest learner (highest learning speed)
    fastest_alg = max(learning_comparison['algorithms'],
                     key=lambda x: learning_comparison['learning_speed'][x]['mean'])

    print(f"Fastest learner: {fastest_alg}")
    print(f"Learning speed: {learning_comparison['learning_speed'][fastest_alg]['mean']:.3f}")

    # Find most stable algorithm (lowest CV)
    most_stable = min(learning_comparison['algorithms'],
                     key=lambda x: learning_comparison['stability'][x]['mean_cv'])

    print(f"Most stable: {most_stable}")
    print(f"Stability (CV): {learning_comparison['stability'][most_stable]['mean_cv']:.3f}")

    # Count significant differences
    significant_count = sum(1 for comp in statistical_comparison['comparisons'].values()
                          if comp['significant'])
    total_comparisons = len(statistical_comparison['comparisons'])

    print(f"Significant differences: {significant_count}/{total_comparisons}")

    print("\nKey Insights:")
    print("• Learning curves reveal different algorithm characteristics")
    print("• Sample efficiency analysis shows when algorithms reach targets")
    print("• Confidence bands quantify uncertainty in learning progress")
    print("• Statistical tests ensure robust conclusions about differences")

    print(f"\nGenerated Files:")
    print("• learning_curves_comparison.png - Learning curves with confidence bands")
    print("• sample_efficiency.png - Sample efficiency analysis")

    # Return results for further analysis
    return {
        'curves': curves_dict,
        'timesteps': timesteps,
        'efficiency': efficiency_results,
        'learning_comparison': learning_comparison,
        'statistical_comparison': statistical_comparison
    }

if __name__ == "__main__":
    results = main()
