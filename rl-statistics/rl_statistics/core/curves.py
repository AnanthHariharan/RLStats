import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Tuple, Optional, List
import warnings

class LearningCurveAnalyzer:
    """
    Analyze and compare learning curves with proper statistical treatment
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize analyzer

        Args:
            confidence_level: Confidence level for bands (0.95 = 95%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def smooth_curves(self, curves: np.ndarray, method: str = 'gaussian',
                     window_size: int = 100) -> np.ndarray:
        """
        Smooth learning curves for better visualization

        Args:
            curves: Shape (n_runs, n_timesteps)
            method: 'gaussian', 'rolling', or 'none'
            window_size: Size of smoothing window

        Returns:
            Smoothed curves with same shape
        """
        if method == 'none':
            return curves

        if method == 'gaussian':
            # Use 1/6 of window size as sigma (99.7% weight within window)
            sigma = window_size / 6
            return np.array([gaussian_filter1d(curve, sigma=sigma) for curve in curves])

        elif method == 'rolling':
            smoothed = []
            for curve in curves:
                # Use pandas-style rolling mean with proper edge handling
                smoothed_curve = np.convolve(curve, np.ones(window_size)/window_size, mode='same')
                smoothed.append(smoothed_curve)
            return np.array(smoothed)

        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    def bootstrap_confidence_bands(self, curves: np.ndarray,
                                  n_bootstrap: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate bootstrap confidence bands for learning curves

        Args:
            curves: Shape (n_runs, n_timesteps)
            n_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (mean_curve, lower_band, upper_band)
        """
        n_runs, n_timesteps = curves.shape

        if n_runs < 2:
            warnings.warn("Need at least 2 runs for meaningful confidence bands")
            mean_curve = np.mean(curves, axis=0)
            return mean_curve, mean_curve, mean_curve

        bootstrap_means = []

        for _ in range(n_bootstrap):
            # Sample runs with replacement
            sampled_indices = np.random.choice(n_runs, size=n_runs, replace=True)
            bootstrap_sample = curves[sampled_indices]
            bootstrap_means.append(np.mean(bootstrap_sample, axis=0))

        bootstrap_means = np.array(bootstrap_means)

        # Calculate percentiles
        lower_percentile = 100 * self.alpha / 2
        upper_percentile = 100 * (1 - self.alpha / 2)

        mean_curve = np.mean(curves, axis=0)
        lower_band = np.percentile(bootstrap_means, lower_percentile, axis=0)
        upper_band = np.percentile(bootstrap_means, upper_percentile, axis=0)

        return mean_curve, lower_band, upper_band

    def t_test_confidence_bands(self, curves: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate confidence bands using t-distribution (faster but assumes normality)

        Args:
            curves: Shape (n_runs, n_timesteps)

        Returns:
            Tuple of (mean_curve, lower_band, upper_band)
        """
        n_runs = curves.shape[0]

        if n_runs < 2:
            warnings.warn("Need at least 2 runs for meaningful confidence bands")
            mean_curve = np.mean(curves, axis=0)
            return mean_curve, mean_curve, mean_curve

        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0, ddof=1)  # Sample standard deviation

        # t-distribution critical value
        t_critical = stats.t.ppf(1 - self.alpha/2, n_runs - 1)

        # Standard error of the mean
        standard_error = std_curve / np.sqrt(n_runs)
        margin_of_error = t_critical * standard_error

        lower_band = mean_curve - margin_of_error
        upper_band = mean_curve + margin_of_error

        return mean_curve, lower_band, upper_band

    def sample_efficiency_analysis(self, curves_dict: Dict[str, np.ndarray],
                                 target_performance: float) -> Dict[str, Dict]:
        """
        Analyze sample efficiency - how quickly algorithms reach target performance

        Args:
            curves_dict: Dictionary mapping algorithm names to learning curves
            target_performance: Performance threshold to analyze

        Returns:
            Dictionary with sample efficiency statistics for each algorithm
        """
        results = {}

        for alg_name, curves in curves_dict.items():
            sample_counts = []
            final_performances = []

            for curve in curves:
                # Check final performance
                final_performances.append(curve[-1])

                # Find first timestep achieving target performance
                achieving_steps = np.where(curve >= target_performance)[0]
                if len(achieving_steps) > 0:
                    sample_counts.append(achieving_steps[0])

            # Calculate statistics
            success_rate = len(sample_counts) / len(curves)

            if sample_counts:
                mean_samples = np.mean(sample_counts)
                median_samples = np.median(sample_counts)
                std_samples = np.std(sample_counts)

                # Bootstrap CI for mean samples
                if len(sample_counts) >= 2:
                    bootstrap_means = []
                    for _ in range(1000):
                        bootstrap_sample = np.random.choice(sample_counts,
                                                          size=len(sample_counts),
                                                          replace=True)
                        bootstrap_means.append(np.mean(bootstrap_sample))

                    ci_lower = np.percentile(bootstrap_means, 100 * self.alpha / 2)
                    ci_upper = np.percentile(bootstrap_means, 100 * (1 - self.alpha / 2))
                else:
                    ci_lower = ci_upper = mean_samples
            else:
                mean_samples = median_samples = std_samples = np.inf
                ci_lower = ci_upper = np.inf

            results[alg_name] = {
                'success_rate': success_rate,
                'mean_samples_to_target': mean_samples,
                'median_samples_to_target': median_samples,
                'std_samples_to_target': std_samples,
                'ci_lower_samples': ci_lower,
                'ci_upper_samples': ci_upper,
                'final_performance_mean': np.mean(final_performances),
                'final_performance_std': np.std(final_performances),
                'n_successful_runs': len(sample_counts),
                'total_runs': len(curves)
            }

        return results

    def plateau_detection(self, curve: np.ndarray, window_size: int = 100,
                         threshold: float = 0.01) -> int:
        """
        Detect when learning plateaus (stops improving significantly)

        Args:
            curve: Single learning curve
            window_size: Window for calculating improvement rate
            threshold: Minimum improvement rate to not be considered plateau

        Returns:
            Timestep where plateau begins (or -1 if no plateau detected)
        """
        if len(curve) < window_size * 2:
            return -1

        improvements = []

        # Calculate improvement rate over sliding windows
        for i in range(window_size, len(curve) - window_size):
            before_window = curve[i-window_size:i]
            after_window = curve[i:i+window_size]

            improvement_rate = (np.mean(after_window) - np.mean(before_window)) / window_size
            improvements.append(improvement_rate)

        # Find first point where improvement consistently below threshold
        for i, improvement in enumerate(improvements):
            if improvement < threshold:
                # Check if it stays below threshold for a while
                remaining_improvements = improvements[i:i+window_size//2]
                if len(remaining_improvements) > 0 and all(imp < threshold for imp in remaining_improvements):
                    return i + window_size  # Convert back to original curve index

        return -1  # No plateau detected

    def compare_learning_efficiency(self, curves_dict: Dict[str, np.ndarray]) -> Dict:
        """
        Comprehensive comparison of learning efficiency across algorithms

        Args:
            curves_dict: Dictionary mapping algorithm names to learning curves

        Returns:
            Dictionary with comprehensive learning efficiency comparison
        """
        results = {
            'algorithms': list(curves_dict.keys()),
            'final_performance': {},
            'learning_speed': {},
            'stability': {},
            'sample_efficiency': {}
        }

        # Analyze final performance
        for alg_name, curves in curves_dict.items():
            final_scores = curves[:, -1]
            results['final_performance'][alg_name] = {
                'mean': np.mean(final_scores),
                'std': np.std(final_scores),
                'median': np.median(final_scores)
            }

        # Analyze learning speed (slope of first half)
        for alg_name, curves in curves_dict.items():
            learning_speeds = []
            midpoint = curves.shape[1] // 2

            for curve in curves:
                # Calculate slope of first half using linear regression
                x = np.arange(midpoint)
                y = curve[:midpoint]
                slope, _, _, _, _ = stats.linregress(x, y)
                learning_speeds.append(slope)

            results['learning_speed'][alg_name] = {
                'mean': np.mean(learning_speeds),
                'std': np.std(learning_speeds),
                'median': np.median(learning_speeds)
            }

        # Analyze stability (variance across timesteps)
        for alg_name, curves in curves_dict.items():
            stabilities = []

            for curve in curves:
                # Calculate coefficient of variation in second half (after learning)
                second_half = curve[len(curve)//2:]
                if np.mean(second_half) != 0:
                    cv = np.std(second_half) / abs(np.mean(second_half))
                    stabilities.append(cv)

            if stabilities:
                results['stability'][alg_name] = {
                    'mean_cv': np.mean(stabilities),
                    'std_cv': np.std(stabilities),
                    'median_cv': np.median(stabilities)
                }
            else:
                results['stability'][alg_name] = {
                    'mean_cv': np.inf,
                    'std_cv': np.inf,
                    'median_cv': np.inf
                }

        return results

# Convenience functions
def analyze_curves(curves_dict: Dict[str, np.ndarray],
                  smooth: bool = True,
                  method: str = 'bootstrap') -> Dict:
    """
    Quick analysis of learning curves

    Args:
        curves_dict: Dictionary mapping algorithm names to curves
        smooth: Whether to apply gaussian smoothing
        method: 'bootstrap' or 't_test' for confidence bands

    Returns:
        Dictionary with analysis results and processed curves
    """
    analyzer = LearningCurveAnalyzer()

    results = {
        'processed_curves': {},
        'confidence_bands': {},
        'efficiency_comparison': analyzer.compare_learning_efficiency(curves_dict)
    }

    for alg_name, curves in curves_dict.items():
        # Smooth if requested
        if smooth:
            processed_curves = analyzer.smooth_curves(curves, method='gaussian')
        else:
            processed_curves = curves

        # Calculate confidence bands
        if method == 'bootstrap':
            mean_curve, lower, upper = analyzer.bootstrap_confidence_bands(processed_curves)
        else:
            mean_curve, lower, upper = analyzer.t_test_confidence_bands(processed_curves)

        results['processed_curves'][alg_name] = processed_curves
        results['confidence_bands'][alg_name] = {
            'mean': mean_curve,
            'lower': lower,
            'upper': upper
        }

    return results

if __name__ == "__main__":
    # Test the learning curve analyzer
    print("Testing Learning Curve Analyzer...")

    np.random.seed(42)

    # Generate sample learning curves
    n_runs = 5
    n_timesteps = 1000

    # Algorithm 1: Fast learner, plateaus early
    curves1 = []
    for _ in range(n_runs):
        curve = 100 * (1 - np.exp(-np.arange(n_timesteps) / 200)) + np.random.normal(0, 5, n_timesteps)
        curves1.append(curve)

    # Algorithm 2: Slow learner, keeps improving
    curves2 = []
    for _ in range(n_runs):
        curve = 80 * (1 - np.exp(-np.arange(n_timesteps) / 500)) + np.random.normal(0, 3, n_timesteps)
        curves2.append(curve)

    curves_dict = {
        'FastLearner': np.array(curves1),
        'SlowLearner': np.array(curves2)
    }

    # Analyze curves
    analysis = analyze_curves(curves_dict)

    print("Analysis Results:")
    print("-" * 40)

    for alg in ['FastLearner', 'SlowLearner']:
        final_perf = analysis['efficiency_comparison']['final_performance'][alg]
        learning_speed = analysis['efficiency_comparison']['learning_speed'][alg]

        print(f"{alg}:")
        print(f"  Final Performance: {final_perf['mean']:.1f} ± {final_perf['std']:.1f}")
        print(f"  Learning Speed: {learning_speed['mean']:.3f} ± {learning_speed['std']:.3f}")
        print()

    print("✓ Learning curve analysis working!")
