import numpy as np
import sys
sys.path.append('.')

from rl_statistics.core.curves import LearningCurveAnalyzer, analyze_curves

def test_curve_analyzer():
    """Test basic curve analyzer functionality"""
    print("Testing Learning Curve Analyzer...")

    np.random.seed(42)

    # Generate simple test curves
    n_runs = 5
    n_timesteps = 100

    # Simple upward trend with noise
    curves = []
    for _ in range(n_runs):
        base_curve = np.linspace(0, 100, n_timesteps)
        noise = np.random.normal(0, 5, n_timesteps)
        curves.append(base_curve + noise)

    curves = np.array(curves)

    analyzer = LearningCurveAnalyzer()

    # Test confidence bands
    mean_curve, lower_band, upper_band = analyzer.bootstrap_confidence_bands(curves)

    print(f"Mean curve shape: {mean_curve.shape}")
    print(f"Lower band shape: {lower_band.shape}")
    print(f"Upper band shape: {upper_band.shape}")

    # Sanity checks
    assert mean_curve.shape == (n_timesteps,), "Mean curve should have correct shape"
    assert lower_band.shape == (n_timesteps,), "Lower band should have correct shape"
    assert upper_band.shape == (n_timesteps,), "Upper band should have correct shape"

    # Lower should be less than upper
    assert np.all(lower_band <= upper_band), "Lower band should be ≤ upper band"

    # Mean should generally be between bands
    between_bands = np.mean((lower_band <= mean_curve) & (mean_curve <= upper_band))
    assert between_bands > 0.8, "Mean should be between bands most of the time"

    print("✓ Basic curve analyzer tests passed")

def test_sample_efficiency():
    """Test sample efficiency analysis"""
    print("\nTesting sample efficiency analysis...")

    np.random.seed(42)

    # Create two algorithms with different learning speeds
    n_runs = 3
    n_timesteps = 200

    # Fast learner
    fast_curves = []
    for _ in range(n_runs):
        curve = 50 * (1 - np.exp(-np.arange(n_timesteps) / 30)) + np.random.normal(0, 2, n_timesteps)
        fast_curves.append(curve)

    # Slow learner
    slow_curves = []
    for _ in range(n_runs):
        curve = 45 * (1 - np.exp(-np.arange(n_timesteps) / 80)) + np.random.normal(0, 2, n_timesteps)
        slow_curves.append(curve)

    curves_dict = {
        'Fast': np.array(fast_curves),
        'Slow': np.array(slow_curves)
    }

    analyzer = LearningCurveAnalyzer()

    # Test sample efficiency
    target_performance = 30
    efficiency = analyzer.sample_efficiency_analysis(curves_dict, target_performance)

    print(f"Target performance: {target_performance}")
    print(f"Fast learner success rate: {efficiency['Fast']['success_rate']:.2f}")
    print(f"Slow learner success rate: {efficiency['Slow']['success_rate']:.2f}")

    # Fast learner should reach target sooner
    if efficiency['Fast']['success_rate'] > 0 and efficiency['Slow']['success_rate'] > 0:
        fast_samples = efficiency['Fast']['mean_samples_to_target']
        slow_samples = efficiency['Slow']['mean_samples_to_target']

        print(f"Fast learner mean samples: {fast_samples:.0f}")
        print(f"Slow learner mean samples: {slow_samples:.0f}")

        assert fast_samples < slow_samples, "Fast learner should reach target sooner"

    print("✓ Sample efficiency tests passed")

def test_smoothing():
    """Test curve smoothing"""
    print("\nTesting curve smoothing...")

    np.random.seed(42)

    # Generate noisy curves
    n_runs = 3
    n_timesteps = 100

    curves = []
    for _ in range(n_runs):
        base_curve = np.linspace(0, 100, n_timesteps)
        noise = np.random.normal(0, 10, n_timesteps)  # High noise
        curves.append(base_curve + noise)

    curves = np.array(curves)

    analyzer = LearningCurveAnalyzer()

    # Test Gaussian smoothing
    smoothed_gaussian = analyzer.smooth_curves(curves, method='gaussian', window_size=10)

    # Test rolling smoothing
    smoothed_rolling = analyzer.smooth_curves(curves, method='rolling', window_size=10)

    # Test no smoothing
    smoothed_none = analyzer.smooth_curves(curves, method='none')

    print(f"Original curves shape: {curves.shape}")
    print(f"Gaussian smoothed shape: {smoothed_gaussian.shape}")
    print(f"Rolling smoothed shape: {smoothed_rolling.shape}")
    print(f"No smoothing shape: {smoothed_none.shape}")

    # Shapes should be preserved
    assert smoothed_gaussian.shape == curves.shape, "Gaussian smoothing should preserve shape"
    assert smoothed_rolling.shape == curves.shape, "Rolling smoothing should preserve shape"
    assert np.array_equal(smoothed_none, curves), "No smoothing should return original"

    # Smoothed curves should be less noisy (lower variance)
    original_var = np.var(curves)
    gaussian_var = np.var(smoothed_gaussian)
    rolling_var = np.var(smoothed_rolling)

    print(f"Original variance: {original_var:.2f}")
    print(f"Gaussian smoothed variance: {gaussian_var:.2f}")
    print(f"Rolling smoothed variance: {rolling_var:.2f}")

    assert gaussian_var < original_var, "Gaussian smoothing should reduce variance"
    assert rolling_var < original_var, "Rolling smoothing should reduce variance"

    print("✓ Smoothing tests passed")

def test_analyze_curves_function():
    """Test the convenience function"""
    print("\nTesting analyze_curves convenience function...")

    np.random.seed(42)

    # Generate test data
    n_runs = 4
    n_timesteps = 50

    curves_dict = {}
    for alg_name in ['Alg1', 'Alg2']:
        curves = []
        for _ in range(n_runs):
            base_curve = np.linspace(0, 100, n_timesteps)
            noise = np.random.normal(0, 5, n_timesteps)
            curves.append(base_curve + noise)
        curves_dict[alg_name] = np.array(curves)

    # Test the convenience function
    results = analyze_curves(curves_dict, smooth=True, method='bootstrap')

    print("Results keys:", list(results.keys()))
    print("Processed curves keys:", list(results['processed_curves'].keys()))
    print("Confidence bands keys:", list(results['confidence_bands'].keys()))

    # Check structure
    assert 'processed_curves' in results, "Should have processed curves"
    assert 'confidence_bands' in results, "Should have confidence bands"
    assert 'efficiency_comparison' in results, "Should have efficiency comparison"

    # Check each algorithm
    for alg_name in ['Alg1', 'Alg2']:
        assert alg_name in results['processed_curves'], f"Should have {alg_name} processed curves"
        assert alg_name in results['confidence_bands'], f"Should have {alg_name} confidence bands"

        bands = results['confidence_bands'][alg_name]
        assert 'mean' in bands, "Should have mean curve"
        assert 'lower' in bands, "Should have lower band"
        assert 'upper' in bands, "Should have upper band"

    print("✓ Convenience function tests passed")

if __name__ == "__main__":
    print("Running learning curve tests...")
    print("=" * 50)

    test_curve_analyzer()
    test_sample_efficiency()
    test_smoothing()
    test_analyze_curves_function()

    print("\n" + "=" * 50)
    print("All learning curve tests passed! ✓")
    print("\nNext steps:")
    print("1. Run the learning curves demo: python examples/learning_curves_demo.py")
    print("2. Try with your own learning curve data")
    print("3. Experiment with different smoothing methods")
