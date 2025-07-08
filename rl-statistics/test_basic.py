import numpy as np
import sys
sys.path.append('.')

from rl_statistics.core.statistics import RLStatistics, bootstrap_ci, permutation_test, compare_algorithms

def test_bootstrap_ci():
    """Test bootstrap confidence intervals"""
    print("Testing bootstrap confidence intervals...")

    # Test with known data
    np.random.seed(42)
    data = np.random.normal(100, 10, 100)  # Mean=100, std=10

    ci_lower, ci_upper = bootstrap_ci(data)

    print(f"Data mean: {np.mean(data):.2f}")
    print(f"Bootstrap CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

    # The CI should contain the true mean (100) most of the time
    assert ci_lower < 100 < ci_upper, "CI should contain true mean"
    print("✓ Bootstrap CI test passed")

def test_permutation_test():
    """Test permutation testing"""
    print("\nTesting permutation test...")

    np.random.seed(42)

    # Test 1: Same distribution (should not be significant)
    data_a = np.random.normal(100, 10, 50)
    data_b = np.random.normal(100, 10, 50)

    p_val = permutation_test(data_a, data_b, n_permutations=1000)
    print(f"Same distribution p-value: {p_val:.4f}")

    # Should not be significant most of the time
    assert p_val > 0.01, "Same distributions should not be significantly different"

    # Test 2: Different distributions (should be significant)
    data_a = np.random.normal(100, 10, 50)
    data_b = np.random.normal(120, 10, 50)  # Different mean

    p_val = permutation_test(data_a, data_b, n_permutations=1000)
    print(f"Different distribution p-value: {p_val:.4f}")

    # Should be significant
    assert p_val < 0.05, "Different distributions should be significantly different"
    print("✓ Permutation test passed")

def test_compare_algorithms():
    """Test algorithm comparison"""
    print("\nTesting algorithm comparison...")

    np.random.seed(42)

    # Create test data
    results = {
        'PPO': np.random.normal(100, 15, 10),
        'SAC': np.random.normal(110, 12, 10),
        'TD3': np.random.normal(105, 20, 10)
    }

    comparison = compare_algorithms(results)

    print("Algorithm summaries:")
    for alg, stats in comparison['summary'].items():
        print(f"  {alg}: {stats['mean']:.2f} ± {stats['std']:.2f} "
              f"[{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}]")

    print("\nPairwise comparisons:")
    for comp, result in comparison['comparisons'].items():
        sig_marker = "***" if result['significant'] else ""
        print(f"  {comp}: p={result['p_value']:.4f} {sig_marker}")

    # Basic sanity checks
    assert len(comparison['summary']) == 3, "Should have 3 algorithms"
    assert len(comparison['comparisons']) == 3, "Should have 3 pairwise comparisons"

    print("✓ Algorithm comparison test passed")

def test_edge_cases():
    """Test edge cases"""
    print("\nTesting edge cases...")

    # Small sample size
    small_data = np.array([1.0, 2.0])
    ci_lower, ci_upper = bootstrap_ci(small_data)
    print(f"Small sample CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

    # Single value
    single_data = np.array([5.0])
    ci_lower, ci_upper = bootstrap_ci(single_data)
    assert ci_lower == ci_upper == 5.0, "Single value should have CI = value"

    print("✓ Edge cases test passed")

if __name__ == "__main__":
    print("Running basic tests for RL Statistics...")
    print("=" * 50)

    test_bootstrap_ci()
    test_permutation_test()
    test_compare_algorithms()
    test_edge_cases()

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("\nNext steps:")
    print("1. Try running the example in statistics.py")
    print("2. Test with your own data")
    print("3. Start implementing visualization functions")
