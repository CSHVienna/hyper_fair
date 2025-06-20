from mc_algo import compute_adjusted_alpha, multiple_test_measure, get_pval_cached, re_rank, GeneratedData
import numpy as np
from plotting import plot_ranking
import matplotlib.pyplot as plt


def measure_fairness_single_point(x_seq, k, omega=1, test_side='lower', alpha=None, verbose=False, plot=False):
    """
    Evaluates fairness at a single cutoff point in a ranked sequence.
    This function computes the p-value for the observed number of protected group members
    in the top-k positions of a ranking, using a statistical test (e.g., hypergeometric test).
    Optionally, it can test the null hypothesis at a given significance level (alpha), print
    verbose output, and plot the ranking with the cutoff.

    Parameters:
        x_seq (np.ndarray): A binary array indicating the ranking of items (1 for protected group, 0 for unprotected).
        k (int): Cutoff position in the ranking (number of top items to consider).
        omega (float, optional): Odds ratio for the noncentral hypergeometric distribution. If 1, uses the standard hypergeometric distribution.
        test_side (str, optional): Type of test to perform: 'lower', 'upper', or 'two-sided'. Default is 'lower'.
        alpha (float, optional): Significance level for hypothesis testing. If None, no hypothesis test is performed.
        verbose (bool, optional): If True, prints detailed output about the test result. Default is False.
        plot (bool, optional): If True, plots the ranking and highlights the cutoff position. Default is False.

    Returns:
        float: The computed p-value for the observed number of protected group members in the top-k positions.
    """

    assert isinstance(x_seq, list) or (isinstance(x_seq, np.ndarray) and x_seq.ndim == 1), "x_seq must be a 1D numpy array"
    assert isinstance(k, int) and k > 0 and k <= len(x_seq), "k must be a positive integer within the bounds of x_seq"
    assert isinstance(omega, (int, float)) and omega > 0, "omega must be a positive number"
    assert test_side in ['lower', 'upper', 'two-sided'], "test_side must be one of ['lower', 'upper', 'two-sided']"
    assert alpha is None or (isinstance(alpha, float) and 0 < alpha < 1), "alpha must be a float between 0 and 1"
    assert isinstance(verbose, bool), "verbose must be a boolean"
    assert isinstance(plot, bool), "plot must be a boolean"

    n = len(x_seq)
    n_protected = np.sum(x_seq)
    y_k = np.sum(x_seq[:k])
    pval = get_pval_cached(y_k, n, n_protected, k, test_side, cache={}, omega=omega)

    if verbose:
        if alpha is not None and pval < alpha:
            print(f"Reject null hypothesis at position {k} with p-value {pval:.4f}")
        elif alpha is not None:
            print(f"Fail to reject null hypothesis at position {k} with p-value {pval:.4f}")
        else:
            print(f"p-value at position {k} is {pval:.4f}")

    if plot:
        if alpha is not None:
            plot_ranking(x_seq, alpha, test_side, omega=omega, ax=None)
            if k!= n:
                plt.plot([k/n, k/n], [0, 1], color='red', linestyle='--', label='position k')
            plt.legend()
            plt.show()
        else:
            plot_ranking(x_seq, None, None, omega=omega, ax=None)
            if k!= n:
                plt.plot([k/n, k/n], [0, 1], color='red', linestyle='--', label='position k')
            plt.legend()
            plt.show()

    return pval


def measure_fairness_multiple_points(x_seq, k, test_side='lower', alpha=None, n_exp=10000, omega=1, quantile='analytical', cache={}, verbose=False, plot=False, seed=None, generatedData=None):
    """
    Measures the fairness of a sequence at multiple points using statistical hypothesis testing.
    This function evaluates the fairness of a given sequence `x_seq` at all subsets [1:j] for j in [1, k].
    It can optionally plot the results and provide verbose output. 
    The function supports different test sides ('lower', 'upper', 'two-sided'), 
    allows for analytical or empirical quantile calculation, and can adjust for multiple testing.
    
    Parameters:
        x_seq (np.ndarray): A binary array indicating the ranking of items (1 for protected group, 0 for unprotected).
        k (int): The number of top items to consider for the test, and consequently the number of tests performed.
        test_side (str, optional): The side of the test to perform. Options are 'lower', 'upper', or 'two-sided'. Default is 'lower'.
        alpha (float, optional): The significance level for the hypothesis test. If None, only the p-value is returned. Default is None.
        n_exp (int, optional): Number of Monte Carlo simulations to run for p-value calculation. Default is 10000.
        omega (float, optional): Odds ratio for the noncentral hypergeometric distribution. If 1, uses the standard hypergeometric distribution.
        quantile (str, optional): Method for quantile calculation. Options are 'analytical' or 'empirical'. Default is 'analytical'.
        cache (dict, optional): Dictionary for caching the pvalue calculations for the analytical case. Default is empty dict.
        verbose (bool, optional): If True, prints detailed output about the hypothesis test result. Default is False.
        plot (bool, optional): If True, plots the ranking and test results. Default is False.
        seed (int, optional): Random seed for reproducibility. Default is None.
        generatedData (GeneratedData, optional): Precomputed data for efficiency. Default is None.
        If provided, it should match the input parameters.

    Returns:
        fairness_score (float): The computed p-value or fairness score for the multiple tests.
        generatedData (GeneratedData): The generated data used for the tests.
    """

    assert isinstance(x_seq, list) or (isinstance(x_seq, np.ndarray) and x_seq.ndim == 1), "x_seq must be a 1D numpy array"
    assert isinstance(k, int) and k > 0 and k <= len(x_seq), "k must be a positive integer within the bounds of x_seq"
    assert isinstance(omega, (int, float)) and omega > 0, "omega must be a positive number"
    assert test_side in ['lower', 'upper', 'two-sided'], "test_side must be one of ['lower', 'upper', 'two-sided']"
    assert alpha is None or (isinstance(alpha, float) and 0 < alpha < 1), "alpha must be a float between 0 and 1"
    assert isinstance(n_exp, int) and n_exp > 0, "n_exp must be a positive integer"
    assert quantile in ['analytical', 'empirical'], "quantile must be one of ['analytical', 'empirical']"
    assert isinstance(cache, dict), "cache must be a dictionary"
    assert isinstance(verbose, bool), "verbose must be a boolean"
    assert isinstance(plot, bool), "plot must be a boolean"
    assert seed is None or isinstance(seed, int), "seed must be an integer or None"
    assert generatedData is None or isinstance(generatedData, GeneratedData), "generatedData must be an instance of GeneratedData or None"
    
    n = len(x_seq)
    n_protected = int(np.sum(x_seq))
    if generatedData is not None:
        assert generatedData.n == n and generatedData.n_protected == n_protected and generatedData.k >= k and generatedData.omega == omega and generatedData.n_exp == n_exp, "generatedData does not match the input parameters"

    if plot and alpha is not None:
        output_dict, generatedData = compute_adjusted_alpha(len(x_seq), int(np.sum(x_seq)), alpha, k=k, omega=omega, generatedData=generatedData, n_exp=n_exp, test_side=test_side, quantile=quantile, cache=cache, seed=seed)
        fairness_score, generatedData = multiple_test_measure(x_seq, k, omega=omega, generatedData=generatedData, test_side=test_side, quantile=quantile, n_exp=n_exp, cache=cache)
    else:
        fairness_score, generatedData = multiple_test_measure(x_seq, k, omega=omega, generatedData=generatedData, test_side=test_side, quantile=quantile, n_exp=n_exp, cache=cache, seed=seed)

    if verbose:
        if alpha is not None and fairness_score < alpha:
            print(f"Reject null hypothesis with p-value {fairness_score:.4f}")
        elif alpha is not None:
            print(f"Fail to reject null hypothesis with p-value {fairness_score:.4f}")
        else:
            print(f"p-value is {fairness_score:.4f}")

    if plot:
        if alpha is not None:
            adjusted_alpha = output_dict[k]['adjusted alpha']
            plot_ranking(x_seq, adjusted_alpha, test_side, omega=omega, ax=None)
            if k != len(x_seq):
                plt.plot([k/len(x_seq), k/len(x_seq)], [0, 1], color='red', linestyle='--', label='position k')
            plt.legend()
            plt.show()
        else:
            plot_ranking(x_seq, None, None, omega=omega, ax=None)
            if k != len(x_seq):
                plt.plot([k/len(x_seq), k/len(x_seq)], [0, 1], color='red', linestyle='--', label='position k')
            plt.legend()
            plt.show()

    return fairness_score, generatedData


def adjust_ranking(x_seq, ids, k, alpha, omega=1, test_side='lower', n_exp=10000, quantile='analytical', cache={}, plot=False, seed=None, generatedData=None):
    """
    Adjusts the ranking of a sequence based on the fairness measure.
    This function computes the adjusted alpha for the given sequence and re-ranks it accordingly.
    It can optionally plot the adjusted ranking and the original ranking for comparison.

    Parameters:
        x_seq (np.ndarray): A binary array indicating the ranking of items (1 for protected group, 0 for unprotected).
        ids (np.ndarray): The array of IDs corresponding to each item in x_seq.
        k (int): The number of top items to consider for the adjustment.
        alpha (float): The significance level for the hypothesis test.
        omega (float, optional): Odds ratio for the noncentral hypergeometric distribution. If 1, uses the standard hypergeometric distribution.
        test_side (str, optional): The side of the test to perform. Options are 'lower', 'upper', or 'two-sided'. Default is 'lower'.
        n_exp (int, optional): Number of Monte Carlo simulations to run for p-value calculation. Default is 10000.
        quantile (str, optional): Method for quantile calculation. Options are 'analytical' or 'empirical'. Default is 'analytical'.
        cache (dict, optional): Dictionary for caching the pvalue calculations for the analytical case. Default is empty dict.
        plot (bool, optional): If True, plots the adjusted ranking and original ranking. Default is False.
        seed (int, optional): Random seed for reproducibility. Default is None.
        generatedData (GeneratedData, optional): Precomputed data for efficiency. Default is None.
        If provided, it should match the input parameters.

    Returns:
        adjusted_alpha (float): The adjusted alpha value for the ranking.
        corrected_ranking (np.ndarray): The adjusted ranking of the sequence.
        corrected_ids (np.ndarray): The IDs corresponding to the adjusted ranking.
    """

    assert isinstance(x_seq, list) or (isinstance(x_seq, np.ndarray) and x_seq.ndim == 1), "x_seq must be a 1D numpy array"
    assert isinstance(x_seq, list) or (isinstance(ids, np.ndarray) and ids.ndim == 1), "ids must be a 1D numpy array"
    assert isinstance(k, int) and k > 0 and k <= len(x_seq), "k must be a positive integer within the bounds of x_seq"
    assert isinstance(alpha, float) and 0 < alpha < 1, "alpha must be a float between 0 and 1"
    assert isinstance(omega, (int, float)) and omega > 0, "omega must be a positive number"
    assert test_side in ['lower', 'upper', 'two-sided'], "test_side must be one of ['lower', 'upper', 'two-sided']"
    assert isinstance(n_exp, int) and n_exp > 0, "n_exp must be a positive integer"
    assert quantile in ['analytical', 'empirical'], "quantile must be one of ['analytical', 'empirical']"
    assert isinstance(cache, dict), "cache must be a dictionary"
    assert isinstance(plot, bool), "plot must be a boolean"
    assert seed is None or isinstance(seed, int), "seed must be an integer or None"
    assert generatedData is None or isinstance(generatedData, GeneratedData), "generatedData must be an instance of GeneratedData or None"
    
    n = len(x_seq)
    n_protected = int(np.sum(x_seq))
    if generatedData is not None:
        assert generatedData.n == n and generatedData.n_protected == n_protected and generatedData.k >= k and generatedData.omega == omega and generatedData.n_exp == n_exp, "generatedData does not match the input parameters"

    output_dict, generatedData = compute_adjusted_alpha(n, n_protected, alpha, k=k, omega=omega, generatedData=generatedData, n_exp=n_exp, test_side=test_side, quantile=quantile, cache=cache, seed=seed)
    adjusted_alpha = output_dict[k]['adjusted alpha']

    corrected_x, corrected_ids = re_rank(x_seq, ids, adjusted_alpha, k, test_side=test_side, omega=omega)

    if plot:
        plot_ranking(x_seq, adjusted_alpha, test_side, omega=omega, ax=None)
        corrected_proportions = np.cumsum(corrected_x) / np.arange(1, n+1)
        proportions = np.cumsum(x_seq) / np.arange(1, n+1)
        if not all(x_seq==corrected_x):
            plt.plot(np.arange(1/n,1+1/n,1/n), corrected_proportions, "#de5805", label = 'Corrected ranking')
        plt.plot(np.arange(1/n,1+1/n,1/n), proportions, '#018571')
        if k != len(x_seq):
            plt.plot([k/len(x_seq), k/len(x_seq)], [0, 1], color='red', linestyle='--', label='position k')
        plt.legend()
        plt.show()

    return corrected_x, corrected_ids, generatedData