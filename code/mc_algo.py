import numpy as np
import scipy.stats as stats
from collections import deque

# Functions to generate sequences

def hypergeometric_sequences_fast(n, n_protected, n_exp=1000, seed=None):
    """
    Generates multiple random permutations of a binary sequence representing a hypergeometric distribution.

    Parameters:
        n (int): Total number of elements in the sequence.
        n_protected (int): Number of elements set to 1 (protected group).
        n_exp (int, optional): Number of random permutations (experiments) to generate. Default is 1000.

    Returns:
        np.ndarray: A 2D array of shape (n_exp, n), where each row is a random permutation of the original sequence
                    containing `n_protected` ones and `n - n_protected` zeros.
    """

    rng = np.random.default_rng(seed)
    seq = np.concatenate((np.ones(n_protected, dtype=int), np.zeros(n - n_protected, dtype=int)))
    rand_keys = rng.random((n_exp, n))
    idx = np.argsort(rand_keys, axis=1)
    return seq[idx]

def wallenius_sequences_fast(n, n_protected, omega, k, n_exp=1000, seed=None):
    """
    Generate sequences of draws from Wallenius' non-central hypergeometric distribution.
    This function simulates multiple independent experiments, each consisting of a sequence of draws without replacement from an urn containing two types of balls (protected and unprotected). The probability of drawing a protected ball at each step is proportional to a specified odds ratio (omega).
    
    Parameters:
        n (int): Total number of balls in the urn.
        n_protected (int): Number of protected balls in the urn.
        omega (float): Odds ratio for drawing a protected ball (omega > 0).
        k (int): Number of draws per experiment (sequence length).
        n_exp (int, optional): Number of independent experiments (sequences) to generate. Default is 1000.
        seed (int or None, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: A 2D array of shape (n_exp, k), where each row corresponds to one experiment.
                    Each element is either 1 (protected ball drawn) or 0 (unprotected ball drawn).
    """

    rng = np.random.default_rng(seed)
    x1 = np.full(n_exp, n_protected, dtype=int)
    x2 = np.full(n_exp, n-n_protected, dtype=int)
    seqs = np.empty((n_exp, k), dtype=int)
    for i in range(k):
        probs = omega * x1 / (omega * x1 + x2)
        U = rng.random(n_exp)
        draw1 = U < probs 
        seqs[:, i] = draw1.astype(int)
        x1[draw1] -= 1
        x2[~draw1] -= 1
    return seqs


# Functions to calculate p-values

def get_pval_cached(y, n, n_protected, k, test_side, cache, omega=1):
    """
    Calculate p-values based on the hypergeometric and Wallenius non-central hypergeometric distributions.
    This function caches results to avoid redundant calculations.

    Parameters:
        y (int): The observed value.
        n (int): Total number of elements in the population.
        n_protected (int): Number of elements in the protected group.
        k (int): Number of draws.
        test_side (str): The side of the test ('lower', 'upper', or 'two-sided').
        cache (dict): A dictionary to store previously computed p-values.
        omega (float, optional): Odds ratio for Wallenius' distribution. Default is 1.

    Returns:
        float: The p-value for the observed value.
    """

    key = (y, n, n_protected, k, test_side, omega)
    if key in cache:
        return cache[key]
    else:
        if test_side == 'lower':
            if omega == 1:
                val = stats.hypergeom.cdf(y, n, n_protected, k)
            else:
                val = stats.nchypergeom_wallenius.cdf(y, n, n_protected, k, omega)
        elif test_side == 'upper':
            if omega == 1:
                val = 1 - stats.hypergeom.cdf(y - 1, n, n_protected, k)
            else:
                val = 1 - stats.nchypergeom_wallenius.cdf(y - 1, n, n_protected, k, omega)
        elif test_side == 'two-sided':
            if omega == 1:
                lower = stats.hypergeom.cdf(y, n, n_protected, k)
                upper = 1 - stats.hypergeom.cdf(y - 1, n, n_protected, k)
            else:
                lower = stats.nchypergeom_wallenius.cdf(y, n, n_protected, k, omega)
                upper = 1 - stats.nchypergeom_wallenius.cdf(y - 1, n, n_protected, k, omega)
            val = 2 * min(lower, upper)
        cache[key] = val
        return val
    
def calculate_analytical_pval(y, n_exp, n, n_protected, k, test_side, cache, omega=1):
    """
    Calculate analytical p-values for the whole array of observed cumulative sums y.
    This function caches results to avoid redundant calculations.

    Parameters:
        y (np.ndarray): A 2D array of observed values.
        n_exp (int): Number of experiments (rows in y).
        n (int): Total number of elements in the population.
        n_protected (int): Number of elements in the protected group.
        k (int): Number of draws.
        test_side (str): The side of the test ('lower', 'upper', or 'two-sided').
        cache (dict): A dictionary to store previously computed p-values.

    Returns:
        np.ndarray: A 2D array of p-values, where each row corresponds to an experiment.
    """

    pvals = np.zeros((n_exp, k), float)
    for j_i in range(k):
        unique_values, indices = np.unique(y[:, j_i], return_inverse=True)
        unique_pvals = np.array([get_pval_cached(val, n, n_protected, j_i + 1, test_side, cache, omega) for val in unique_values])
        pvals[:, j_i] = unique_pvals[indices]
    return pvals
    
def calculate_empirical_pval(y, n_exp, test_side):
    """
    Calculate empirical p-values based on the ranks of the observed values.
    This function computes the empirical p-values for each observed value in y.

    Parameters:
        y (np.ndarray): A 2D array of observed values.
        n_exp (int): Number of experiments (rows in y).
        test_side (str): The side of the test ('lower', 'upper', or 'two-sided').

    Returns:
        np.ndarray: A 2D array of p-values, where each row corresponds to an experiment.
    """

    if test_side == 'lower':
        pvals = (stats.rankdata(y, method='max', axis=0)) / n_exp
    elif test_side == 'upper':
        pvals = (stats.rankdata(-y, method='max', axis=0)) / n_exp
    elif test_side == 'two-sided':
        pvals_low = (stats.rankdata(y, method='max', axis=0)) / n_exp
        pvals_up = (stats.rankdata(-y, method='max', axis=0)) / n_exp
        pvals = 2 * np.min([pvals_low, pvals_up], axis=0)
    return pvals


# Class to encapsulate the generated data and its properties

class GeneratedData:
    """
    A class for generating and analyzing sequences based on hypergeometric or Wallenius distributions, 
    with support for p-value calculations.
    Attributes:
        n (int): Total number of elements in each sequence.
        n_protected (int): Number of protected elements in each sequence.
        k (int): Number of draws or selections per experiment.
        omega (float, optional): Odds ratio for Wallenius distribution. Defaults to 1 (hypergeometric).
        n_exp (int, optional): Number of experiments (sequences) to generate. Defaults to 1000.
        x (np.ndarray or None): Generated sequences. None if not yet generated.
        y (np.ndarray or None): Cumulative sums of sequences. None if not yet calculated.
        pvals_dict (dict or None): Dictionary containing p-value calculation method and results.
        cache (dict or None): Optional cache for analytical p-value calculations.
    Methods:
        generate_sequences():
            Generates sequences using either the hypergeometric or Wallenius distribution, 
            depending on the value of omega.
        calculate_cumulative_sums():
            Calculates the cumulative sums of the generated sequences and stores them in `y`.
        calculate_pvals(test_side):
            Calculates p-values for the specified test side ('lower', 'upper', or 'two-sided') 
            using either empirical or analytical methods, as specified in `pvals_dict`.
        __str__():
            Returns a string representation of the GeneratedData instance.
        get_info():
            Prints information about the current state of the data, including whether sequences 
            and cumulative sums have been generated, and which p-values have been calculated.
    """
    def __init__(self, n, n_protected, k, omega=1, n_exp=1000, x=None, y=None, pvals_dict=None, cache=None, seed=None):
        assert isinstance(n, int) and n > 0, "n must be a positive integer"
        assert isinstance(n_protected, int) and 0 < n_protected < n, "n_protected must be an integer between 0 and n (excluded)"
        assert isinstance(k, int) and 1 <= k <= n, "k must be an integer between 1 and n"
        assert isinstance(omega, (int, float)) and omega > 0, "omega must be a positive number"
        assert isinstance(n_exp, int) and n_exp > 0, "n_exp must be a positive integer"
        assert x is None or isinstance(x, np.ndarray), "x must be a numpy array or None"
        assert y is None or isinstance(y, np.ndarray), "y must be a numpy array or None"
        assert pvals_dict is None or (isinstance(pvals_dict, dict) and 'quantile' in pvals_dict), "pvals_dict must be a dictionary containing the key 'quantile' or None"
        assert cache is None or isinstance(cache, dict), "cache must be a dictionary or None"
        assert seed is None or isinstance(seed, int), "seed must be an integer or None"

        self.n = n
        self.n_protected = n_protected
        self.k = k
        self.omega = omega
        self.n_exp = n_exp
        self.x = x
        self.y = y
        self.pvals_dict = pvals_dict
        self.cache = cache
        self.seed = seed

    def generate_sequences(self):
        if self.omega==1 and self.k==self.n:
            self.x = hypergeometric_sequences_fast(self.n, self.n_protected, self.n_exp, self.seed)
        else:
            self.x = wallenius_sequences_fast(self.n, self.n_protected, self.omega, self.k, self.n_exp, self.seed)

    def calculate_cumulative_sums(self):
        if self.x is None:
            self.generate_sequences()
        self.y = np.cumsum(self.x, axis=1)

    def calculate_pvals(self, test_side, quantile='analytical'):
        if self.pvals_dict is None or self.pvals_dict['quantile'] != quantile:
            self.pvals_dict = {'quantile': quantile}
        if self.y is None:
            self.calculate_cumulative_sums()
        if self.pvals_dict['quantile'] == 'empirical':
            self.pvals_dict[test_side] = calculate_empirical_pval(self.y[:,:self.k], self.n_exp, test_side)
        elif self.pvals_dict['quantile'] == 'analytical':
            self.pvals_dict[test_side] = calculate_analytical_pval(self.y, self.n_exp, self.n, self.n_protected, self.k, test_side, self.cache, self.omega)

    def __str__(self):
        return f"GeneratedData(n={self.n}, n_protected={self.n_protected}, k={self.k}, omega={self.omega}, n_exp={self.n_exp})"

    def get_info(self):
        print(self)
        if self.x is None:
            print("The sequences have not been generated yet.")
        elif self.y is None:
            print("The cumulative sums have not been calculated yet.")
        else:
            print(f"Generated {self.n_exp} sequences of length {self.n} with {self.n_protected} protected elements.")
        if 'lower' in self.pvals_dict or 'upper' in self.pvals_dict or 'two-sided' in self.pvals_dict:
            sides_tested = [side for side in ['lower', 'upper', 'two-sided'] if side in self.pvals_dict]
            print(f"Calculated the {self.pvals_dict['quantile']} p-values for the following test sides: {', '.join(sides_tested)}.")


# Function to compute adjusted alpha

def compute_adjusted_alpha(n, n_protected, alpha, k=None, omega=1, generatedData = None, test_side='lower', quantile='analytical', n_exp=1000, cache={}, verbose=False, seed=None):
    """
    Computes an adjusted significance level (alpha) for multiple hypothesis testing.
    Parameters:
        n (int): Total number of samples.
        n_protected (int): Number of protected samples.
        alpha (float): Desired significance level (e.g., 0.05).
        k (int or list of int, optional): Number(s) of hypotheses to consider. If None, defaults to n.
        omega (float, optional): Odds ratio for Wallenius distribution. Default is 1 (hypergeometric).
        generatedData (GeneratedData, optional): Pre-generated data object. If None, data is generated internally.
        test_side (str, optional): Specifies the test side ('lower', 'upper', etc.). Default is 'lower'.
        quantile (str, optional): Method for quantile calculation ('analytical', etc.). Default is 'analytical'.
        n_exp (int, optional): Number of experiments (sequences) to generate. Defaults to 1000.
        cache (dict, optional): Optional cache for resuing analytical p-values. 
        verbose (bool, optional): If True, prints detailed output. Default is False.
        seed (int, optional): Random seed for reproducibility. Default is None.
    Returns:
        output_dict (dict): Dictionary mapping each k_i to a dictionary with:
            - 'adjusted alpha': The adjusted significance level for k_i.
            - 'p fail alpha': Probability of failing the test at the original alpha.
            - 'p fail adjusted alpha': Probability of failing the test at the adjusted alpha.
        generatedData (GeneratedData): The (possibly newly created) GeneratedData object used for calculations.
    Notes:
        - The function simulates p-values and computes the minimum p-value across k_i hypotheses for each experiment.
        - The adjusted alpha is chosen such that the probability of the minimum p-value being below it matches the desired alpha.
        - Used for controlling the family-wise error rate in multiple testing scenario.
    """
    if k is None: k = n
    if type(k)==int: 
        k=[k]
    k_max = max(k)
    output_dict = {k_i:{'adjusted alpha':np.nan, 'p fail alpha':np.nan, 'p fail adjusted alpha':np.nan} for k_i in k}

    if generatedData is None or generatedData.n != n or generatedData.n_protected != n_protected or generatedData.k < k_max or generatedData.omega != omega or generatedData.n_exp != n_exp:
        generatedData = GeneratedData(n, n_protected, k_max, omega, n_exp, pvals_dict={'quantile': quantile}, cache=cache, seed=seed)
    if generatedData.pvals_dict['quantile'] != quantile or test_side not in generatedData.pvals_dict:
        generatedData.calculate_pvals(test_side, quantile)
    pvals = generatedData.pvals_dict[test_side]

    for k_i in k:
        z = np.min(pvals[:,:k_i], axis=1)
        adjusted_alpha = np.quantile(z, alpha, method='lower')
        output_dict[k_i]['adjusted alpha'] = adjusted_alpha

        p_fail_alpha = np.mean(z < alpha)
        output_dict[k_i]['p fail alpha'] = p_fail_alpha

        p_fail_adjusted_alpha = np.mean(z < adjusted_alpha)
        output_dict[k_i]['p fail adjusted alpha'] = p_fail_adjusted_alpha
        if verbose: 
            print('Probability of failing the test for alpha ',alpha,':', p_fail_alpha)
            print('Adjusted alpha:', adjusted_alpha, '; probability of failing the test with adjusted alpha:', p_fail_adjusted_alpha)

    return output_dict, generatedData
   

# Function to compute fairness measure for multiple tests

def multiple_test_measure(x_seq, k, omega=1, generatedData=None, test_side='lower', quantile='empirical', n_exp=1000, cache={}, seed=None):
    """
    Computes a fairness measure based on the minimum p-value across multiple tests.
    This function compares the observed sequence to a null distribution generated by either the hypergeometric or Wallenius distribution.

    Parameters:
        x_seq (np.ndarray): A binary sequence of length n, where 1 represents a protected individual.
        k (int): The position up to which to compute the p-value.
        omega (float, optional): Odds ratio for Wallenius distribution. Default is 1 (hypergeometric).
        generatedData (GeneratedData, optional): Pre-generated data object. If None, data is generated internally.
        test_side (str, optional): Specifies the test side ('lower', 'upper', etc.). Default is 'lower'.
        quantile (str, optional): Method for quantile calculation ('empirical', 'analytical'). Default is 'empirical'.
        n_exp (int, optional): Number of experiments (sequences) to generate. Defaults to 1000.
        cache (dict, optional): Optional cache for resuing analytical p-values.
        seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
        fairness_score (float): The fairness score, representing the proportion of experiments where the minimum p-value is less than or equal to the observed p-value.
        generatedData (GeneratedData): The (possibly newly created) GeneratedData object used for calculations.
    """

    n = len(x_seq)
    n_protected = int(np.sum(x_seq))
    y_seq = np.cumsum(x_seq)

    if generatedData is None or generatedData.n != n or generatedData.n_protected != n_protected or generatedData.k != k or generatedData.omega != omega or generatedData.n_exp != n_exp:
        generatedData = GeneratedData(n, n_protected, k, omega, n_exp, pvals_dict={'quantile': quantile}, cache=cache, seed=seed)
    if generatedData.pvals_dict['quantile'] != quantile or test_side not in generatedData.pvals_dict:
        generatedData.calculate_pvals(test_side, quantile)
    y = generatedData.y
    pvals = generatedData.pvals_dict[test_side]

    if quantile == 'empirical':
        if test_side == 'lower':
            pval_seq = np.array([np.mean(y_seq[i] < y[:,i]) for i in range(k)])
        elif test_side == 'upper':
            pval_seq = np.array([np.mean(-y_seq[i] < -y[:,i]) for i in range(k)])
        elif test_side == 'two-sided':
            pval_seq_low = np.array([np.mean(y_seq[i] < y[:,i]) for i in range(k)])
            pval_seq_up = np.array([np.mean(-y_seq[i] < -y[:,i]) for i in range(k)])
            pval_seq = 2 * np.min([pval_seq_low, pval_seq_up], axis=0)
    elif quantile == 'analytical':
        pval_seq = np.array([get_pval_cached(y_seq[i], n, n_protected, i+1, test_side, cache, omega) for i in range(k)])
    
    z_seq = np.min(pval_seq)
    z = np.min(pvals[:,:k], axis=1)
    fairness_score = np.mean(z <= z_seq)
    return fairness_score, generatedData


# Function to re-rank the sequence based on adjusted alpha

def re_rank(ranking, ids, adjusted_alpha, k, test_side='lower', omega=1):
    """
    Adjusts a binary ranking to satisfy fairness constraints based on the hypergeometric distribution.

    This function modifies the input `ranking` (a binary array indicating protected group membership)
    and the corresponding `ids` array to ensure that, for the top-k positions, the number of protected
    group members falls within statistically determined lower and upper bounds. The bounds are computed
    using the hypergeometric distribution and the specified significance level (`adjusted_alpha`).

    Parameters:
        ranking (np.ndarray): A binary array indicating the ranking of items (1 for protected group, 0 for unprotected).
        ids (np.ndarray): An array of item identifiers corresponding to the ranking.
        adjusted_alpha (float): The significance level for the hypergeometric distribution.
        k (int): The number of top positions to consider for the fairness constraints.
        test_side (str): The side of the test ('lower', 'upper', or 'two-sided').
        omega (float, optional): Odds ratio for Wallenius distribution. Default is 1 (hypergeometric).

    Returns:
        ranking (np.ndarray): The adjusted binary ranking array after re-ranking.
        ids (np.ndarray): The adjusted array of item identifiers after re-ranking.

    Notes:
        The function uses the hypergeometric distribution to compute the minimum and/or maximum number of
        protected group members allowed in the top-k positions, according to the specified significance level.
        Items are swapped within the ranking to satisfy these constraints, if possible.
    """

    if k is None: k = len(ranking)
    ranking = ranking.copy()
    ids = ids.copy()
    n = len(ranking)
    n_protected = sum(ranking)
    if test_side == 'lower':
        if omega == 1:
            lower_limit = np.array([stats.hypergeom.ppf(adjusted_alpha, n, n_protected, i+1) for i in range(k)])
        else:
            lower_limit = np.array([stats.nchypergeom_wallenius.ppf(adjusted_alpha, n, n_protected, i+1, omega) for i in range(k)])
        upper_limit = np.array([n_protected for i in range(k)])
    elif test_side == 'upper':
        lower_limit = np.array([0 for i in range(k)])
        if omega == 1:
            upper_limit = np.array([stats.hypergeom.ppf(1-adjusted_alpha, n, n_protected, i+1) for i in range(k)])
        else:
            upper_limit = np.array([stats.nchypergeom_wallenius.ppf(1-adjusted_alpha, n, n_protected, i+1, omega) for i in range(k)])
    elif test_side == 'two-sided':
        if omega == 1:
            lower_limit = np.array([stats.hypergeom.ppf(adjusted_alpha/2, n, n_protected, i+1) for i in range(k)])
            upper_limit = np.array([stats.hypergeom.ppf(1-adjusted_alpha/2, n, n_protected, i+1) for i in range(k)])
        else:
            lower_limit = np.array([stats.nchypergeom_wallenius.ppf(adjusted_alpha/2, n, n_protected, i+1, omega) for i in range(k)])
            upper_limit = np.array([stats.nchypergeom_wallenius.ppf(1-adjusted_alpha/2, n, n_protected, i+1, omega) for i in range(k)])
            
    q_ones = deque(i for i in range(n) if ranking[i] == 1)
    q_zeros = deque(i for i in range(n) if ranking[i] == 0)
    cum_sum = 0
    for i, x in enumerate(ranking[:k]):
        cum_sum = cum_sum + x
        if cum_sum < lower_limit[i]:
            if q_ones:
                new_id = q_ones.popleft()
                ranking[i] = 1
                ranking[new_id] = 0
                ids[i:new_id+1] = np.roll(ids[i:new_id+1], 1)
                cum_sum += 1
            else:
                break
        elif cum_sum > upper_limit[i]:
            if q_zeros:
                new_id = q_zeros.popleft()
                ranking[i] = 0
                ranking[new_id] = 1
                ids[i:new_id+1] = np.roll(ids[i:new_id+1], 1)
                cum_sum -= 1
            else:
                break
        elif x == 1 and q_ones:
            q_ones.popleft()
        elif x == 0 and q_zeros:
            q_zeros.popleft()
    return ranking, ids

            


