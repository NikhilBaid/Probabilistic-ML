import numpy as np
from scipy.special import logsumexp
from matplotlib import pyplot as plt

# load data
X = np.loadtxt('/Users/baidn/Downloads/binarydigits.txt')
N, D = X.shape  # N = number of data points, D = dimensionality
K_val = [2, 3, 4, 7, 10]  # different Ks

def EM(K, X, max_iter):
    """
    Performs the EM algorithm

    parameters:
    K: Number of mixture components
    X: Data matrix of shape (N, D)
    max_iter: Maximum number of iterations

    returns:
    pi: Learned mixing proportions of shape (K,)
    P: Learned Bernoulli parameters of shape (K, D)
    log_likelihoods: List of log-likelihood values at each iteration
    """
    thrs = 1e-6  # threshold for convergence 

    # unifrom mixing
    pi = np.full(K, 1.0 / K)

    # Bernoulli parameters random between 0.1 and 0.9 - avoid log 1
    P = np.random.rand(K, D) * 0.8 + 0.1

    def E_step(X, P, pi):
        """
        E-step - responsibility

        returns:
        R: Responsibility
        """
        #log probabilities to prevent underflow
        log_P = np.log(P)  
        log_1_minus_P = np.log(1 - P) 

        # log-likelihood for each component and data point
        log_likelihood = np.zeros((N, K))  
        for k in range(K):
            # log-likelihood for component k
            log_prob = X @ log_P[k] + (1 - X) @ log_1_minus_P[k]  #
            log_likelihood[:, k] = log_prob + np.log(pi[k])

        #responsibilities in log space to prevent underflow
        log_sum = logsumexp(log_likelihood, axis=1, keepdims=True)  #
        log_R = log_likelihood - log_sum  
        R = np.exp(log_R)  #convert back

        return R

    def M_step(X, R):
        """
        M - Update params pi and P.

        returns:
        pi_new: Updated mixing proportions
        P_new: Updated Bernoulli params
        """
        epsilon = 1e-7  #prevetn div by 0

        # Update mixing proportions as in c)
        N_k = np.sum(R, axis=0)  
        pi_new = N_k / N  

        # Update Bernoulli parameters as in c)
        P_new = np.zeros((K, D))  
        for k in range(K):
            # weighted sum of data points for each feature
            numerator = R[:, k] @ X  
            denominator = N_k[k] + epsilon  # no div by 0
            P_new[k] = numerator / denominator  

        # Ensure P_new values are within (0, 1)
        P_new = np.clip(P_new, epsilon, 1 - epsilon)

        return pi_new, P_new

    def compute_log_likelihood(X, pi, P):
        """
        computes the log-likelihood

        params:
        uses current parameters for pi and P

        returns:
        total_log_likelihood: Scalar value of the total log-likelihood
        """
        # log probabilities
        log_P = np.log(P) 
        log_1_minus_P = np.log(1 - P) 

        #log-likelihood for components and data points
        log_likelihood = np.zeros((N, K))  
        for k in range(K):
            log_prob = X @ log_P[k] + (1 - X) @ log_1_minus_P[k]  
            log_likelihood[:, k] = log_prob + np.log(pi[k])

        #total log-likelihood using logsumexp
        log_sum = logsumexp(log_likelihood, axis=1)  
        total_log_likelihood = np.sum(log_sum)

        return total_log_likelihood

    log_likelihoods = []

    for iteration in range(max_iter):
        #E
        R = E_step(X, P, pi)

        #M
        pi, P = M_step(X, R)

        #log-likelihood
        ll = compute_log_likelihood(X, pi, P)
        log_likelihoods.append(ll)

        # convergence check
        if iteration > 0:
            ll_change = np.abs(log_likelihoods[-1] - log_likelihoods[-2])
            if ll_change < thrs:
                print(f"Converged at iteration {iteration}.")
                break

        print(f"Iteration {iteration + 1}: Log-Likelihood = {ll:.4f}")

    return pi, P, log_likelihoods

# Dictionaries to store results for different K values
log_likelihood_histories = {}
learned_parameters = {}

#loop over each K value 
for K in K_val:
    print(f"Run {K}")

    
    pi_learned, P_learned, ll_history = EM(K, X, max_iter=100)

    # store history
    log_likelihood_histories[K] = ll_history
    learned_parameters[K] = {
        'pi': pi_learned,
        'P': P_learned
    }

    #display mixing
    print(f"pi for K={K}")
    print(pi_learned)

# plot log-likelihoods for each K
plt.figure(figsize=(10, 6))
for K in K_val:
    ll_history = log_likelihood_histories[K]
    iterations = range(1, len(ll_history) + 1)
    plt.plot(iterations, ll_history, label=f'K = {K}')

plt.title('Log-likelihood vs iteration for Different K')
plt.xlabel('Iteration')
plt.ylabel('Log-likelihood')
plt.legend()
plt.show()

# plot the learned Bernoulli parameters P for each K
for K in K_val:
    P_learned = learned_parameters[K]['P']
    num_components = P_learned.shape[0]


    #subplot
    cols = min(num_components, 5)
    rows = (num_components + cols - 1) // cols
    #looping through each if tge mixings to plot the paramter distribution
    plt.figure(figsize=(cols * 2, rows * 2))
    for k in range(num_components):
        plt.subplot(rows, cols, k + 1)
        plt.imshow(P_learned[k].reshape(8, 8), interpolation='none', cmap='bwr')
        plt.title(f"K={K}, p = {round(learned_parameters[K]['pi'][k],2)}")
        
    plt.tight_layout()
    plt.show()