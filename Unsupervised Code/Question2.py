import numpy as np
from scipy.special import logsumexp
from scipy.special import betaln as beta_func
Y = np.loadtxt('/Users/baidn/Downloads/binarydigits.txt')
N, D = Y.shape

# Model a
m1 = np.log(0.5) * (N * D)  

# Model b
S = np.sum(Y)  # Total 1s
Y2 = N * D - S  # Total 0s
m2 = beta_func(S + 1, Y2 + 1)  # Log probability with Beta prior

# Model c
S_d = np.sum(Y, axis=0)  # Count of 1s for each component
m3 = np.sum(beta_func(S_d + 1, N - S_d + 1))  # Sum log probabilities for each p_d

# Normalise
log_probs = [m1, m2, m3]
total_log_prob = logsumexp(log_probs) 

print("relative posterior log relative and absolute probabilities :")
print(f"Model a: {m1 - total_log_prob} and {np.exp(m1 - total_log_prob)}")
print(f"Model b: {m2 - total_log_prob} and {np.exp(m2 - total_log_prob)}")
print(f"Model c: {m3 - total_log_prob} and {np.exp(m3 - total_log_prob)}")   