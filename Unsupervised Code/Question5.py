import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
# reading the file
with open('/Users/baidn/Downloads/symbols.txt', 'r') as f:
    symbols = [line.rstrip('\n') for line in f]
    symbols = [' ' if symbol == '' else symbol for symbol in symbols]
#print("Symbols:", symbols) 

# symbol to index and index to symbol mappings
symbol_to_i = {s: i for i, s in enumerate(symbols)}
i_to_symbol = {i: s for i, s in enumerate(symbols)}
K = len(symbols)

# read war and peace for calculating proposals
with open('/Users/baidn/Downloads/WarAndPeace.txt', 'r', encoding='utf-8') as f:
    text = f.read()
#ignoring the intro text - only consdering the book
text = text[834:].lower()
#count symbols and pairs in loop and calc probability
symbol_c = np.zeros(K, dtype=np.int64)
pair_c = np.zeros((K, K), dtype=np.int64)
prev_idx = None
for s in text:
    if s in symbol_to_i:
        idx = symbol_to_i[s]
        symbol_c[idx] += 1
        if prev_idx is not None:
            pair_c[prev_idx, idx] += 1
        prev_idx = idx
    else:
        continue  # skip symbols not in our set
symbol_c[symbol_c == 0] = 1 #make probabilities slightly more than 0 for symb that dont occur such as not to leave these out - leads to better results 
N = symbol_c.sum()
phi_arr = symbol_c / N  

symbol_counts_no_zeros = np.where(symbol_c == 0, 1, symbol_c)
psi_arr = pair_c / symbol_counts_no_zeros[:, None]
psi_arr = np.nan_to_num(psi_arr)
print("Psi array shape:", psi_arr.shape)
print("Psi array:", psi_arr)
print(phi_arr)
#heatmap for the transition matrix and a graph forphi 
char = ['space' if s == ' ' else s for s in symbols]

plt.figure(figsize=(12, 10))
plt.imshow(psi_arr, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.xticks(ticks=np.arange(K), labels=char, rotation=90, fontsize=8)
plt.yticks(ticks=np.arange(K), labels=char, fontsize=8)
plt.title('Transition Matrix')
plt.xlabel('Next symbol')
plt.ylabel('Current symbol')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(char, phi_arr)
plt.xlabel('Symb')
plt.ylabel('Phi Prob')

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

epsilon = 1e-10  #to prevent log(0)
#log probabilities
log_phi = np.log(phi_arr + epsilon)
log_psi = np.log(psi_arr + epsilon)

# read encrypted text
with open('/Users/baidn/Downloads/message.txt', 'r') as f:
    encrypt_txt = f.read().strip()

# encrypted symbols to indices
encrypt_i = [symbol_to_i[s] for s in encrypt_txt if s in symbol_to_i]
sigma = -np.ones(K, dtype=int)  # initializing sigma
# keep track of assigned characters
cipher_i = set()
plaintext_i = set()

com_char = [' ', 'e', 'a', 't', 'i', 'n']

# get indices of common characters
com_char_i = [symbol_to_i[c] for c in com_char if c in symbol_to_i]

# Sort encrypted symbols by frequency in descending order
sort_encrypt_i = sorted(range(K), key=lambda x: -symbol_c[x])

# Select the top frequent encrypted symbols and map to common characters
top_encrypt = sort_encrypt_i[:len(com_char_i)]
for enc_idx, plain_idx in zip(top_encrypt, com_char_i):
    sigma[enc_idx] = plain_idx
    cipher_i.add(enc_idx)
    plaintext_i.add(plain_idx)

# remaining cipher and plaintext indices and random mapping of these
r_cipher_i = set(range(K)) - cipher_i
r_plaintext_i = set(range(K)) - plaintext_i
r_cipher_i = list(r_cipher_i)
r_plaintext_i = list(r_plaintext_i)
random.shuffle(r_plaintext_i)

for enc_idx, avail_idx in zip(r_cipher_i, r_plaintext_i):
    sigma[enc_idx] = avail_idx
    cipher_i.add(enc_idx)
    plaintext_i.add(avail_idx)

#inverse
sigma_inv = np.zeros(K, dtype=int)
sigma_inv[sigma.astype(int)] = np.arange(K)

#log likelihodd
def compute_log_likelihood(decrypt_i):
    log_ml = log_phi[decrypt_i[0]]
    for i in range(1, len(decrypt_i)):
        prev_idx = decrypt_i[i - 1]
        curr_idx = decrypt_i[i]
        log_ml += log_psi[prev_idx, curr_idx]
    return log_ml


decrypt_i = [sigma_inv[idx] for idx in encrypt_i]
log_p_e_g_sigma = compute_log_likelihood(decrypt_i)

# MH sampler and tracking acceptance rate
iter = 10000  
acceptance = 0
for iteration in range(1, iter + 1):
    #new sigma by swapping two symbols
    i, j = random.sample(range(K), 2)
    sigma_prop = sigma.copy()
    sigma_prop[i], sigma_prop[j] = sigma_prop[j], sigma_prop[i]
    
    #update sigma_inverse accordingly
    sigma_i_prop = np.zeros(K, dtype=int)
    sigma_i_prop[sigma_prop.astype(int)] = np.arange(K)
    
    #decrypt the message with proposed sigma
    decrypted_i_prop = [sigma_i_prop[idx] for idx in encrypt_i]
    
    #new log-likelihood, use logs to prevent some small values being treated as zeros
    log_p_e_g_sigma_prop = compute_log_likelihood(decrypted_i_prop)
    
    #acceptance probability
    del_log_ml = log_p_e_g_sigma_prop - log_p_e_g_sigma
    acceptance_prob = min(1, np.exp(del_log_ml))
    
    # accpet/reject step
    if random.random() < acceptance_prob:
        sigma = sigma_prop
        sigma_inv = sigma_i_prop
        decrypt_i = decrypted_i_prop
        log_p_e_g_sigma = log_p_e_g_sigma_prop
        acceptance += 1
    
    
    # 100 iterations print
    if iteration % 100 == 0:
        decrypted_text_sample = ''.join([i_to_symbol[idx] for idx in decrypt_i[:60]])
        print(f"Iteration {iteration}: {decrypted_text_sample}")
print(f"overall Acceptance Rate: {acceptance / iter}")