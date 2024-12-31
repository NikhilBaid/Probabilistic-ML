import numpy as np
import matplotlib.pyplot as plt


X = np.loadtxt('/Users/baidn/Downloads/ssm_spins.txt').T  
k = 4  # dimension of latent
d, T = X.shape  #dimension data and number of time steps

#defining A
t1 = 2 * np.pi / 180
t2 = 2 * np.pi / 90
A = 0.99 * np.array([
    [np.cos(t1), -np.sin(t1), 0, 0],
    [np.sin(t1), np.cos(t1), 0, 0],
    [0, 0, np.cos(t2), -np.sin(t2)],
    [0, 0, np.sin(t2), np.cos(t2)]
])
#Q
Q = np.eye(k) - A @ A.T
#C
C = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [0.5, 0.5, 0.5, 0.5]])
R = np.eye(d)

#initial states taken as mean of normal - 0 and Identity - was not sure if I should instead sample from the normal dist ect. but ended up converging to similar results
y_init = np.zeros(k)
Q_init = np.eye(k)

def run_ssm_kalman(X, y_init, Q_init, A, Q, C, R, mode='smooth'):
    """
    Calculates kalman-smoother estimates of SSM state posterior.
    :param X:       data, [d, t_max] numpy array
    :param y_init:  initial latent state, [k,] numpy array
    :param Q_init:  initial variance, [k, k] numpy array
    :param A:       latent dynamics matrix, [k, k] numpy array
    :param Q:       innovariations covariance matrix, [k, k] numpy array
    :param C:       output loading matrix, [d, k] numpy array
    :param R:       output noise matrix, [d, d] numpy array
    :param mode:    'forw' or 'filt' for forward filtering, 'smooth' for also backward filtering
    :return:
    y_hat:      posterior mean estimates, [k, t_max] numpy array
    V_hat:      posterior variances on y_t, [t_max, k, k] numpy array
    V_joint:    posterior covariances between y_{t+1}, y_t, [t_max, k, k] numpy array
    likelihood: conditional log-likelihoods log(p(x_t|x_{1:t-1})), [t_max,] numpy array
    """
    d, k = C.shape
    t_max = X.shape[1]

    # dimension checks
    assert np.all(X.shape == (d, t_max)), "Shape of X must be (%d, %d), %s provided" % (d, t_max, X.shape)
    assert np.all(y_init.shape == (k,)), "Shape of y_init must be (%d,), %s provided" % (k, y_init.shape)
    assert np.all(Q_init.shape == (k, k)), "Shape of Q_init must be (%d, %d), %s provided" % (k, k, Q_init.shape)
    assert np.all(A.shape == (k, k)), "Shape of A must be (%d, %d), %s provided" % (k, k, A.shape)
    assert np.all(Q.shape == (k, k)), "Shape of Q must be (%d, %d), %s provided" % (k, k, Q.shape)
    assert np.all(C.shape == (d, k)), "Shape of C must be (%d, %d), %s provided" % (d, k, C.shape)
    assert np.all(R.shape == (d, d)), "Shape of R must be (%d, %d), %s provided" % (d, k, R.shape)

    y_filt = np.zeros((k, t_max))  # filtering estimate: \hat(y)_t^t
    V_filt = np.zeros((t_max, k, k))  # filtering variance: \hat(V)_t^t
    y_hat = np.zeros((k, t_max))  # smoothing estimate: \hat(y)_t^T
    V_hat = np.zeros((t_max, k, k))  # smoothing variance: \hat(V)_t^T
    K = np.zeros((t_max, k, X.shape[0]))  # Kalman gain
    J = np.zeros((t_max, k, k))  # smoothing gain
    likelihood = np.zeros(t_max)  # conditional log-likelihood: p(x_t|x_{1:t-1})

    I_k = np.eye(k)

    # forward pass

    V_pred = Q_init
    y_pred = y_init

    for t in range(t_max):
        x_pred_err = X[:, t] - C.dot(y_pred)
        V_x_pred = C.dot(V_pred.dot(C.T)) + R
        V_x_pred_inv = np.linalg.inv(V_x_pred)
        likelihood[t] = -0.5 * (np.linalg.slogdet(2 * np.pi * (V_x_pred))[1] +
                                x_pred_err.T.dot(V_x_pred_inv).dot(x_pred_err))

        K[t] = V_pred.dot(C.T).dot(V_x_pred_inv)

        y_filt[:, t] = y_pred + K[t].dot(x_pred_err)
        V_filt[t] = V_pred - K[t].dot(C).dot(V_pred)

        # symmetrise the variance to avoid numerical drift
        V_filt[t] = (V_filt[t] + V_filt[t].T) / 2.0

        y_pred = A.dot(y_filt[:, t])
        V_pred = A.dot(V_filt[t]).dot(A.T) + Q

    # backward pass

    if mode == 'filt' or mode == 'forw':
        # skip if filtering/forward pass only
        y_hat = y_filt
        V_hat = V_filt
        V_joint = None
    else:
        V_joint = np.zeros_like(V_filt)
        y_hat[:, -1] = y_filt[:, -1]
        V_hat[-1] = V_filt[-1]

        for t in range(t_max - 2, -1, -1):
            J[t] = V_filt[t].dot(A.T).dot(np.linalg.inv(A.dot(V_filt[t]).dot(A.T) + Q))
            y_hat[:, t] = y_filt[:, t] + J[t].dot((y_hat[:, t + 1] - A.dot(y_filt[:, t])))
            V_hat[t] = V_filt[t] + J[t].dot(V_hat[t + 1] - A.dot(V_filt[t]).dot(A.T) - Q).dot(J[t].T)

        V_joint[-2] = (I_k - K[-1].dot(C)).dot(A).dot(V_filt[-2])

        for t in range(t_max - 3, -1, -1):
            V_joint[t] = V_filt[t + 1].dot(J[t].T) + J[t + 1].dot(V_joint[t + 1] - A.dot(V_filt[t + 1])).dot(J[t].T)

    return y_hat, V_hat, V_joint, likelihood


#running kalman for filt
y_filt, V_filt, _, L_filt = run_ssm_kalman(X, y_init, Q_init, A, Q, C, R, mode='filt')

#plotting filtered means
plt.figure(figsize=(10, 6))
for i in range(k):
    plt.plot(y_filt[i, :], label=f'Latent {i+1}')
plt.title('filtered latent')
plt.xlabel('time')
plt.ylabel('value')
plt.legend()
plt.show()
#running kalman for smooth
y_smooth, V_smooth, _, L_smooth = run_ssm_kalman(X, y_init, Q_init, A, Q, C, R, mode='smooth')

# plotting smooth mean
plt.figure(figsize=(10, 6))
for i in range(k):
    plt.plot(y_smooth[i, :], label=f'Latent {i+1}')
plt.title('smoothed latent')
plt.xlabel('time')
plt.ylabel('value')
plt.legend()
plt.show()
# plot the log-determinants
plt.figure(figsize=(10, 6))
plt.plot(np.linalg.slogdet(V_filt)[1])
plt.title('Log-Determinant filt')
plt.xlabel('time')
plt.ylabel('ld')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(np.linalg.slogdet(V_smooth)[1])
plt.title('Log-Determinant smooth')
plt.xlabel('Time')
plt.ylabel('ld')
plt.show()

def em_lgssm(X, y_init, Q_init, A_init, Q_init_param, C_init, R_init, num_iter=50):
    """
    Run EM algorithm to estimate parameters of LGSSM using E and M updates pre calculated
    params
    X: data [d,t]
    y_init: initial latent state
    Q_init: initial variance
    A_init:initial latent dynamics matrix [k, k] 
    Q_init_param: initial innovariations covariance matrix [k, k] 
    C_init: initial output loading matrix [d, k] 
    R_init:output noise matrix [d, d] 
    num_iter:number of iterations
    output
    A: updated A param
    Q: updated q param
    C: updated C param
    R: updated R param
    llh: log likelihoods stored each iteration
    """
    # Initialize parameters
    A = A_init
    Q = Q_init_param
    C = C_init
    R = R_init
    
    # Store log-likelihoods
    llh = []
    
    for iteration in range(num_iter):
        # E - running kalman smoother and log likelihoods
        y_hat, V_hat, V_joint, likelihoods = run_ssm_kalman(X, y_init, Q_init, A, Q, C, R, mode='smooth')
        
        total_llh = np.sum(likelihoods)
        llh.append(total_llh)
        
        #M
        T = X.shape[1]
        k = y_hat.shape[0]
        d = X.shape[0]
        
        xx = np.zeros((d, d))
        xy = np.zeros((d, k))
        yy = np.zeros((k, k))
        y_y_prev = np.zeros((k, k))
        yy_prev = np.zeros((k, k))
        yy_t = np.zeros((k, k))
        
        for t in range(T):
            x_t = X[:, t][:, np.newaxis]#d
            y_t = y_hat[:, t][:, np.newaxis] #k 
            V_t = V_hat[t] #kx k
            
            xx += x_t @ x_t.T
            xy += x_t @ y_t.T
            
            yy += y_t @ y_t.T + V_t
            
        for t in range(1, T):
            y_t = y_hat[:, t][:, np.newaxis] # k 
            y_prev = y_hat[:, t-1][:, np.newaxis] # k 
            V_t = V_hat[t] # k xk
            V_prev = V_hat[t-1] # k x k
            V_joint_t = V_joint[t-1] # kxk
            
            y_y_prev += y_t @ y_prev.T + V_joint_t
            yy_prev += y_prev @ y_prev.T + V_prev
            yy_t += y_t @ y_t.T + V_t
            
        # Update all parameters as in M step
        C = xy @ np.linalg.inv(yy)
        A = y_y_prev @ np.linalg.inv(yy_prev)
        R = (1/T) * (xx - xy @ C.T)
        Q = (1/(T-1)) * (yy_t - y_y_prev @ A.T)
    return A, Q, C, R, llh
#use generating parameters
A_gen, Q_gen, C_gen, R_gen, llh_gen = em_lgssm(X, y_init, Q_init, A, Q, C, R, num_iter=50)

#random initialisations - i randomised the A and C matrices and let Q and R be identities - was unsure if I shuld randomise everything?
llh_random_runs = []

for i in range(10):
    A_rand = np.random.randn(k, k)
    Q_rand = np.eye(k)
    C_rand = np.random.randn(d, k)
    R_rand = np.eye(d)
    
    A_rand_n, Q_rand_n, C_rand_n, R_rand_n, llh_rand = em_lgssm(X, y_init, Q_init, A_rand, Q_rand, C_rand, R_rand, num_iter=50)
    
    llh_random_runs.append(llh_rand)

plt.figure(figsize=(10, 6))

# plotting log likelihoods
plt.plot(llh_gen, label='Generating')
for idx, llh_rand in enumerate(llh_random_runs):
    plt.plot(llh_rand, label=f'Rand {idx+1}')

plt.title('Log-likelihood vs iteration')
plt.xlabel('iteration')
plt.ylabel('Log-likelihood')
plt.legend()
plt.show()