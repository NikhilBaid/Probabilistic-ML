import numpy as np
from matplotlib import pyplot as plt

def ml_parameters(Y):
    """
    Compute the ML estimates for the Bernoulli parameters.
    parameters:
        Y
    returns:
        params_ml: Array of ML estimates for each pixel
    """
    N, D = Y.shape
    S_d = np.sum(Y, axis=0) 
    params_ml = S_d / N
    return params_ml

def map_parameters(Y, alpha=3, beta=3):
    """
    Compute the MAP estimates for the Bernoulli parameters with Beta priors.
    parameters:
        Y: N x D binary matrix.
        alpha: beta distribution param
        beta: beta distribution param
    returns:
        params_map: Array of MAP estimates
    """
    N, D = Y.shape
    S_d = np.sum(Y, axis=0)  #sum of each column 
    params_map = (S_d + alpha - 1) / (N + alpha + beta - 2)
    return params_map

def display_image(params, title):
    """
    Display a D-dimensional parameter vector as an 8x8 image with a colour scale
    parameters:
        params: Parameter vector
        title: image title
    """
    plt.figure(figsize=(4,4))
    img=np.reshape(params, (8,8))
    plt.imshow(img, interpolation="none", cmap='winter')
    plt.title(title)
    plt.colorbar()
    plt.show()
    
def plot_difference(params_ml, params_map):
    """
    Plot the difference between ML and MAP parameters
    parameters:
        params_ml: ML estimates
        params_map: MAP estimates
    """
    difference = params_ml - params_map
    plt.figure(figsize=(4,4))
    plt.imshow(np.reshape(difference, (8,8)), interpolation="none", cmap='bwr')
    plt.title('Difference')
   
    plt.colorbar()
    plt.show()
    
def main():
  
    Y = load_data('/Users/baidn/Downloads/binarydigits.txt')
    N, D = Y.shape
   
    #compute ml and map
    params_ml = ml_parameters(Y)
    alpha, beta_val = 3, 3
    params_map = map_parameters(Y, alpha, beta_val)

    # plot the images and difference
    display_image(params_ml, title='ML Parameters')
    display_image(params_map, title='MAP Parameters (alpha=3, beta=3)')
    
    plot_difference(params_ml, params_map)

if __name__ == "__main__":
    main()