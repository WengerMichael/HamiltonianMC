import numpy as np
import jax.numpy as jnp
import warnings

def log_gaussian_15d(x):
    dim = 15
    if len(x) != dim :
        warnings.warn("The dimension of the input does not match the dimension of the distribution")
        return
    x = jnp.array(x)
    
    # Define mean vector and covariance matrix for the specified dimension
    mean = np.array([1, -1.5,2,3,-1,5,6,0,1,-1,0,2,-2,0,1])
    cov_matrix = jnp.array([[ 1.0,  0.5,  0.0,  0.5,  1.0,  0.5,  0.5,  0.5,  0.5,  1.0,  0.0,  0.5,  1.0,  0.5,  1.0],
[ 0.5,  2.0,  0.5,  0.0,  0.5,  1.0,  0.5,  0.5,  0.5,  0.5,  1.0,  0.5,  0.5,  1.0,  0.5],
 [ 0.0,  0.5,  1.5,  0.5,  0.5,  0.5,  0.0,  0.5,  1.0,  0.5,  0.5,  0.0,  0.5,  0.5,  1.0],
 [ 0.5,  0.0,  0.5,  2.0,  0.5,  0.5,  1.0,  0.5,  0.0,  0.5,  0.5,  1.0,  0.5,  0.5,  0.5],
 [ 1.0,  0.5,  0.5,  0.5,  1.0,  0.5,  0.5,  1.0,  0.5,  0.0,  0.5,  0.5,  0.5,  1.0,  0.5],
 [ 0.5,  1.0,  0.5,  0.5,  0.5,  2.0,  0.5,  0.5,  0.5,  0.5,  1.0,  0.5,  0.5,  0.5,  0.5],
 [ 0.5,  0.5,  0.0,  1.0,  0.5,  0.5,  1.5,  0.5,  0.5,  0.5,  0.0,  0.5,  0.5,  0.5,  1.0],
 [ 0.5,  0.5,  0.5,  0.5,  1.0,  0.5,  0.5,  1.0,  0.5,  0.5,  0.5,  0.0,  0.5,  0.5,  0.5],
 [ 0.5,  0.5,  1.0,  0.0,  0.5,  0.5,  0.5,  0.5,  2.0,  0.5,  0.5,  0.5,  0.5,  1.0,  0.5],
 [ 1.0,  0.5,  0.5,  0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  1.5,  0.5,  1.0,  0.5,  0.5,  0.5],
 [ 0.0,  1.0,  0.5,  0.5,  0.5,  1.0,  0.0,  0.5,  0.5,  0.5,  1.0,  0.5,  0.5,  0.5,  0.5],
 [ 0.5,  0.5,  0.0,  1.0,  0.5,  0.5,  0.5,  0.0,  0.5,  1.0,  0.5,  2.0,  0.5,  0.5,  0.5],
 [ 1.0,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  1.5,  0.5,  0.5],
 [ 0.5,  1.0,  0.5,  0.5,  1.0,  0.5,  0.5,  0.5,  1.0,  0.5,  0.5,  0.5,  0.5,  2.0,  0.5],
 [ 1.0,  0.5,  1.0,  0.5,  0.5,  0.5,  1.0,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  2.0]])


    # Ensure the covariance matrix is of the correct dimension
    #assert cov_matrix.shape == (dim, dim), f"Covariance matrix must be {dim}x{dim}."

    # Calculate the inverse and determinant of the covariance matrix
    cov_inv = jnp.linalg.inv(cov_matrix)
    cov_det = jnp.linalg.det(cov_matrix)

    # Calculate the normalization factor
    normalization_factor = 1 / ((2 * jnp.pi) ** (dim / 2) * jnp.sqrt(cov_det))

    # Calculate the exponent term
    diff = x - mean
    exponent_term = -0.5 * jnp.dot(jnp.dot(diff.T, cov_inv), diff)

    # Calculate the Gaussian distribution value
    gaussian_value = normalization_factor * jnp.exp(exponent_term)

    # Return the logarithm of the Gaussian distribution
    return np.log(gaussian_value)

def log_gaussian_2d(x):
    dim = 2
    if len(x) != dim :
        warnings.warn("The dimension of the input does not match the dimension of the distribution")

    x = jnp.array(x)
    mean = np.array([1,-1.5])
    cov_matrix = jnp.array([[4,0.8],[0.7,1.5]])
    
    # Ensure the covariance matrix is 2x2
    #assert cov_matrix.shape == (2, 2), "Covariance matrix must be 2x2."

    # Calculate the inverse and determinant of the covariance matrix
    cov_inv = jnp.linalg.inv(cov_matrix)
    cov_det = jnp.linalg.det(cov_matrix)

    # Calculate the normalization factor
    normalization_factor = 1 / (2 * jnp.pi * jnp.sqrt(cov_det))

    # Calculate the exponent term
    diff = x - mean
    exponent_term = -0.5 * jnp.dot(jnp.dot(diff.T, cov_inv), diff)

    # Calculate the Gaussian distribution value
    gaussian_value = normalization_factor * jnp.exp(exponent_term)

    # Return the logarithm of the Gaussian distribution
    return np.log(gaussian_value)
