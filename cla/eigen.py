__all__ = [
  'power_method',
  'jacobi_eigen'
]

from cla.utils import random_array, is_symmetric, eye, generate_p_matrix, get_greater_value_outside_diagonal
from cla.constants import constants

def power_method (arr, threshold=constants.epsilon):
  # Start with random array
  x = random_array((arr.shape[0], 1))
  # First element equals 1
  x[0][0] = 1
  # Iterations
  r = 1000
  lambda_i = constants.epsilon
  while (r > threshold):
    y = arr * x
    new_lambda_i = y[0][0]
    x = y / new_lambda_i
    r = abs(new_lambda_i - lambda_i) / abs(new_lambda_i)
    lambda_i = new_lambda_i
  return lambda_i, x

def jacobi_eigen (arr, threshold=constants.epsilon, return_det=False, show_warnings=True):
  if not is_symmetric(arr):
    raise ValueError ("Input matrices for Jacobi method must be symmetric.")
  A = arr
  X = eye(arr.shape[0])
  greater, i, j = get_greater_value_outside_diagonal(A)
  while (greater > threshold):
    P = generate_p_matrix(A, i, j)
    A = P.t * A * P
    X = X * P
    greater, i, j = get_greater_value_outside_diagonal(A)
  det = 1
  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      if (i == j):
        det *= A[i][j]
  if (det == 0):
    print ("WARNING: This is a singular matrix, so this decomposition may not be unique.")
  if return_det:
    return A, X, det
  return A, X