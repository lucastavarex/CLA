__all__ = [
  "gauss_seidel"
]

from copy import deepcopy
from cla.constants import constants
from cla.utils import random_array, zeros, vector_norm, is_diagonally_dominant, is_definite_positive

def gauss_seidel (A, B, threshold=constants.epsilon):
  if (not is_diagonally_dominant(A)):
    if (not is_definite_positive(A)):
      raise ValueError("The matrix \"Matriz_A\" is neither strictly diagonal dominant nor positive definite. Therefore the Gauss-Seidel method won't converge!")
  prev_x = random_array(B.shape)
  r = 1000
  n = B.shape[0]
  while (r > threshold):
    x = zeros(B.shape)
    for i in range(x.shape[0]):
      x[i][0] = B[i][0]
      for j in range(0, i):
        x[i][0] -= A[i][j]*x[j][0]
      for j in range(i+1, n):
        x[i][0] -= A[i][j]*prev_x[j][0]
      x[i][0] /= A[i][i]
    r = vector_norm(x - prev_x, 2) / vector_norm(x, 2)
    prev_x = deepcopy(x)
  return x