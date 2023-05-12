__all__ = [
  "eye",
  "zeros",
  "ones",
  "random_array",
  "det",
  "vector_norm",
  "is_square",
  "is_diagonally_dominant",
  "is_symmetric",
  "is_definite_positive",
  "invert",
]

import math
from cla.Array import Array
from cla.constants import constants

def eye (shape):
  if not issubclass(int, shape.__class__):
    raise TypeError("Shape of identity must be an integer")
  res = []
  for i in range(shape):
    res.append([])
    for j in range(shape):
      if i == j:
        res[i].append(1)
      else:
        res[i].append(0)
  return Array(res)

def zeros (shape):
  if len(shape) != 2:
    raise ValueError("Shape must have two values!")
  res = []
  for i in range(shape[0]):
    res.append([])
    for j in range(shape[1]):
      res[i].append(0)
  return Array(res)

def ones (shape):
  if len(shape) != 2:
    raise ValueError("Shape must have two values!")
  res = []
  for i in range(shape[0]):
    res.append([])
    for j in range(shape[1]):
      res[i].append(1)
  return Array(res)

def random_array (shape):
  import random
  random.seed()
  if len(shape) != 2:
    raise ValueError("Shape must have two values!")
  res = []
  for i in range(shape[0]):
    res.append([])
    for j in range(shape[1]):
      res[i].append(random.random())
  return Array(res)

def det (arr):
  from cla.decomposition import lu_decomposition
  _, _, det = lu_decomposition(arr, return_det=True, show_warnings=False)
  return round(det, constants.decimal_places)

def det_jacobi (arr):
  from cla.eigen import jacobi_eigen
  _, _, det = jacobi_eigen(arr, return_det=True, show_warnings=False)
  return det

def vector_norm (vec, p=2):
  if ((vec.shape[0] != 1) and (vec.shape[1] != 1)):
    raise ValueError("Input for vector_norm must be a vector (first or second dimension equals 1)")
  p_sum = 0
  for i in range(vec.shape[0]):
    for j in range(vec.shape[1]):
      p_sum += vec[i][j] ** p
  return round(p_sum ** (1./p), constants.decimal_places)

def is_square (arr):
  """
  Checks whether a matrix is square
  """
  return arr.shape[0] == arr.shape[1]

def is_diagonally_dominant (arr):
  """
  Checks whether a matrix is diagonally dominant
  """
  dom = zeros((arr.shape[0], 1))
  for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
      if (i != j):
        dom[i][0] += abs(arr[i][j])
  for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
      if (i == j):
        if (abs(arr[i][j]) < dom[i][0]):
          return False
  return True

def is_symmetric (arr):
  """
  Checks whether a matrix is symmetric
  """
  if not is_square(arr):
    raise ValueError("Matrix must be a square before checking whether it's symmetric")
  for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
      if (arr[i][j] != arr[j][i]):
        return False
  return True

def is_definite_positive (arr):
  """
  Checks whether a matrix is definite positive
  """
  if not is_square(arr):
    return False
  for k in range(1, arr.shape[0] + 1):
    determinant = det(arr.submatrix((k,k)))
    if (determinant < 0):
      return False
  return True

def invert (arr, bypass=False):
  if ((arr.shape[0] != arr.shape[1]) and (not bypass)):
    raise ValueError("Inverting arrays has only been tested for square matrices. If you want to proceed anyway, set bypass=True on function call.")
  from cla.gauss import gauss_jordan_elimination, gauss_elimination
  arr, gauss_mid = gauss_elimination (arr, return_intermediates=True, show_steps=False, return_pivots=False)
  arr, jordan_mid = gauss_jordan_elimination (arr, return_intermediates=True, show_steps=False)
  m = eye(arr.shape[0])
  for i in range(arr.shape[0]):
    m[i][i] = 1. / arr[i][i]
  inv = eye(arr.shape[0])
  for mid in gauss_mid:
    inv = mid * inv
  for mid in jordan_mid:
    inv = mid * inv
  inv = m * inv
  return inv

def get_greater_value_outside_diagonal (arr):
  greater = 0
  greater_i = 0
  greater_j = 0
  for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
      if (i != j):
        val = abs(arr[i][j])
        if val > greater:
          greater = val
          greater_i = i
          greater_j = j
  return greater, greater_i, greater_j

def generate_p_matrix (arr, i, j):
  if (arr[i][i] != arr[j][j]):
    phi = 1. / 2. * math.atan((2 * arr[i][j]) / (arr[i][i] - arr[j][j]))
  else:
    phi = math.pi / 4
  p = eye (arr.shape[0])
  p[i][i] = math.cos(phi)
  p[i][j] = -math.sin(phi)
  p[j][i] = math.sin(phi)
  p[j][j] = math.cos(phi)
  return p