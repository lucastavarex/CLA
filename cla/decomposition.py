__all__ = [
  'lu_decomposition',
  'cholesky_decomposition'
]

from cla.utils import eye, zeros
from cla.gauss import gauss_elimination

def mi_to_li (arr):
  for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
      if i != j:
        arr[i][j] *= -1
  return arr

def lu_decomposition (arr, return_det=False, show_warnings=True):
  # First of all, gauss elimination (arr will be U)
  arr, intermediates, pivots = gauss_elimination(arr, return_intermediates=True, show_steps=False, return_pivots=True)
  # Then, Li=Mi with elements other than diagonal multiplied by -1
  l_n = []
  for i, m in enumerate(intermediates):
    if not pivots[i]:
      l_n.append(mi_to_li(intermediates[i]))
    else:
      l_n.append(intermediates[i])
  # L = Ln * ... * L2 * L1
  L = eye(arr.shape[0])
  for li in l_n:
    L = L * li
  # U=arr, L=L
  det = 1
  for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
      if (i == j):
        det *= arr[i][j]
  if (det == 0):
    print ("WARNING: This is a singular matrix, so this decomposition may not be unique.")
  if return_det:
    return L, arr, det
  return L, arr

def cholesky_decomposition (arr):
  from cla.utils import is_definite_positive, is_symmetric
  if (not is_definite_positive(arr) or not is_symmetric(arr)):
    raise ValueError("Unable to perform Cholesky decomposition. The given matrix is not positive definite symmetric.")
  L = zeros(arr.shape)
  for i in range(arr.shape[0]):
    L[i][i] = arr[i][i]
    for k in range(0, i):
      L[i][i] -= (L[i][k] ** 2)
    L[i][i] = (L[i][i] ** (1/2))
    for j in range(i + 1, arr.shape[1]):
      L[j][i] = arr[i][j]
      for k in range(0, i):
        L[j][i] -= (L[i][k]*L[j][k])
      L[j][i] /= L[i][i]
  return L, L.t