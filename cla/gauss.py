__all__ = [
  'gauss_elimination',
  'gauss_jordan_elimination',
]

from cla.utils import eye

def gauss_elimination (arr, return_intermediates=False, show_steps=True, return_pivots=False):
  m_n = []
  pivots = []
  # One column at a time
  for col_idx in range(arr.shape[1] - 1):
    if show_steps:
      print ("A = {}".format(arr))
    # Check for zeros on the diagonal
    if arr[col_idx][col_idx] == 0:
      # Switch row with one that has no zero values on diag
      for row_idx in range(col_idx + 1, arr.shape[0]):
        if arr[col_idx][row_idx] != 0:
          # Get identity as base for pivoting
          p = eye(arr.shape[0])
          # Switch rows
          tmp = p[row_idx]
          p[row_idx] = p[col_idx]
          p[col_idx] = tmp
          if show_steps:
            print ("P = {}".format(p))
          m_n.append(p)
          pivots.append(1)
          arr = p * arr
          if show_steps:
            print ("P * A = {}".format(arr))
          break
    # Get identity as base for combination
    m = eye(arr.shape[0])
    for row_idx in range(col_idx + 1, arr.shape[0]):
      # Fill coefficients of combination
      m[row_idx][col_idx] = -arr[row_idx][col_idx] / arr[col_idx][col_idx]
    m_n.append(m)
    pivots.append(0)
    if show_steps:
      print ("M = {}".format(m))
    # Multiply
    arr = m * arr
    if show_steps:
      print ("A' = {}".format(arr))
      print ("------------------")
  if return_intermediates:
    if return_pivots:
      return arr, m_n, pivots
    return arr, m_n
  else:
    return arr

def gauss_jordan_elimination (arr, return_intermediates=False, show_steps=True):
  m_n = []
  # One column at a time (reverse)
  for col_idx in range(arr.shape[0] - 1, 0, -1):
    if show_steps:
      print ("A = {}".format(arr))
    # Get identity as base for combination
    m = eye(arr.shape[0])
    for row_idx in range(col_idx - 1, -1, -1):
      # Fill coefficients of combination
      m[row_idx][col_idx] = -arr[row_idx][col_idx] / arr[col_idx][col_idx]
    m_n.append(m)
    if show_steps:
      print ("M = {}".format(m))
    # Multiply
    arr = m * arr
    if show_steps:
      print ("A' = {}".format(arr))
      print ("------------------")
  if return_intermediates:
    return arr, m_n
  else:
    return arr
