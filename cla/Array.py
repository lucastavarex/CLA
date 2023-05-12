__all__ = [
  'Array'
]

from copy import deepcopy
from cla.constants import constants

class Array (object):

  def __init__ (self, iterable, *args, **kwargs):
    self.__iterable = iterable
    self.__shape = []
    self.__shape.append(len(iterable))
    shape2 = None
    for row in iterable:
      try:
        if not shape2:
          shape2 = len(row)
          self.__shape.append(shape2)
        elif shape2 != len(row):
          raise ValueError("Shape mismatch")
      except TypeError:
        self.__shape.append(1)
        self.__shape.reverse()
        break
  
  @property
  def shape (self):
    return self.__shape

  @property
  def iterable (self):
    return self.__iterable

  @property
  def t (self):
    return self.transpose()

  # Rounding elements according to epsilon
  def __epsilonRound (self):
    it = deepcopy(self.__iterable)
    if (self.shape[0] > 1):
      for i in range(self.shape[0]):
        for j in range(self.shape[1]):
          it[i][j] = round(it[i][j], constants.decimal_places)
    else:
      for j in range(self.shape[1]):
        it[j] = round(it[j], constants.decimal_places)
    return it

  # Pretty printer
  def __repr__ (self):
    ret = "cla.Array(\n"
    it = self.__epsilonRound()
    if self.shape[0] == 1:
      ret += "{},\n".format(it)
    else:
      for i in range(self.shape[0]):
        ret += "{},\n".format(it[i])
    return ret + ")"

  def __str__ (self):
    return self.__repr__()

  # Overrides multiplication
  def __rmul__ (self, other):
    return self.__mul__(other)

  # Overrides multiplication
  def __mul__ (self, other):
    if issubclass(self.__class__, other.__class__):
      return self.__matmul(self, other)
    else:
      it = deepcopy(self.__iterable)
      for i in range(self.shape[0]):
        for j in range(self.shape[1]):
          it[i][j] *= other
          if (abs(it[i][j]) <= constants.epsilon):
            it[i][j] = 0.0
      return Array(it)

  # Item getter
  def __getitem__ (self, key):
    if issubclass(tuple, key.__class__):
      return self.__iterable[key[0]][key[1]]
    return self.__iterable[key]

  # Item setter
  def __setitem__ (self, key, val):
    self.__iterable[key] = val

  # Get submatrix
  def submatrix (self, shape):
    try:
      if len(shape) != 2:
        raise ValueError("Shape of submatrix must have two values")
      for i in range(2):
        if (shape[i] < 1) or (shape[i] > self.shape[i]):
          raise ValueError("Shape must be 1 <= k <= n")
      it = []
      for i in range(shape[0]):
        it.append([])
        for j in range(shape[1]):
          it[i].append(self.__iterable[i][j])
      return Array(it)
    except TypeError:
      raise TypeError("Input for submatrix must be an iterable with two values")

  # Matrix multiplication implementation
  def __matmul (self, mat1, mat2):
    from cla.utils import zeros
    try:
      assert mat1.shape[1] == mat2.shape[0]
    except AssertionError:
      raise AssertionError("Matrices can't be multiplied with shapes {} and {}".format(mat1.shape, mat2.shape))
    result = zeros((mat1.shape[0], mat2.shape[1]))
    for i in range(mat1.shape[0]):
      for j in range(mat2.shape[1]):
        for k in range(mat2.shape[0]):
          result[i][j] += mat1[i][k] * mat2[k][j]
        if (abs(result[i][j]) <= constants.epsilon):
          result[i][j] = 0.0
    return result

  # Matrix comparison
  def __eq__ (self, other):
    if issubclass(self.__class__, other.__class__):
      try:
        for i in range(self.shape[0]):
          for j in range(self.shape[1]):
            if (abs(self[i][j] - other[i][j]) > constants.epsilon):
              return False
        return True
      except IndexError:
        return False
    else:
      raise TypeError("Always compare cla.Array objects instead of anything else")
  
  def __ne__ (self, other):
    return not self.__eq__(other)

  # Matrix adding
  def __add__ (self, other):
    if issubclass(self.__class__, other.__class__):
      try:
        assert self.shape == other.shape
      except AssertionError:
        raise AssertionError("Shape mismatch ({} and {})".format(self.shape, other.shape))
      it = deepcopy(self.__iterable)
      for i in range(self.shape[0]):
        for j in range(self.shape[1]):
          it[i][j] += other[i][j]
          if (abs(it[i][j]) <= constants.epsilon):
            it[i][j] = 0.0
      return Array(it)
    else:
      raise TypeError("Always compare cla.Array objects instead of anything else")
    return self
  
  def __radd__ (self, other):
    return self.__add__(other)
  
  # Matrix subtraction
  def __sub__ (self, other):
    if issubclass(self.__class__, other.__class__):
      try:
        assert self.shape == other.shape
      except AssertionError:
        raise AssertionError("Shape mismatch ({} and {})".format(self.shape, other.shape))
      it = deepcopy(self.__iterable)
      for i in range(self.shape[0]):
        for j in range(self.shape[1]):
          it[i][j] -= other[i][j]
          if (abs(it[i][j]) <= constants.epsilon):
            it[i][j] = 0.0
      return Array(it)
    else:
      raise TypeError("Always compare cla.Array objects instead of anything else")
    return self

  def __rsub__ (self, other):
    return self.__sub__(other)

  # Matrix division
  def __truediv__ (self, other):
    if issubclass(self.__class__, other.__class__):
      return self * ~other
    else:
      it = deepcopy(self.__iterable)
      for i in range(self.shape[0]):
        for j in range(self.shape[1]):
          it[i][j] /= other
      return Array(it)

  def __rtruediv__ (self, other):
    return self.__truediv__(other)

  # Getting negative of matrix
  def __neg__ (self):
    it = deepcopy(self.__iterable)
    for i in range(self.shape[0]):
      for j in range(self.shape[1]):
        it[i][j] = -it[i][j]
    return Array(it)

  # Getting absolute values of matrix
  def __abs__ (self):
    it = deepcopy(self.__iterable)
    for i in range(self.shape[0]):
      for j in range(self.shape[1]):
        it[i][j] = abs(it[i][j])
    return Array(it)

  # Inverting matrix
  def __invert__ (self):
    from cla.utils import invert
    arr = Array(deepcopy(self.__iterable))
    return invert(arr)

  # Get transpose
  def transpose (self):
    from cla.utils import zeros
    new_arr = zeros((self.shape[1], self.shape[0]))
    for i in range(self.shape[1]):
      for j in range(self.shape[0]):
        new_arr[i][j] = self.__iterable[j][i]
    return new_arr

  # Get trace
  def trace (self):
    if self.shape[0] != self.shape[1]:
      raise ValueError ("Can only get trace of square matrices")
    tr = 0
    for i in range(self.shape[0]):
      for j in range(self.shape[1]):
        if i == j:
          tr += self.__iterable[i][j]
    return tr
