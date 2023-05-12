__all__ = [
  'constants'
]

# Decorator for making immutable attributes on classes
def const(cls):
  is_special = lambda name: (name.startswith("__") and name.endswith("__"))
  class_contents = {n: getattr(cls, n) for n in vars(cls) if not is_special(n)}
  def unbind(value):
    return lambda self: value
  propertified_contents = {
    name: property(unbind(value)) for (name, value) in class_contents.items()
  }
  receptor = type(cls.__name__, (object,), propertified_contents)
  return receptor()

# Declare new constants here
@const
class constants (object):
  epsilon = 1e-7
  decimal_places = 7 
  max_iter = 500
 