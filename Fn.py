def Fn(name, f, n_out=1):  # pylint: disable=invalid-name
  """Returns a layer with no weights that applies the function `f`.

  `f` can take and return any number of arguments, and takes only positional
  arguments -- no default or keyword arguments. It often uses JAX-numpy (`jnp`).
  The following, for example, would create a layer that takes two inputs and
  returns two outputs -- element-wise sums and maxima:

      `Fn('SumAndMax', lambda x0, x1: (x0 + x1, jnp.maximum(x0, x1)), n_out=2)`

  The layer's number of inputs (`n_in`) is automatically set to number of
  positional arguments in `f`, but you must explicitly set the number of
  outputs (`n_out`) whenever it's not the default value 1.

  Args:
    name: Class-like name for the resulting layer; for use in debugging.
    f: Pure function from input tensors to output tensors, where each input
        tensor is a separate positional arg, e.g., `f(x0, x1) --> x0 + x1`.
        Output tensors must be packaged as specified in the `Layer` class
        docstring.
    n_out: Number of outputs promised by the layer; default value 1.

  Returns:
    Layer executing the function `f`.
  """
  argspec = inspect.getfullargspec(f)
  if argspec.defaults is not None:
    raise ValueError('Function has default arguments (not allowed).')
  if argspec.varkw is not None:
    raise ValueError('Function has keyword arguments (not allowed).')
  if argspec.varargs is not None:
    raise ValueError('Function has variable args (not allowed).')

  def _forward(xs):  # pylint: disable=invalid-name
    if not isinstance(xs, (tuple, list)):
      xs = (xs,)
    return f(*xs)

  n_in = len(argspec.args)
  name = name or 'Fn'
  return PureLayer(_forward, n_in=n_in, n_out=n_out, name=name)
