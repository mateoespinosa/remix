import dill


def serialized_function_execute(serialized):
    """
    Helper function to execute a serialized serialized (using dill)

    :param str serialized: The string containing a serialized tuple
        (function, args) which was generated using dill.
    :returns X: the result of executing function(*args)
    """
    function, args = dill.loads(serialized)
    return function(*args)
