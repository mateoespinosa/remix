"""
File containing util functions for computing resource allocations as done by
different runs/methods.
"""
import time
import tracemalloc


def resource_compute(function, *args, **kwargs):
    """
    Evaluates function(*args, **kwargs) and returns the time and memory
    consumption of that evaluation.

    :param fun function: An arbitrary function to run with arguments
        *args and **kwargs.
    :param List[Any] args: Positional arguments for the provided function.
    :param Dictionary[str, Any] kwargs: Key-worded arguments to the provided
        function.

    :returns Tuple[Any, float, float]: Tuple (call_results, time, memory)
        where `results` are the results of calling
        function(*args, **kwargs), time is the time it took for executing
        that function in seconds, and memory is the total memory consumption
        for that function in MB.

    """

    # Start our clock and memory handlers
    start_time = time.time()
    tracemalloc.start()

    # Run the function
    result = function(*args, **kwargs)

    # And compute memory usage
    memory, peak = tracemalloc.get_traced_memory()
    # Tracemalloc reports memory in Kibibytes, so let's change it to MB
    memory = memory * (1024 / 1000000)
    tracemalloc.stop()

    return result, (time.time() - start_time), memory
