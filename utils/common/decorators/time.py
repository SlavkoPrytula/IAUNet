import time
import functools

def timeit(func):
    """
    Decorator to measure the execution time of a method.
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)  # Call the function being decorated
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__!r} completed in {elapsed_time:.4f} seconds")
        return value
    return wrapper_timer


def timeit_evaluator(cls):
    """
    Class decorator to add timing around 'forward' and 'evaluate' methods for Evaluator object.
    """
    original_forward = cls.forward
    original_evaluate = cls.evaluate
    
    @functools.wraps(cls.forward)
    def timed_forward(self, *args, **kwargs):
        if not hasattr(self, 'start_time'):
            self.start_time = time.time()
        return original_forward(self, *args, **kwargs)

    @functools.wraps(cls.evaluate)
    def timed_evaluate(self, *args, **kwargs):
        result = original_evaluate(self, *args, **kwargs)
        if hasattr(self, 'start_time'):
            elapsed_time = time.time() - self.start_time
            class_name = self.__class__.__name__
            print(f"{class_name}: evaluation complete in {elapsed_time:.2f}(s)")
            delattr(self, 'start_time')
        return result

    cls.forward = timed_forward
    cls.evaluate = timed_evaluate
    return cls
