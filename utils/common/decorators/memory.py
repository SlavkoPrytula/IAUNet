import time
import functools
import psutil
import os


def memory_evaluator(cls):
    original_forward = cls.forward
    
    @functools.wraps(cls.forward)
    def memory_forward(self, *args, **kwargs):
        start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        result = original_forward(self, *args, **kwargs)
        
        end_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        memory_used = end_memory - start_memory

        class_name = self.__class__.__name__
        print(f"{class_name}: memory used {memory_used:.2f}(MB)")

        return result

    cls.forward = memory_forward
    return cls
