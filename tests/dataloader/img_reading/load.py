import numpy as np
import time

N = 512
n = 25
a = np.random.randint(0, 255, (N, N)).astype(np.float32)

def run1():
    imgs = np.empty((N, N, n), dtype='float32')
    for i in range(n):
        imgs[:, :, i] = a
    return imgs

def run2():
    imgs = np.empty((n, N, N), dtype='float32')
    for num in range(n):
        imgs[num, :, :] = a
    return imgs.transpose((1, 2, 0))

def run3():
    imgs = np.empty((N, n, N), dtype='float32')
    for num in range(n):
        imgs[:, num, :] = a
    return imgs.transpose((0, 2, 1))

num_runs = 100

start_time = time.time()
for _ in range(num_runs):
    result1 = run1()
mean_time_run1 = (time.time() - start_time) / num_runs

start_time = time.time()
for _ in range(num_runs):
    result2 = run2()
mean_time_run2 = (time.time() - start_time) / num_runs

start_time = time.time()
for _ in range(num_runs):
    result3 =  run3()
mean_time_run3 = (time.time() - start_time) / num_runs

print(f"run1: {mean_time_run1} (s)")
print(f"run2: {mean_time_run2} (s)")
print(f"run3: {mean_time_run3} (s)")


# assert np.all(result1 == result2) and np.all(result1 == result3)
# print(result1.shape, result2.shape, result3.shape)
