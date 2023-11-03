import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import math
import arviz as az

def generate_poisson(lambda_param):
    return np.random.poisson(lambda_param)

def generate_normal(mean_t, stddev_t):
    return np.random.normal(mean_t, stddev_t)

def generate_exponential(scale):
    return np.random.exponential(scale)

def generate_wait_time_samples(lambda_param, mean_t, alpha, num_samples):
    wait_time_samples = []

    for _ in range(num_samples):
        total_wait_time = 0
        num_clients = generate_poisson(lambda_param)

        for _ in range(num_clients):
            order_time = generate_normal(mean_t, stddev_t)
            cook_time = generate_exponential(alpha)
            total_wait_time += order_time + cook_time

        average_wait_time = total_wait_time / num_clients if num_clients > 0 else 0
        wait_time_samples.append(average_wait_time)

    return wait_time_samples

lambda_param = 20
mean_t = 2.0
alpha = 3.0
stddev_t = 0.5
num_samples = 100

wait_time_samples = generate_wait_time_samples(lambda_param, mean_t, alpha, num_samples)

print(wait_time_samples)

mean_wait_time = np.mean(wait_time_samples)
print(f"Media timpilor de așteptare medii: {mean_wait_time:.2f} minute")


#EX2

model = pm.Model()
with model:
    alpha = pm.Exponential("alpha", lam=1)

    obs = pm.Exponential("obs", lam=alpha, value=wait_time_samples, observed=True)

    step = pm.Metropolis()
    trace = pm.sample(2000, step, cores=2)


pm.summary(trace)

plt.figure(figsize=(8, 6))
plt.hist(trace["alpha"], bins=30, density=True, alpha=0.5, color='b')
plt.title("Distribuția estimată pentru α")
plt.xlabel("Valoarea lui α")
plt.ylabel("Densitate")
plt.show()
