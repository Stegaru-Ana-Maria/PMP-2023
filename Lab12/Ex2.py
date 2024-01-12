import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum()*4/N
    error = abs((pi - np.pi) / pi) * 100
    return error

N_values = [100, 1000, 10000]

num_runs = 100

errors = np.zeros((len(N_values), num_runs))

for i, N in enumerate(N_values):
    for j in range(num_runs):
        errors[i, j] = estimate_pi(N)

mean_errors = np.mean(errors, axis=1)
std_errors = np.std(errors, axis=1)

plt.figure(figsize=(10, 6))
plt.errorbar(N_values, mean_errors, yerr=std_errors, fmt='o-', capsize=5)
plt.axhline(0, color='r', linestyle='--', label='Eroarea reală (0%)')
plt.xlabel('Numărul de puncte (N)')
plt.ylabel('Eroare medie (%)')
plt.legend()
plt.show()
