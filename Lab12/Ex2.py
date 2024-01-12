import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum() * 4 / N
    error = abs((pi - np.pi) / pi) * 100
    return error

N_values = [100, 1000, 10000]
num_simulations = 100  

mean_errors = []
std_errors = []

for N in N_values:
    errors = [estimate_pi(N) for _ in range(num_simulations)]
    mean_errors.append(np.mean(errors))
    std_errors.append(np.std(errors))

plt.errorbar(N_values, mean_errors, yerr=std_errors, fmt='o-', capsize=5)
plt.xscale('log') 
plt.xlabel('Numarul de puncte (N)')
plt.ylabel('Media erorii (%)')
plt.title('Estimarea lui π cu media și deviația erorii pentru diferite valori ale lui N')
plt.grid(True)
plt.show()

