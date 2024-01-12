import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def posterior_grid(grid_points=50, heads=6, tails=9, prior_type="uniform"):
    grid = np.linspace(0, 1, grid_points)

    if prior_type == "uniform":
        prior = np.repeat(1/grid_points, grid_points)  
    elif prior_type == "binary":
        prior = (grid <= 0.5).astype(int)
    elif prior_type == "linear":
        prior = abs(grid - 0.5) 
    else:
        raise ValueError("Invalid prior_type")

    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

data = np.repeat([0, 1], (10, 3))
points = 30
h = data.sum()
t = len(data) - h

grid_uniform, posterior_uniform = posterior_grid(points, h, t, prior_type="uniform")
grid_binary, posterior_binary = posterior_grid(points, h, t, prior_type="binary")
grid_linear, posterior_linear = posterior_grid(points, h, t, prior_type="linear")

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(grid_uniform, posterior_uniform, 'o-')
plt.title(f'Uniform')
plt.yticks([])
plt.xlabel('θ')

plt.subplot(132)
plt.plot(grid_binary, posterior_binary, 'o-')
plt.title(f'Binary')
plt.yticks([])
plt.xlabel('θ')

plt.subplot(133)
plt.plot(grid_linear, posterior_linear, 'o-')
plt.title(f'Linear')
plt.yticks([])
plt.xlabel('θ')

plt.show()
