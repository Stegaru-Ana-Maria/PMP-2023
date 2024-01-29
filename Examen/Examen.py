import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

# Ex 1
# Punctul a
# Am incarcat fisierul intr-un DataFrame
data = pd.read_csv("BostonHousing.csv")

rm = data["rm"].values
crim = data["crim"].values
indus = data["indus"].values
medv = data["medv"].values

# afisez cateva randuri ale DataFrame-ului 
print(data.head())

# Punctul b

with pm.Model() as model:
    # definesc parametrii pentru variabilele independente
    beta_rm = pm.Normal('beta_rm', mu=0, sigma=1)  # coeficient pentru numarul mediu de camere 
    beta_crim = pm.Normal('beta_crim', mu=0, sigma=1)  # coeficient pentru rata criminalitatii 
    beta_indus = pm.Normal('beta_indus', mu=0, sigma=1)  # coeficient pentru proportia suprafetei comerciale non-retail 
    alpha = pm.Normal('alpha', mu=0, sigma=1)  

    # ecuatia de regresie
    medv_estimated = alpha + beta_rm * data['rm'] + beta_crim * data['crim'] + beta_indus * data['indus']

    # distributia pentru variabila dependenta observata "medv"
    medv_observed = pm.Normal('medv_observed', mu=medv_estimated, sigma=1, observed=data['medv'])


with model:
    trace = pm.sample(2000, tune=1000, target_accept=0.9)

# Punctul c

# Afisez rezultatele
pm.summary(trace, hdi_prob=0.95)


# Ex 2
# Punctul a
def posterior_grid(grid_points=50, heads=6, tails=9):
    # definirea grilei pentru probabilitatea θ
    grid = np.linspace(0, 1, grid_points)
    # distributia uniforma ca prior
    prior = np.repeat(1/grid_points, grid_points)
    
    # verosimilitatea pentru o distributie Bernoulli (prima apariție a unei steme)
    # dacă θ este 1 => avem o stema, altfel probabilitatea este 0
    likelihood = np.where(grid < 1, 0, 1)
    
    # calculez distributia a posteriori
    posterior = likelihood * prior
    posterior /= posterior.sum()
    
    return grid, posterior

# Simulez datele pentru prima aparitie a unei steme
data = np.repeat([0, 1], (10, 3))
points = 10
h = data.sum()

t = len(data) - h

# Calculez distributias a posteriori
grid, posterior = posterior_grid(points, h, t)

# Trasez distributia a posteriori
plt.plot(grid, posterior, 'o-')
plt.title(f'Prima stema la θ = 1, cozi = {t}')
plt.yticks([])
plt.xlabel('θ')
plt.show()

# Punctul b

# gasesc valoarea θ care maximizeaza probabilitatea a posteriori
max_posterior_idx = np.argmax(posterior)
theta_max_posterior = grid[max_posterior_idx]

# afisez θ 
print(f'Theta care maximizează probabilitatea a posteriori este: {theta_max_posterior}')
