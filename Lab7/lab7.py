import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm

df = pd.read_csv('auto-mpg.csv')


df = df.dropna()

plt.figure(figsize=(10, 6))
plt.scatter(df['horsepower'], df['mpg'], color='blue', alpha=0.5)
plt.title('Relația dintre Cai Putere și Consumul de Combustibil (mpg)')
plt.xlabel('Cai Putere (CP)')
plt.ylabel('Mile pe Galon (mpg)')
plt.grid(True)
plt.show()

X = df['horsepower'].values
y = df['mpg'].values


with pm.Model() as linear_model:
    alpha = pm.Normal('alpha', mu=0, tau=1.0 / 10**2)
    beta = pm.Normal('beta', mu=0, tau=1.0 / 10**2)
    sigma = pm.Uniform('sigma', lower=0, upper=10)

    mu = alpha + beta * df['CP'].values
    likelihood = pm.Normal('mpg', mu=mu, tau=1.0 / sigma**2, observed=df['mpg'].values)


with linear_model:
    trace = pm.sample(2000, tune=1000)



alpha_mean = np.mean(trace['alpha'])
beta_mean = np.mean(trace['beta'])

regression_lines = alpha_samples + beta_samples[:, None] * df['CP'].values
