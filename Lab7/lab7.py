import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import arviz as az

# Punctul a
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
link = 'https://drive.google.com/file/d/1QFTznskcFmqNUXtx5fuMlisH3rSKG_Zj/view'
id = link.split('/')[-2]

downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('auto-mpg.csv')
data = pd.read_csv('auto-mpg.csv')

to_drop = ['cylinders', 'displacement', 'weight', 'acceleration', 'model year', 'origin', 'car name']
df = data.drop(to_drop, axis=1)
df.replace('?', np.nan, inplace=True)
df= df.dropna()
df = df.astype(float)
df= df[df['mpg'] > 0]

X = df['horsepower'].values
y = df['mpg'].values

plt.figure(figsize=(10, 6))
plt.scatter(df['horsepower'], df['mpg'], color='blue', alpha=0.5)
plt.title('Relația dintre Cai Putere și Consumul de Combustibil (mpg)')
plt.xlabel('Cai Putere (CP)')
plt.ylabel('Mile pe Galon (mpg)')
plt.grid(True)
plt.show()

# Punctul b
with pymc.Model() as model:
    alpha = pymc.Normal('alpha', mu=0, tau=0.001)
    beta = pymc.Normal('beta', mu=0, tau=0.001)
    epsilon = pymc.HalfCauchy('epsilon', 5)
    x = pymc.Normal('x', mu=np.mean(X), tau=1.0/np.var(X), observed=X)
    mu = alpha + beta * x
    y = pymc.Normal('y', mu=mu, sigma=epsilon,observed=df['mpg'])

    trace = pymc.sample(10000, tune=5000, return_inferencedata=True)

# Punctul c
plt.plot(df['horsepower'], df['mpg'], 'C0.')
posterior_g = trace.posterior.stack(samples={"chain", "draw"})
alpha_m = posterior_g['alpha'].mean().item()
beta_m = posterior_g['beta'].mean().item()
draws = range(0, posterior_g.samples.size, 10)
plt.plot(df['horsepower'], posterior_g['alpha'][draws].values + posterior_g['beta'][draws].values * df['horsepower'][:, None], c='gray', alpha=0.5)
plt.plot(df['horsepower'], alpha_m + beta_m * df['horsepower'], c='k', label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend()

# Punctul d
az.plot_posterior(trace, hdi_prob=0.95, var_names=['alpha', 'beta'])
plt.scatter(df['horsepower'], df['mpg'], color='blue', alpha=0.5, label='Observations')
plt.title('Relația dintre Cai Putere și Consumul de Combustibil (mpg) cu Regiunea 95%HDI')
plt.xlabel('Cai Putere (CP)')
plt.ylabel('Mile pe Galon (mpg)')
plt.legend()
plt.show()
