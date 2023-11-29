import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import arviz as az
from sklearn.preprocessing import StandardScaler

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
link = 'https://drive.google.com/file/d/1_5IImO6_DohQ2hbxzg2lTK-je6oYIw2b/view'
id = link.split('/')[-2]

downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('Prices.csv')
data = pd.read_csv('Prices.csv')

# Ex 1

with pm.Model() as model:
    # Distribuții a priori pentru parametri
    alpha = pm.Normal('alpha', mu=0, tau=0.01)
    beta1 = pm.Normal('beta1', mu=0, tau=0.01)
    beta2 = pm.Normal('beta2', mu=0, tau=0.01)
    sigma = pm.Uniform('sigma', lower=0, upper=1000)
    mu = alpha + beta1 * data['Speed'] + beta2 * np.log(data['HardDrive'])
    price = pm.Normal('price', mu=mu, tau=1/sigma**2, observed=data['Price'])

with model:
    trace = pm.sample(2000, tune=1000)

print(pm.summary(trace).round(2))
pm.plot_posterior(trace, var_names=['alpha', 'beta1', 'beta2', 'sigma'])

# Ex 2

az.plot_posterior(trace, var_names=['beta1', 'beta2'], hdi_prob=0.95)
plt.show()

# Ex 3

# Afisez sumarul statistic al distributiei a posteriori
print(pm.summary(trace).round(2))

# Atât frecvența procesorului (Speed), cât și mărimea hard diskului (HardDrive), par să fie predictorii utili ai prețului de vânzare, 
# deoarece intervalele HDI pentru coeficienții lor nu includ zero, indicând o asociere semnificativă.

new_data = pd.DataFrame({'Speed': [33], 'HardDrive': [540]})

scaler = StandardScaler().fit(data[['Speed', 'HardDrive']])
new_data_scaled = scaler.transform(new_data)

with model:
    post_pred = pm.sample_posterior_predictive(trace, samples=5000, random_seed=42)

mu_new = trace['alpha'] + trace['beta1'] * new_data_scaled[:, 0] + trace['beta2'] * np.log(new_data_scaled[:, 1])

hdi_90 = pm.stats.hpd(mu_new, hdi_prob=0.9)

plt.figure(figsize=(8, 6))
plt.hist(post_pred['price'].mean(axis=1), bins=50, color='skyblue', alpha=0.7, label='Simulated Prices')
plt.axvline(hdi_90[0], color='red', linestyle='--', label='HDI 5%')
plt.axvline(hdi_90[1], color='red', linestyle='--', label='HDI 95%')
plt.title('Simulated Prices and 90% HDI')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.legend()
plt.show()










