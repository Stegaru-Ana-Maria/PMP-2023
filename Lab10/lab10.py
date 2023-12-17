import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
import numpy as np
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
import arviz as az
from oauth2client.client import GoogleCredentials
from sklearn.preprocessing import StandardScaler

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
link = 'https://drive.google.com/file/d/15vw89v1FhjcwBDUQj8chXBp154NVNxFx/view?usp=sharing'
id = link.split('/')[-2]

downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('dummy.csv')
dummy_data = pd.read_csv('dummy.csv',header=None, delimiter=' ')

# Ex1
# Punctul a
x_1 = dummy_data.iloc[:, 0].values
y_1 = dummy_data.iloc[:, 1].values
order = 5
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()

with pm.Model() as model_p:
    alpha = pm.Normal('alpha', mu=0, tau=1.0)
    beta = pm.Normal('beta', mu=0, tau=1.0/10, size=order)
    epsilon = pm.HalfCauchy('epsilon', beta=5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, tau=1.0/epsilon**2, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
alpha_p_post = idata_p.posterior['alpha'].mean(axis=(0, 1)).item()
beta_p_post = idata_p.posterior['beta'].mean(axis=(0, 1))
y_p_post = alpha_p_post + np.dot(beta_p_post, x_1s)

plt.plot(x_1s[0], y_p_post, 'C2', label=f'model order {order}, sigma=10')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Punctul b
with pm.Model() as model_p_sigma_100:
    alpha = pm.Normal('alpha', mu=0, sigma=1.0)
    beta = pm.Normal('beta', mu=0, sigma=100.0, size=order)
    epsilon = pm.HalfCauchy('epsilon', beta=5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
    idata_p_sigma_100 = pm.sample(2000, return_inferencedata=True)

with pm.Model() as model_p_sigma_array:
    alpha = pm.Normal('alpha', mu=0, sigma=1.0)
    beta = pm.Normal('beta', mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), size=order)
    epsilon = pm.HalfCauchy('epsilon', beta=5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
    idata_p_sigma_array = pm.sample(2000, return_inferencedata=True)

x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)

alpha_p_post_100 = idata_p_sigma_100.posterior['alpha'].mean(axis=(0, 1)).item()
beta_p_post_100 = idata_p_sigma_100.posterior['beta'].mean(axis=(0, 1))
y_p_post_100 = alpha_p_post_100 + np.dot(beta_p_post_100, x_1s)

alpha_p_post_array = idata_p_sigma_array.posterior['alpha'].mean(axis=(0, 1)).item()
beta_p_post_array = idata_p_sigma_array.posterior['beta'].mean(axis=(0, 1))
y_p_post_array = alpha_p_post_array + np.dot(beta_p_post_array, x_1s)

plt.plot(x_1s[0], y_p_post_100, 'C3', label=f'model order {order}, sigma=100')
plt.plot(x_1s[0], y_p_post_array, 'C4', label=f'model order {order}, sigma=array([10, 0.1, 0.1, 0.1, 0.1])')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# EX 2
num_new_points = 500
x_new = np.linspace(dummy_data.iloc[:, 0].min(), dummy_data.iloc[:, 0].max(), num_new_points)
y_new = np.interp(x_new, dummy_data.iloc[:, 0], dummy_data.iloc[:, 1]) + np.random.normal(loc=0, scale=1, size=num_new_points)

dummy_data_new = pd.concat([dummy_data, pd.DataFrame({'x': x_new, 'y': y_new})], ignore_index=True)

plt.scatter(dummy_data.iloc[:, 0], dummy_data.iloc[:, 1], c='C0', marker='.', label='Date Existente')
plt.scatter(x_new, y_new, c='r', marker='.', label='Date Noi')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

x_1 = dummy_data_new.iloc[:, 0].values
y_1 = dummy_data_new.iloc[:, 1].values
order = 5
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()

with pm.Model() as model_p_sigma_100_new:
    alpha = pm.Normal('alpha', mu=0, sigma=1.0)
    beta = pm.Normal('beta', mu=0, sigma=100.0, size=order)
    epsilon = pm.HalfCauchy('epsilon', beta=5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
    idata_p_sigma_100_new = pm.sample(2000, return_inferencedata=True)

with pm.Model() as model_p_sigma_array_new:
    alpha = pm.Normal('alpha', mu=0, sigma=1.0)
    beta = pm.Normal('beta', mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), size=order)
    epsilon = pm.HalfCauchy('epsilon', beta=5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
    idata_p_sigma_array_new = pm.sample(2000, return_inferencedata=True)

x_new_plot = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
alpha_p_post_100_new = idata_p_sigma_100_new.posterior['alpha'].mean(axis=(0, 1)).item()
beta_p_post_100_new = idata_p_sigma_100_new.posterior['beta'].mean(axis=(0, 1))
y_p_post_100_new = alpha_p_post_100_new + np.dot(beta_p_post_100_new, x_1s)

alpha_p_post_array_new = idata_p_sigma_array_new.posterior['alpha'].mean(axis=(0, 1)).item()
beta_p_post_array_new = idata_p_sigma_array_new.posterior['beta'].mean(axis=(0, 1))
y_p_post_array_new = alpha_p_post_array_new + np.dot(beta_p_post_array_new, x_1s)

plt.plot(x_1s[0], y_p_post_100_new, 'C3', label=f'model order {order}, sigma=100')
plt.plot(x_1s[0], y_p_post_array_new, 'C4', label=f'model order {order}, sigma=array([10, 0.1, 0.1, 0.1, 0.1])')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
