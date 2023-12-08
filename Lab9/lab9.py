import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
link = 'https://drive.google.com/file/d/1nj2PykVs_VU92S9_tLi_xBaj61u2j46M/view?usp=sharing'
id = link.split('/')[-2]

downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('Admission.csv')
data = pd.read_csv('Admission.csv')

 # Ex 1
scaler = StandardScaler()
data[['GRE', 'GPA']] = scaler.fit_transform(data[['GRE', 'GPA']])

with pm.Model() as logistic_model:
    beta0 = pm.Normal("beta0", mu=0, tau=0.01)
    beta1 = pm.Normal("beta1", mu=0, tau=0.01)
    beta2 = pm.Normal("beta2", mu=0, tau=0.01)
    p = pm.invlogit(beta0 + beta1 * data["GRE"] + beta2 * data["GPA"])
    admission = pm.Bernoulli("admission", p, observed=data["Admission"])
    trace = pm.sample(2000, tune=1000, chains=2)

pm.summary(trace).round(2)
plt.show()
print(pm.summary(trace))
az.plot_pair(trace, var_names=['beta0', 'beta1', 'beta2'])
plt.show()
