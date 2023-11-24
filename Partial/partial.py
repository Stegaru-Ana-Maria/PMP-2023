import random
import numpy as np
from scipy import stats
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd

# Subiectul 1
# Numărul de încercari in experimentul Bernoulli a jucatorului P0
trials = 1
# Numărul de încercari in experimentul Bernoulli a jucatorului P1
trials1 = 2
# Probabilitatea de succes intr-o singur incercare (moneda masluita- jucator P0)
theta_real = 0.33
# Probabilitatea de succes intr-o singur incercare (moneda normala- jucator P1)
theta_real1 = 0.5
# Generez date simulate folosind o distributie Bernoulli cu probabilitatea theta_real
data = stats.bernoulli.rvs(p=theta_real, size=20000)
data1 = stats.bernoulli.rvs(p=theta_real1, size=20000)


# rezultate posibile
bb = []
bs = []
sb = []
ss = []
for i in range(20000):
  # moneda masluita
    stema_moneda1 = stats.binom.rvs(1,0.33, size=10) 
    # moneda normala
    stema_moneda2 = stats.binom.rvs(1,0.5, size=10)
    bb_c = 0
    bs_c = 0
    sb_c = 0
    ss_c = 0
    for j in range(10):
        if stema_moneda1[j] == 0 and stema_moneda2[j] == 0:
            bb_c += 1
        elif stema_moneda1[j] == 0 and stema_moneda2[j] == 1:
            bs_c += 1
        elif stema_moneda1[j] == 1 and stema_moneda2[j] == 0:
            sb_c += 1
        elif stema_moneda1[j] == 1 and stema_moneda2[j] == 1:
            ss_c += 1
    bb.append(bb_c)
    bs.append(bs_c)
    sb.append(sb_c)
    ss.append(ss_c)


# Modelul
with pm.Model() as model_a:
    # Definim distributia a priori pentru θ folosind distribuția Beta
    θ = pm.Beta('θ', alpha=1., beta=1.)  
    # Definim distribuția likelihood pentru y, care este o distributie Bernoulli
    # Parametrul p al distributiei Bernoulli este legat de variabila θ
    y = pm.Bernoulli('y', p=θ, observed=data) 
    # Realizam 1000 de iteratii de esantionare (sampling) folosind metoda MCMC
    idata = pm.sample(1000, random_seed=123, return_inferencedata=True)


# Subiectul 2
# 1

# Exemplu de generare a valorilor unei variabile aleatoare X - N (mu, sigma):
mu = 0.
sigma = 1.
X = stats.norm(mu, sigma)
x = X.rvs(200) # generează 200 valori aleatoare din distribuția normală
print(x)

# 2
# with pm.Model() as model:
#   # Definesc o distributie a priori Beta pentru parametrul θ
#   θ = pm.Beta('θ', alpha=1., beta=1.)
#    # Definesc distribuția likelihood Bernoulli pentru observațiile 'data', cu parametrul p legat de θ
#   y = pm.Bernoulli('y', p=θ, observed=x)
#   # Realizez 1000 de iteratii de esantionare (sampling) folosind metoda MCMC si salvez datele in formatul Inferentiala
#   idata = pm.sample(1000, random_seed=123, return_inferencedata=True)


with pm.Model() as model_g:
    # Definim o variabila uniforma pentru medie (μ) între 40 și 70
    μ = pm.Uniform('μ', lower=40, upper=70)   
    # Definim o variabilă HalfNormal pentru deviația standard (σ) cu sigma 10
    σ = pm.HalfNormal('σ', sigma=10)    
    # Definim o variabilă Normală pentru observații (y) cu parametrii μ și σ, legată de datele observate
    y = pm.Normal('y', mu=μ, sigma=σ, observed=x)   
    # Efectuăm 1000 de iterații de eșantionare și salvăm datele în formatul Inferențială
    idata_g = pm.sample(1000, return_inferencedata=True)


# 3
# Esantionam datele predictive posterioare folosind datele inferentiale si modelul
ppc_t = pm.sample_posterior_predictive(idata_g, model=model_g, samples=1000, random_seed=123, var_names=['σ'])
# Folosim biblioteca ArviZ pentru a crea un grafic al datelor predictive posterioare
az.plot_ppc(idata_g, figsize=(12, 6), num_pp_samples=100)
# Limitez axa x la intervalul [40, 70]
plt.xlim(40, 70)

plt.show()
