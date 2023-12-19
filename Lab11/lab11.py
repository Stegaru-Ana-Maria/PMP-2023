import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Ex1
np.random.seed(42)

clusters = 3
n_cluster = [200, 150, 150]
n_total = sum(n_cluster)
means = [5, 0, -5]
std_devs = [2, 2, 2]

mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))

plt.figure(figsize=(10, 6))
plt.hist(mix, bins=30, density=True, alpha=0.5, color='blue')
plt.title('Mixtură de trei distribuții gaussiene')
plt.xlabel('Valoare')
plt.ylabel('Densitate')
plt.show()

# Ex2
num_components = [2, 3, 4]

for components in num_components:
    with pm.Model() as model:
        mus = [pm.Normal(f'mu{i}', mu=0, tau=1/(102)) for i in range(components)]
        taus = [pm.Gamma(f'tau_{i}', alpha=1, beta=1/(102)) for i in range(components)]
        weights = pm.Dirichlet(f'weights', a=np.ones(components))

        y = pm.Mixture(f'y', w=weights, comp_dists=[pm.Normal.dist(mu=m, tau=t) for m, t in zip(mus, taus)], observed=mix)

        trace = pm.sample(1000, tune=1000)

    pm.plot_trace(trace)
    plt.suptitle(f'Mixtura Gaussiana cu {components} componente')
    plt.show()
