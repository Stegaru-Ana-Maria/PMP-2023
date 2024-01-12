import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def beta_binomial_posterior(x, a_prior, b_prior, y, N):
    prior = stats.beta.pdf(x, a_prior, b_prior)
    likelihood = stats.binom.pmf(y, N, x)
    posterior = prior * likelihood
    posterior /= posterior.sum()
    return posterior

def metropolis_beta_binomial(draws=10000, a_prior=1, b_prior=1):
    trace = np.zeros(draws)
    old_x = 0.5
    old_prob = beta_binomial_posterior(old_x, a_prior, b_prior, y=0, N=1)
    delta = np.random.normal(0, 0.5, draws)

    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = beta_binomial_posterior(new_x, a_prior, b_prior, y=0, N=1)
        acceptance = new_prob / old_prob

        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x

    return trace

a_prior, b_prior = 2, 5
prior_beta = stats.beta(a_prior, b_prior)

trace_metropolis = metropolis_beta_binomial(draws=10000, a_prior=a_prior, b_prior=b_prior)

plt.figure(figsize=(10, 6))

x_prior = np.linspace(0, 1, 100)
y_prior = prior_beta.pdf(x_prior)
plt.plot(x_prior, y_prior, 'C1-', lw=3, label='True distribution (Prior)')

plt.hist(trace_metropolis[trace_metropolis > 0], bins=25, density=True, alpha=0.7, label='Estimated distribution (Metropolis)')
plt.xlabel('θ')
plt.ylabel('pdf(θ)')
plt.yticks([])
plt.legend()

plt.xlim(0, 1)
plt.title('Distribuție a posteriori estimată cu Metropolis pentru modelul beta-binomial')
plt.show()
