import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

lambda1 = 4
lambda2 = 6

p_mecanic1 = 0.4
p_mecanic2 = 0.6

valori = 10000

#Generare 10000 valori pt X
samples = []
for _ in range(valori) :
    if np.random.rand() < p_mecanic1 :
        sample =stats.expon(scale=1/lambda1).rvs()
    else :
        sample = stats.expon(scale=1/lambda2).rvs()
    samples.append(sample)

# Media si deviatia standard
mean_x = np.mean(samples)
std_dev_x = np.std(samples)

print(f"Media lui X: {mean_x}")
print(f"Deviația standard a lui X: {std_dev_x}")

# Grafic
plt.hist(samples,bins=50,density=True,alpha=0,color='b')
x = np.linspace(0,max(samples),100)
pdf_mecanic1 = stats.expon(scale=1/lambda1).pdf(x)
pdf_mecanic2 = stats.expon(scale=1/lambda2).pdf(x)
plt.plot(x, pdf_mecanic1 * p_mecanic1 + pdf_mecanic2 * (1 - p_mecanic1), 'r-', lw=2)
plt.xlabel('Timp de servire X')
plt.ylabel('Densitate')
plt.title('Densitatea distribuției lui X')
plt.show()


