import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


alpha = [4, 4, 5, 5]
beta = [1/3, 1/2, 1/2, 1/3]
lambda_ = [3, 2, 2, 3]
prob_server = [0.25, 0.25, 0.3, 0.2]
lambda_latenta = 4

samples = []
for _ in range(10000):
    server_index = np.random.choice(4, p=prob_server)
    timp_procesare = stats.gamma(alpha[server_index], scale=1/lambda_[server_index]).rvs()
    latenta = stats.expon(scale=1/lambda_latenta).rvs()
    timp_total = timp_procesare + latenta
    samples.append(timp_total)

# probabilitatea ca X să fie mai mare de 3 milisecunde
probabilitate = np.mean(np.array(samples) > 3)
print(f"Probabilitatea ca X > 3 milisecunde: {probabilitate}")

# grafic
plt.hist(samples, bins=50, density=True, alpha=0.6, color='g')
plt.xlabel('Timpul total de servire (X)')
plt.ylabel('Densitate')
plt.title('Densitatea distribuției lui X')
plt.show()


