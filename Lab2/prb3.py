import numpy as np
import matplotlib.pyplot as plt


num_experiments = 100
number_coin_flips = 10
p_stema = 0.3

results = {'ss': 0, 'sb': 0, 'bs': 0, 'bb': 0}

for _ in range(num_experiments):
    experiment_results = np.random.choice(['s', 'b'], size=(number_coin_flips, 2), p=[1-p_stema, p_stema])
    result_string = ''.join([''.join(experiment) for experiment in experiment_results])
    if result_string not in results:
        results[result_string] = 0
    results[result_string] += 1

labels = list(results.keys())
values = list(results.values())

plt.bar(labels, values)
plt.xlabel('Rezultat')
plt.ylabel('Număr de apariții')
plt.title('Distribuția rezultatelor experimentului cu două monezi')
plt.show()
