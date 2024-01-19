import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

# Ex 1
data_centrat= az.load_arviz_data("centered_eight")

print("Model Centrat:")
print(f"Numărul de lanțuri: {data_centrat.posterior.chain.size}")
print(f"Mărimea totală a eșantionului: {data_centrat.posterior.chain.size}")

az.plot_posterior(data_centrat, var_names=["mu", "tau"], point_estimate="mean")
plt.suptitle("Centered Eight")
plt.show()

data_noncentrat= az.load_arviz_data("non_centered_eight")

print("\nModel Necentrat:")
print(f"Numărul de lanțuri: {data_noncentrat.posterior.draw.size}")
print(f"Mărimea totală a eșantionului: {data_noncentrat.posterior.draw.size}")

az.plot_posterior(data_noncentrat, var_names=["mu", "tau"], point_estimate="mean")
plt.suptitle("Non-Centered Eight")
plt.show()

# Ex 2
summaries = pd.concat([az.summary(data_centrat, var_names=['mu']), az.summary(data_noncentrat, var_names=['mu'])])
summaries.index = ['centered', 'non_centered']
print(summaries)

summaries1 = pd.concat([az.summary(data_centrat, var_names=['tau']), az.summary(data_noncentrat, var_names=['tau'])])
summaries1.index = ['centered', 'non_centered']
print(summaries1)
