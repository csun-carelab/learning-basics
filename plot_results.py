import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


train_size = [20, 40, 60, 80, 100]

data_path = "results/less_noise/"

results = []
for ts in train_size:
    data_file = data_path + "fse_" + str(ts) + ".csv"
    df = pd.read_csv(data_file, header=None)
    df['size'] = ts
    results.append(df.to_numpy())
results = np.vstack(results)
df = pd.DataFrame(results)

plt.figure()
methods = ['baseline', 'beacon', 'beacon+play']
for m in range(len(methods)):
    sns.lineplot(data=df, x=len(df.columns)-1, y=m, label=methods[m], linewidth=3)
plt.legend(loc='upper right')
plt.xticks(train_size)
plt.xlabel("Train demos")
plt.ylabel("FSE Loss")
# plt.savefig('plots/fse_lessnoise.png', dpi=1080)
plt.show()

print("Done.")
