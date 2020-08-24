import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os
import seaborn as sns

df = pd.read_csv("data.csv", encoding="UTF-8")
k_offset = 5
max_k = 25 - k_offset

fig, ax = plt.subplots(max_k + 1, 1, figsize=(14, 1.7 * (max_k + 1)))

for k in range(0, max_k + 1):
    data = df.query("k=='" + str(k) + "'")
    ax[max_k - k].set_ylim(-5, 100)
    ax[max_k - k].set_ylabel("prob[%]")
    ax[max_k - k].set_yticks([0, 25, 50, 75, 100])
    ax[max_k - k].grid()
    ax[max_k - k].plot(data['e'] - 127, data['prob'], label='bQR No.' + str(k))
    ax[max_k - k].legend()

plt.savefig("exponent_distribution.pdf", bbox_inches="tight", transparent=True)
