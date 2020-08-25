import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os
import seaborn as sns

# max exponent (horizontal axis)
max_e = 130

# log2(m)
m_log = 23

data_list = [
        'fp16_notc',
        'fp16_tc_nocor',
        'fp32_notc',
        'fp32_tc_nocor',
        'fp32_tc_cor',
        'mixed_tc_cor_emu',
        'tf32_tc_nocor_emu',
        'tf32_tc_cor_emu',
        ]

def get_color(d):
    if 'fp16' in d:
        return '#51318f'
    if 'fp32' in d:
        return '#006c3a'
    if 'tf32' in d:
        return '#ed6c00'
    if 'mixed' in d:
        return '#333132'
    return '#ffffff'

def get_linestyle(d):
    if 'notc' in d:
        return '-'
    if 'nocor' in d:
        return '-.'
    if 'cor' in d:
        return ':'
    return '--'

k_offset = 5
max_k = m_log - k_offset

df = pd.read_csv("data.csv", encoding="UTF-8")

fig, ax = plt.subplots(max_k + 1, 1, figsize=(8, 1.7 * (max_k + 1)))
for k in range(0, max_k + 1):
    for d in data_list:
        data = df.query("k=='" + str(k) + "'&mode=='" + d + "'")
        ax[max_k - k].set_xlim(-130, max_e)
        ax[max_k - k].set_ylim(-5, 75)
        ax[max_k - k].set_ylabel("prob[%]")
        ax[max_k - k].set_yticks([0, 25, 50, 75])
        ax[max_k - k].grid()
        ax[max_k - k].plot(data['e'] - 127, data['prob'], label=d, linestyle=get_linestyle(d), color=get_color(d))
        ax[max_k - k].legend(ncol=3)
        ax[max_k - k].text(-125, 60, "bQR k=" + str(k))
        ax[max_k - k].vlines(-15, -10, 75, colors='r')
        ax[max_k - k].vlines(15, -10, 75, colors='r')
        ax[max_k - k].text(-18, 38, "FP16 border", color='r', ha='right')

plt.savefig("exponent_distribution.png", bbox_inches="tight", transparent=True)
