import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os

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

df = pd.read_csv("data.csv", encoding="UTF-8")

plt.figure(figsize=(12, 6))
plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams["font.size"] = 32
plt.rcParams["xtick.labelsize"] = 23
plt.rcParams["ytick.labelsize"] = 23
plt.rcParams["legend.fontsize"] = 20
plt.grid()
plt.xlabel("$m$ : $m \\times 16$")
plt.xscale("log", basex=2)
plt.xlim([2**9, 2**25])
plt.ylabel("Orthogonality")
plt.yscale("log", basey=10)


for d in data_list:
    data = df.query("compute_mode=='" + d + "'")
    plt.plot(data['m'], data['orthogonality_diag'], label=d + '-Diag', marker='*', color=get_color(d), linestyle=get_linestyle(d))
    plt.plot(data['m'], data['orthogonality_nondiag'], label=d + '-NonDiag', marker='v', color=get_color(d), linestyle=get_linestyle(d))

plt.legend(loc='best',
    bbox_to_anchor=(0, 1.05, 1., -.1),
    borderaxespad=0.,
    ncol=2,
    mode="expand")


plt.savefig("q_eval.pdf", bbox_inches="tight", transparent=True)
