#Der Plot zeigt das Konfidenzintervall und das spuckt LMMELSM ja auch direkt aus. D.h., in jeder Zeile ist der 2.5-97.5 %-Wert geplottet. Zusätzlich habe ich den p-value darüber geschrieben - den könnte man aber weglassen, um Platz zu sparen.

#Ich würde vorschlagen, wir machen 2 Plots. Einen für Location mit der Beschrfitung earlier <-> later und dann alle Linien für sowohl snare als auch woodblock parameter (wenn man die p-Werte weglässt passen ja vllt. sogar die between und within Parameter rauf). Und einen für Scale mit der Achsenbeschriftung consistent performance <-> irregular performance. Dann hätte jeder Plot, glaube ich 11 Linien. Wenn das zu unübersichtlich wird, dann machen wir nur between? Und dann lass uns hier nur die Parameter der univariaten Modelle auf diese Weise plotten - wir hatten mit Gabriel ja besprochen, dass dies unser Schwerpunkt sein sollte.

#todo
# test with other files
# wenn nötig noch den zugehörigen p-wert aus verwandten file (error wenn nicht da)

import numpy as np
import sys
import os.path
import re

model_name = sys.argv[1] #e.g. snare_musicality
result_folder = sys.argv[2]
file_path = result_folder + 'models/singleVariable/'+model_name+'.txt'

# Open and read the file
with open(file_path, 'r') as file:
    intercept_lines = []
    variable_lines = []

    for line in file:
        if re.search(r'\s+(\w+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)',  line): #thanks, chatGPT (match lines in the format "label number number," where "label" is one or more word characters, and "number" can be a positive or negative decimal number with one or more digits before and after the decimal point)
            if "deviation" in line:
                intercept_lines.append(line.strip())
            else:
                variable_lines.append(line.strip())

#e.g. deviation 0.0791 0.0787 0.021 0.0383 0.121 where order is  param  predictor   Mean Median    SD   Q2.5   Q97.5
loc_intercept = intercept_lines[0]
scale_intercept = intercept_lines[1]
loc_RE = intercept_lines[2]
scale_RE = intercept_lines[3]

loc_variable = variable_lines[0]
scale_variable = variable_lines[1]
loc_group = variable_lines[2]
scale_group = variable_lines[3]

1/0
###########################################
# plot the bootstrap confidence intervals #
###########################################
params_CI = np.vstack([
    scipy.stats.scoreatpercentile(large_bootstrap_params[:,t==0], 2.5,
        axis=0),
    scipy.stats.scoreatpercentile(large_bootstrap_params[:,t==0], 97.5,
        axis=0)]).T

fig = plt.figure(figsize=(3.42, 4))
gs = mpl.gridspec.GridSpec(nrows=2, ncols=2,
        width_ratios=[1,3])

title_ax1 = fig.add_subplot(gs[0,:], frame_on=False)
title_ax1.tick_params(**blind_ax)
title_ax1.set_title(r'\textbf{bias effect} ($c$) at $t=0$', size=10)

text_ax1 = fig.add_subplot(gs[0,0], frame_on=False)
text_ax1.tick_params(**blind_ax)

text_ax1.text(0.95, 5./6., r'\nth{1} \textalpha', ha='right',
        va='center', transform=text_ax1.transAxes, size=10)
text_ax1.text(0.95, 4./6., r'\nth{2} \textalpha', ha='right',
        va='center', transform=text_ax1.transAxes, size=10)
text_ax1.text(0.95, 3./6., r'\nth{1} \straighttheta', ha='right',
        va='center', transform=text_ax1.transAxes, size=10)
text_ax1.text(0.95, 2./6., r'\nth{2} \straighttheta', ha='right',
        va='center', transform=text_ax1.transAxes, size=10)
text_ax1.text(0.95, 1./6., r'trial index', ha='right',
        va='center', transform=text_ax1.transAxes, size=10)

title_ax2 = fig.add_subplot(gs[1,:], frame_on=False)
title_ax2.tick_params(**blind_ax)
title_ax2.set_title(r"""\textbf{sensitivity effect} ($d'$) at $t=0$""", size=10)

text_ax2 = fig.add_subplot(gs[1,0], frame_on=False)
text_ax2.tick_params(**blind_ax)

text_ax2.text(0.95, 5./6., r'\nth{1} \textalpha', ha='right',
        va='center', transform=text_ax2.transAxes, size=10)
text_ax2.text(0.95, 4./6., r'\nth{2} \textalpha', ha='right',
        va='center', transform=text_ax2.transAxes, size=10)
text_ax2.text(0.95, 3./6., r'\nth{1} \straighttheta', ha='right',
        va='center', transform=text_ax2.transAxes, size=10)
text_ax2.text(0.95, 2./6., r'\nth{2} \straighttheta', ha='right',
        va='center', transform=text_ax2.transAxes, size=10)
text_ax2.text(0.95, 1./6., r'trial index', ha='right',
        va='center', transform=text_ax2.transAxes, size=10)

ci_ax1 = fig.add_subplot(gs[0,1], frame_on=False)
ci_ax1.tick_params(**dict(left=False, labelleft=False, right=False,
    labelright=False))
ci_ax1.plot([0,1], [0,0], 'k-', transform=ci_ax1.transAxes)
ci_ax1.axvline(0, ls=':', lw=0.75)

trans1 = mpl.transforms.blended_transform_factory(
    ci_ax1.transData, ci_ax1.transAxes)

for i in xrange(5):
    if params_p[i]<0.05:
        color_now=color1
    else:
        color_now='k'
    ci_ax1.plot(-params_CI[i], 2*[(5-i)/6.], ls='-', c=color_now,
            transform=trans1)
    ci_ax1.text(-params_CI[i].mean(), (5-i)/6. + 0.02, r'$p=%.3f$' % params_p[i],
            transform=trans1, ha='center', va='bottom', size=7, color=color_now)

ci_ax2 = fig.add_subplot(gs[1,1], frame_on=False, sharex=ci_ax1)
ci_ax2.tick_params(**dict(left=False, labelleft=False, right=False,
    labelright=False))
ci_ax2.plot([0,1], [0,0], 'k-', transform=ci_ax2.transAxes)
ci_ax2.axvline(0, ls=':', lw=0.75)

trans2 = mpl.transforms.blended_transform_factory(
    ci_ax2.transData, ci_ax2.transAxes)

for i in xrange(5):
    if params_p[i+5]<0.05:
        color_now=color1
    else:
        color_now='k'
    ci_ax2.plot(params_CI[i+5], 2*[(5-i)/6.], ls='-', c=color_now,
            transform=trans2)
    ci_ax2.text(params_CI[i+5].mean(), (5-i)/6. + 0.02,
            r'$p=%.3f$' % params_p[i+5],
            transform=trans2, ha='center', va='bottom', size=7,
            color=color_now)

ci_ax1.set_xlim([-np.abs(params_CI[:10]).max(), np.abs(params_CI[:10]).max()])

ci_ax1.set_xlabel('model coefficients')
ci_ax2.set_xlabel('model coefficients')

ci_ax1.text(0.025, 0.025, r'\textit{more liberal}', ha='left', va='bottom',
        transform=ci_ax1.transAxes, size=7)
ci_ax1.text(0.975, 0.025, r'\textit{more conservative}', ha='right',
        va='bottom', transform=ci_ax1.transAxes, size=7)

ci_ax2.text(0.025, 0.025, r'\textit{lower sensitivity}', ha='left', va='bottom',
        transform=ci_ax2.transAxes, size=7)
ci_ax2.text(0.975, 0.025, r'\textit{higher sensitivity}', ha='right',
        va='bottom', transform=ci_ax2.transAxes, size=7)

gs.tight_layout(fig, pad=0.3)
fig.savefig('params_sig.pdf')
