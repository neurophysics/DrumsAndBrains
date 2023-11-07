# questions for gunnar
# ? should we add an averaged intercept or sth as we want to make a point with earlier and later?
# how about we divide into snare and wdblk for overview? im gonna do snare for now


import numpy as np
import scipy
from scipy import stats
import sys
import os.path
import re
import matplotlib as mpl
import matplotlib.pyplot as plt

result_folder = sys.argv[1]
snare = True #change this to make plots for duple or triple
condition = ['duple' if snare else 'triple'][0]

location_dict = {} ##should have:
# snare trial
# snare session
# snare musicality
# snare EEG1 within
# snare EEG2 within
# snare EEG1 between
# snare EEG2 between
# and these 7 again for wdblk so 14 overall
# only do snare for now...
scale_dict = {} ##should have:

names = ['EEG comp 2\n(within subj.)', 'EEG comp 2\n(between subj.)','EEG comp 1\n(within subj.)', 'EEG comp 1\n(between subj.)','musicality', 'session','trial',  'intercept']
name_i = 0

#see if plot is for duple or triple rhythm:
if snare:
        all_files = [result_folder + 'models/singleVariable/' + name +'.txt' for name in [ #it gets plotted in reverse order
                     'snare_Snare2_within',
                     'snare_Snare2_between',
                     'snare_Snare1_within',
                     'snare_Snare1_between',
                     'snare_musicality0995',
                     'snare_session',
                     'snare_trial0995',
                     'snare_intercept']]
else:
         all_files = [result_folder + 'models/singleVariable/' + name +'.txt' for name in [
                     'wdBlk_WdBlk2_within',
                     'wdBlk_WdBlk2_between',
                     'wdBlk_WdBlk1_within',
                     'wdBlk_WdBlk1_between',
                     'wdBlk_musicality',
                     'wdBlk_session',
                     'wdBlk_trial_10k',
                     'wdBlk_intercept']]

#loop over all single variabkle model files
for file_path in all_files:
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

                #e.g. deviation 0.0791 0.0787 0.021 0.0383 0.121 where order is  param  predictor   Mean Median    SD   Q2.5   Q97.5 so last five are numbers
                data_dict = {}
                name_list = ['loc_intercept', 'scale_intercept', 'loc_RE','scale_RE']
                for c,i in enumerate(intercept_lines):
                        parts = i.split()
                        key = name_list[c]
                        values = [float(value) for value in parts[-5:]]
                        data_dict[key] = values
                name_list_var = ['loc_variable', 'scale_variable', 'loc_group','scale_group'] #order in teh files never changes
                for c,i in enumerate(variable_lines):
                        parts = i.split()
                        key = name_list_var[c]
                        values = [float(value) for value in parts[-5:]]
                        data_dict[key] = values

                #we only need the one location and scale CIs for now
                key = names[name_i]
                if key=='intercept': #exception bc some lines are missing
                        location_dict[key] = data_dict['loc_intercept']
                        scale_dict[key] = data_dict['scale_intercept']
                else:
                        location_dict[key] = data_dict['loc_variable']
                        scale_dict[key] = data_dict['scale_variable']
                name_i+=1


#params_p=np.arange(0.1,0.8,0.1) #CHANGE LATER by reading from _p file

#final data should have shape (number of variables, 2) 2 bc its an interval

#############
params_CI_loc = np.vstack(
                [value[-2:] for key, value in location_dict.items()]
                )

params_CI_scale = np.vstack(
                [value[-2:] for key, value in scale_dict.items()]
                )


###########################################
# plot the bootstrap confidence intervals #
###########################################

blind_ax = dict(top=False, bottom=False, left=False, right=False,
        labelleft=False, labelright=False, labeltop=False,
        labelbottom=False)

fig = plt.figure()

gs = mpl.gridspec.GridSpec(nrows=1, ncols=5, width_ratios=[1,5,0.1,0.1,5]) #have one empty/title one in the middle for some spacing between plots

### middle - title
maintitle_ax = fig.add_subplot(gs[0,2], frame_on=False)
maintitle_ax.tick_params(**blind_ax)
maintitle_ax.set_title('CI of single variable models in '+
                       condition + ' condition', size=15, pad=25.0)

### left - location
title_ax1 = fig.add_subplot(gs[0,:2], frame_on=False)
title_ax1.tick_params(**blind_ax)
title_ax1.set_title(r'\textbf{location}', size=10)

# left - variable names
text_ax1 = fig.add_subplot(gs[0,0], frame_on=False)
text_ax1.tick_params(**blind_ax)

for e,d in enumerate(names):
        text_ax1.text(0.95, float(e+1)/(len(location_dict)+1), d,         ha='right', va='center', transform=text_ax1.transAxes, size=10)

# left - CI
ci_ax1 = fig.add_subplot(gs[0,1], frame_on=False)
ci_ax1.tick_params(**dict(left=False, labelleft=False, right=False,
    labelright=False))
ci_ax1.plot([0,1], [0,0], 'k-', transform=ci_ax1.transAxes)
ci_ax1.axvline(0, ls=':', lw=0.75)

trans1 = mpl.transforms.blended_transform_factory(
    ci_ax1.transData, ci_ax1.transAxes)

for i in range(len(params_CI_loc)):
    if (params_CI_loc[i][1]*params_CI_loc[i][0])>=0:
        color_now='red'
    else:
        color_now='k'
    ci_ax1.plot(params_CI_loc[i], 2*[float(i+1)/(len(location_dict)+1)], ls='-', c=color_now,
            transform=trans1)
## p values take up too much space probably
#     ci_ax1.text(-params_CI_loc[i].mean(), float(len(location_dict)-i)/len(location_dict) + 0.02, r'$p=%.3f$' % params_p[i],
#             transform=trans1, ha='center', va='bottom', size=7, color=color_now)


### right - scale
title_ax2 = fig.add_subplot(gs[0,3:], frame_on=False)
title_ax2.tick_params(**blind_ax)
title_ax2.set_title(r"""\textbf{scale}""", size=10)

# right - variable names
text_ax2 = fig.add_subplot(gs[0,3], frame_on=False)
text_ax2.tick_params(**blind_ax)

for e,d in enumerate(names):
        text_ax2.text(0.95, float(e+1)/(len(scale_dict)+1), '', ha='right', va='center', transform=text_ax2.transAxes,
        size=10)

# right - CI
ci_ax2 = fig.add_subplot(gs[0,4], frame_on=False)
ci_ax2.tick_params(**dict(left=False, labelleft=False, right=False,
    labelright=False))
ci_ax2.plot([0,1], [0,0], 'k-', transform=ci_ax2.transAxes)
ci_ax2.axvline(0, ls=':', lw=0.75)

trans2 = mpl.transforms.blended_transform_factory(
    ci_ax2.transData, ci_ax2.transAxes)

for i in range(len(params_CI_scale)):
    if (params_CI_scale[i][1]*params_CI_scale[i][0])>=0:
        color_now='r'
    else:
        color_now='k'
    ci_ax2.plot(params_CI_scale[i], 2*[float(i+1)/(len(scale_dict)+1)], ls='-', c=color_now,
            transform=trans2)
    ## p values take up too much space probably
    # ci_ax2.text(params_CI_scale[i].mean(), float(len(scale_dict)-i)/len(scale_dict),
    #         r'$p=%.3f$' % params_p[i],
    #         transform=trans2, ha='center', va='bottom', size=7,
    #         color=color_now)

ci_ax1.set_xlim([-np.abs(params_CI_loc).max(), np.abs(params_CI_loc).max()])
ci_ax2.set_xlim([-np.abs(params_CI_scale).max(), np.abs(params_CI_scale).max()])

# bottom - plot legend
ci_ax1.set_xlabel('model coefficients')
ci_ax2.set_xlabel('model coefficients')

ci_ax1.text(0.025, 0.025, r'\textit{earlier}', ha='left', va='bottom',
        transform=ci_ax1.transAxes, size=7)
ci_ax1.text(0.975, 0.025, r'\textit{later}', ha='right',
        va='bottom', transform=ci_ax1.transAxes, size=7)

ci_ax2.text(0.025, 0.025, r'\textit{consistent}', ha='left', va='bottom',
        transform=ci_ax2.transAxes, size=7)
ci_ax2.text(0.975, 0.025, r'\textit{irregular}', ha='right',
        va='bottom', transform=ci_ax2.transAxes, size=7)

#gs.tight_layout(fig, pad=0.3)

fig.savefig(os.path.join(result_folder, 'models/singleVariable/paramsCI_'+ condition +'.pdf'))
