# ? should we add an averaged intercept or sth as we want to make a point with earlier and later?

import numpy as np
import scipy
from scipy import stats
import sys
import os.path
import re
import matplotlib as mpl
import matplotlib.pyplot as plt

result_folder = sys.argv[1]
singleVariable = True
snare = False #change this to make plots for duple or triple
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



name_i = 0 # loop through names
if singleVariable:
        names = ['EEG 2, triple freq.\n(within subj.)',
                'EEG 2, triple freq.\n(between subj.)',
                'EEG 1, triple freq.\n(within subj.)',
                'EEG 1, triple freq.\n(between subj.)',
                'EEG 2, duple freq.\n(within subj.)',
                'EEG 2, duple freq.\n(between subj.)',
                'EEG 1, duple freq.\n(within subj.)',
                'EEG 1, duple freq.\n(between subj.)',
                'musicality', 'session', 'trial',  'intercept']       #see if plot is for duple or triple rhythm:
        if snare:
                all_files = [result_folder + 'models/singleVariable/' + name +'.txt' for name in [ #it gets plotted in reverse order
                       'snare_Snare2_within',
                        'snare_Snare2_between',
                        'snare_Snare1_within',
                        'snare_Snare1_between',                        'snare_WdBlk2_within',
                        'snare_WdBlk2_between',
                        'snare_WdBlk1_within',
                        'snare_WdBlk1_between',
                        'snare_musicality0995',
                        'snare_session',
                        'snare_trial10k',
                        'snare_intercept']]
        else:
                all_files = [result_folder + 'models/singleVariable/' + name +'.txt' for name in [
                        'wdBlk_Snare2_within',
                        'wdBlk_Snare2_between',
                        'wdBlk_Snare1_within',
                        'wdBlk_Snare1_between',                        'wdBlk_WdBlk2_within',
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
                                #ignore intercept for scale so just 0
                                scale_dict[key] = [0]*len(data_dict['scale_intercept'])
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

        #### read p values ####
        params_p_loc = []
        params_p_scale = []
        all_p_files = [f[:-4]+'_p.txt' for f in all_files]
        for file_path in all_p_files:
                # Open and read the file
                with open(file_path, 'r') as file:
                        p_lines = []
                        for line in file:
                                p_lines.append(line.strip().split())
                params_p_loc.append(float(p_lines[2][-1]))
                params_p_scale.append(float(p_lines[4][-1]))


else: #all models
        #### read model ####
        if snare:
                names = [
                        'EEG 2, duple freq.\n(within subj.)',
                        'EEG 2, duple freq.\n(between subj.)',
                        'EEG 1, duple freq.\n(within subj.)',
                        'EEG 1, duple freq.\n(between subj.)',
                        'musicality', 'session', 'trial',  'intercept']
                file_path = result_folder + 'models/snare_all25k099.txt'
        else:
                 names = [
                        'EEG 2, triple freq.\n(within subj.)',
                        'EEG 2, triple freq.\n(between subj.)',
                        'EEG 1, triple freq.\n(within subj.)',
                        'EEG 1, triple freq.\n(between subj.)',
                        'musicality', 'session', 'trial',  'intercept']
                 file_path = result_folder + 'models/wdBlk_all25k099.txt'
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

        # last two values are CI, order as in names
        location_list = [
                variable_lines[6].split()[-2:],#EEG2 within
                variable_lines[4].split()[-2:], #EEG2 between
                variable_lines[5].split()[-2:], #EEG1 within
                variable_lines[3].split()[-2:], #EEG1 between
                variable_lines[0].split()[-2:], #musicality
                variable_lines[2].split()[-2:], #session
                variable_lines[1].split()[-2:], #trial
                intercept_lines[0].split()[-2:] # intercept
        ]
        params_CI_loc = np.vstack([list(map(float, sublist)) for sublist in location_list])
        scale_list = [ #7 lines later
                variable_lines[7+6].split()[-2:],#EEG2 within
                variable_lines[7+4].split()[-2:], #EEG2 between
                variable_lines[7+5].split()[-2:], #EEG1 within
                variable_lines[7+3].split()[-2:], #EEG1 between
                variable_lines[7+0].split()[-2:], #musicality
                variable_lines[7+2].split()[-2:], #session
                variable_lines[7+1].split()[-2:], #trial
                [0.,0.] # intercept holds no info for scale
                ]
        params_CI_scale = np.vstack([list(map(float, sublist)) for sublist in scale_list])

        #### read p values ####
        if snare:
                file_path_p = result_folder + 'models/snare_all25k099_p.txt'
        else:
                file_path_p = result_folder + 'models/wdBlk_all25k099_p.txt'
        with open(file_path_p, 'r') as file:
                p_lines = []
                for line in file:
                        p_lines.append(line.strip().split())

        params_p_loc = [ #first line is header
                float(p_lines[8][-1]),#EEG2 within
                float(p_lines[6][-1]), #EEG2 between
                float(p_lines[7][-1]), #EEG1 within
                float(p_lines[5][-1]), #EEG1 between
                float(p_lines[2][-1]), #musicality
                float(p_lines[4][-1]), #session
                float(p_lines[3][-1]), #trial
                float(p_lines[1][-1]) # intercept
        ]
        params_p_scale = [ #first line is header, 8 lines later (includes intercept)
                float(p_lines[8+8][-1]),#EEG2 within
                float(p_lines[8+6][-1]), #EEG2 between
                float(p_lines[8+7][-1]), #EEG1 within
                float(p_lines[8+5][-1]), #EEG1 between
                float(p_lines[8+2][-1]), #musicality
                float(p_lines[8+4][-1]), #session
                float(p_lines[8+3][-1]), #trial
                float(p_lines[8+1][-1]) # intercept
        ]

###########################################
# plot the bootstrap confidence intervals #
###########################################

blind_ax = dict(top=False, bottom=False, left=False, right=False,
        labelleft=False, labelright=False, labeltop=False,
        labelbottom=False)

if singleVariable:
        fig = plt.figure(figsize=(7, 7))
else:
        fig = plt.figure()

gs = mpl.gridspec.GridSpec(nrows=1, ncols=4, width_ratios=[1,5,0.1,5]) #have one empty/title one in the middle for some spacing between plots so it'S (variable names, latency, spacing with title, jitter)'

if singleVariable:
        fig.suptitle('CI of univariate models in '+
                        condition + ' condition', fontsize=15)
else:
        fig.suptitle('CI of multivariate model in '+
                       condition + ' condition', fontsize=15)


### left - location

# most left - variable names
text_ax1 = fig.add_subplot(gs[0,0], frame_on=False)
text_ax1.tick_params(**blind_ax)

for e,d in enumerate(names):
        text_ax1.text(0.95, float(e+1)/(len(params_CI_loc)+1), d,         ha='right', va='center', transform=text_ax1.transAxes, size=10)

# left - CI
ci_ax1 = fig.add_subplot(gs[0,1], frame_on=False)
ci_ax1.tick_params(**dict(left=False, labelleft=False, right=False,
    labelright=False))
ci_ax1.plot([0,1], [0,0], 'k-', transform=ci_ax1.transAxes)
ci_ax1.axvline(0, ls=':', lw=0.75)
ci_ax1.set_title(r'\textbf{latency (s)}', size=10)

trans1 = mpl.transforms.blended_transform_factory(
    ci_ax1.transData, ci_ax1.transAxes)

for i in range(len(params_CI_loc)):
        if (params_CI_loc[i][1]*params_CI_loc[i][0])>0:
                color_now='red'
                # p values only for significant parameters
                current_p = r'$p=%.3f$' % params_p_loc[i]
                if params_p_loc[i]<0.001:
                        current_p = r'$p<0.001$'
                ci_ax1.text(params_CI_loc[i].mean(), #middle of bar
                            float(i+1)/(len(params_CI_loc)+1)+0.01,
                            current_p,
                        transform=trans1, ha='center', va='bottom', size=7, color=color_now)
        else:
                color_now='k'
        ci_ax1.plot(params_CI_loc[i],
                    2*[float(i+1)/(len(params_CI_loc)+1)],
                    ls='-', c=color_now,transform=trans1)

### middle - spacing

### right - scale
# right - CI
title_ax2 = fig.add_subplot(gs[0,3], frame_on=False)
title_ax2.tick_params(**blind_ax)
title_ax2.set_title(r"""\textbf{jitter}""", size=10)

ci_ax2 = fig.add_subplot(gs[0,3], frame_on=False)
ci_ax2.tick_params(**dict(left=False, labelleft=False, right=False,
    labelright=False))
ci_ax2.plot([0,1], [0,0], 'k-', transform=ci_ax2.transAxes)
ci_ax2.axvline(0, ls=':', lw=0.75)

trans2 = mpl.transforms.blended_transform_factory(
    ci_ax2.transData, ci_ax2.transAxes)

for i in range(len(params_CI_scale)-1): #subtract one because last one is intercept we ignore
        if (params_CI_scale[i][1]*params_CI_scale[i][0])>0:
                color_now='r'
                # p values only for significant parameters
                current_p = r'$p=%.3f$' % params_p_scale[i]
                if params_p_scale[i]<0.001:
                        current_p = r'$p<0.001$'
                ci_ax2.text(
                params_CI_scale[i].mean(),                             float(i+1)/(len(params_p_scale)+1)+0.01,
                current_p,
                transform=trans2, ha='center', va='bottom', size=7, color=color_now)
        else:
                color_now='k'
        ci_ax2.plot(params_CI_scale[i],
                2*[float(i+1)/(len(params_CI_scale)+1)],
                ls='-', c=color_now, transform=trans2)


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
if singleVariable:
        fig.savefig(os.path.join(result_folder, 'models/singleVariable/paramsCI_'+ condition +'.pdf'))
else:
        fig.savefig(os.path.join(result_folder, 'models/paramsCI_'+ condition +'.pdf'))

