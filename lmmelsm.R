### LMMELSM: calculates REWB model with scale and variance
# see Devlaminck et al. 2011 - Multisubject learning for 
# common spatial patterns in motor-imagery BCI
# setting cores higher than numbe rof chains makes no sense
# setwd('/Volumes/1TB_SSD/Arbeit/Charite/DrumsAndBrains')

library('LMMELSM')

##### Calculate all snare models and store to Results/models/lmmelsm_snare_...#####
file = "Results/snare_data_silence.csv"
snare_data <- read.csv(file, sep=',')

fit_snareAll <- lmmelsm(
	       list(
	observed ~ deviation,
	location ~ musicality + trial + session + Snare1_between + Snare2_between + 
	  Snare3_between + Snare1_within + Snare2_within + Snare3_within,
	scale ~ musicality + trial + session + Snare1_between + Snare2_between + 
	  Snare3_between + Snare1_within + Snare2_within + Snare3_within,
	between ~ musicality + trial + session + Snare1_between + Snare2_between + 
	  Snare3_between + Snare1_within + Snare2_within + Snare3_within),
	group = subject, data = snare_data, cores=8, iter=10000)
sink("Results/models/lmmelsm_snare_all.txt")
print(summary(fit_snareAll))


# test univariate models
uni_lmmelms_fct <- function(x, data) lmmelsm(
              list(
                as.formula(paste("observed ~", "deviation")),
                as.formula(paste("location ~", x)),
                as.formula(paste("scale ~", x)),
                as.formula(paste("between ~", x))),
              group = subject, data = data, cores = 8, iter=10000)

fit_musicality <- uni_lmmelms_fct(x = "musicality", data=snare_data)
sink("Results/models/lmmelsm_snare_musicality.txt")
print(summary(fit_musicality))

fit_trial <- uni_lmmelms_fct(x = "trial", data=snare_data)
sink("Results/models/lmmelsm_snare_trial.txt")
print(summary(fit_trial))

fit_session <- uni_lmmelms_fct(x = "session", data=snare_data)
sink("Results/models/lmmelsm_snare_session.txt")
print(summary(fit_session))

fit_Snare1_between <- uni_lmmelms_fct(x = "Snare1_between", data=snare_data)
sink("Results/models/lmmelsm_snare_Snare1_between.txt")
print(summary(fit_Snare1_between))

fit_Snare2_between <- uni_lmmelms_fct(x = "Snare2_between", data=snare_data)
sink("Results/models/lmmelsm_snare_Snare2_between.txt")
print(summary(fit_Snare2_between))

fit_Snare3_between <- uni_lmmelms_fct(x = "Snare3_between", data=snare_data)
sink("Results/models/lmmelsm_snare_Snare3_between.txt")
print(summary(fit_Snare3_between))

fit_Snare1_within <- uni_lmmelms_fct(x = "Snare1_within", data=snare_data)
sink("Results/models/lmmelsm_snare_Snare1_within.txt")
print(summary(fit_Snare1_within))

fit_Snare2_within <- uni_lmmelms_fct(x = "Snare2_within", data=snare_data)
sink("Results/models/lmmelsm_snare_Snare2_within.txt")
print(summary(fit_Snare2_within))

fit_Snare3_within <- uni_lmmelms_fct(x = "Snare3_within", data=snare_data)
sink("Results/models/lmmelsm_snare_Snare3_within.txt")
print(summary(fit_Snare3_within))

sink() # to free memory
##### Calculate all wdBlk models and store to Results/models/lmmelsm_wdBlk_...#####
file = "Results/wdBlk_data_silence.csv"
wdBlk_data <- read.csv(file, sep=',')

fit_wdBlkAll <- lmmelsm(
  list(
    observed ~ deviation,
    location ~ musicality + trial + session + WdBlk1_between + WdBlk2_between + 
      WdBlk3_between + WdBlk1_within + WdBlk2_within + WdBlk3_within,
    scale ~ musicality + trial + session + WdBlk1_between + WdBlk2_between + 
      WdBlk3_between + WdBlk1_within + WdBlk2_within + WdBlk3_within,
    between ~ musicality + trial + session + WdBlk1_between + WdBlk2_between + 
      WdBlk3_between + WdBlk1_within + WdBlk2_within + WdBlk3_within),
  group = subject, data = wdBlk_data, cores=8, iter=10000)
sink("Results/models/lmmelsm_wdBlk_all.txt")
print(summary(fit_wdBlkAll))


# test univariate models
uni_lmmelms_fct <- function(x, data) lmmelsm(
  list(
    
    as.formula(paste("observed ~", "deviation")),
    as.formula(paste("location ~", x)),
    as.formula(paste("scale ~", x)),
    as.formula(paste("between ~", x))),
  group = subject, data = data, cores = 8, iter=10000)

fit_musicality <- uni_lmmelms_fct(x = "musicality", data=wdBlk_data)
sink("Results/models/lmmelsm_wdBlk_musicality.txt")
print(summary(fit_musicality))

fit_trial <- uni_lmmelms_fct(x = "trial", data=wdBlk_data)
sink("Results/models/lmmelsm_wdBlk_trial.txt")
print(summary(fit_trial))

fit_session <- uni_lmmelms_fct(x = "session", data=wdBlk_data)
sink("Results/models/lmmelsm_wdBlk_session.txt")
print(summary(fit_session))

fit_wdBlk1_between <- uni_lmmelms_fct(x = "WdBlk1_between", data=wdBlk_data)
sink("Results/models/lmmelsm_wdBlk_WdBlk1_between.txt")
print(summary(fit_wdBlk1_between))

fit_wdBlk2_between <- uni_lmmelms_fct(x = "WdBlk2_between", data=wdBlk_data)
sink("Results/models/lmmelsm_wdBlk_WdBlk2_between.txt")
print(summary(fit_wdBlk2_between))

fit_wdBlk3_between <- uni_lmmelms_fct(x = "WdBlk3_between", data=wdBlk_data)
sink("Results/models/lmmelsm_wdBlk_WdBlk3_between.txt")
print(summary(fit_wdBlk3_between))

fit_wdBlk1_within <- uni_lmmelms_fct(x = "WdBlk1_within", data=wdBlk_data)
sink("Results/models/lmmelsm_wdBlk_WdBlk1_within.txt")
print(summary(fit_wdBlk1_within))

fit_wdBlk2_within <- uni_lmmelms_fct(x = "WdBlk2_within", data=wdBlk_data)
sink("Results/models/lmmelsm_wdBlk_WdBlk2_within.txt")
print(summary(fit_wdBlk2_within))

fit_wdBlk3_within <- uni_lmmelms_fct(x = "WdBlk3_within", data=wdBlk_data)
sink("Results/models/lmmelsm_wdBlk_WdBlk3_within.txt")
print(summary(fit_wdBlk3_within))

# (not working) apply for all of wdBlk_data except first two (subject, deviation)
# uni_fits <- lapply(wdBlk_data[,-2:-1], uni_lmmelms_fct, data=wdBlk_data) 
