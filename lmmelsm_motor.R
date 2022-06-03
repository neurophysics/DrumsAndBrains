### LMMELSM: calculates REWB model with scale and variance
# see Devlaminck et al. 2011 - Multisubject learning for 
# common spatial patterns in motor-imagery BCI
# setting cores higher than numbe rof chains makes no sense
# 
setwd('/Volumes/1TB_SSD/Arbeit/Charite/DrumsAndBrains')

library('LMMELSM')

##### Calculate all snare models and store to Results/motor/lmmelsm_snare_...#####
file = "Results/snare_data_motor.csv"
snare_data <- read.csv(file, sep=',')

# todo: add within and between
# only take alpha and beta erd and delta ers?
fit_motor_snareAll <- lmmelsm(
  list(
    observed ~ deviation,
    location ~ musicality + trial + session + BP_within + BP_between + 
      ERD1_within + ERD1_between + ERD2_within + ERD2_between +
      ERS1_within + ERS1_between + ERS2_within + ERS2_between, #ohne ERS wenn p<0.05
    scale ~ musicality + trial + session + BP_within + BP_between + 
      ERD1_within + ERD1_between + ERD2_within + ERD2_between +
      ERS1_within + ERS1_between + ERS2_within + ERS2_between,
    between ~ musicality + trial + session + BP_within + BP_between + 
      ERD1_within + ERD1_between + ERD2_within + ERD2_between +
      ERS1_within + ERS1_between + ERS2_within + ERS2_between),
  group = subject, data = snare_data, cores=8, iter=10000)
sink("Results/motor/lmmelsm_motor_snare_all.txt")
print(summary(fit_motor_snareAll))


# test univariate models
uni_lmmelms_fct <- function(x, data) lmmelsm(
  list(
    as.formula(paste("observed ~", "deviation")),
    as.formula(paste("location ~", x)),
    as.formula(paste("scale ~", x)),
    as.formula(paste("between ~", x))),
  group = subject, data = data, cores = 8, iter=10000)

fit_motor_musicality_snare <- uni_lmmelms_fct(x = "musicality", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_musicality.txt")
print(summary(fit_motor_musicality_snare))

fit_motor_trial_snare <- uni_lmmelms_fct(x = "trial", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_trial.txt")
print(summary(fit_motor_trial_snare))

fit_motor_session_snare <- uni_lmmelms_fct(x = "session", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_session.txt")
print(summary(fit_motor_session_snare))

fit_motor_BP_snare <- uni_lmmelms_fct(x = "BP_within", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_BP_within.txt")
print(summary(fit_motor_BP_snare))

fit_motor_BPbw_snare <- uni_lmmelms_fct(x = "BP_between", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_BP_between.txt")
print(summary(fit_motor_BPbw_snare))

fit_motor_ERS1_snare <- uni_lmmelms_fct(x = "ERS1_within", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERS1_within.txt")
print(summary(fit_motor_ERS1_snare))

fit_motor_ERS1bw_snare <- uni_lmmelms_fct(x = "ERS1_between", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERS1_between.txt")
print(summary(fit_motor_ERS1bw_snare))

fit_motor_ERS2_snare <- uni_lmmelms_fct(x = "ERS2_within", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERS2_within.txt")
print(summary(fit_motor_ERS2_snare))

fit_motor_ERS2bw_snare <- uni_lmmelms_fct(x = "ERS2_between", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERS2_between.txt")
print(summary(fit_motor_ERS2bw_snare))

fit_motor_ERD1_snare <- uni_lmmelms_fct(x = "ERD1_within", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERD3_within.txt")
print(summary(fit_motor_ERD1_snare))

fit_motor_ERD1bw_snare <- uni_lmmelms_fct(x = "ERD1_between", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERD3_between.txt")
print(summary(fit_motor_ERD1bw_snare))

fit_motor_ERD2_snare <- uni_lmmelms_fct(x = "ERD2_within", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERD4_within.txt")
print(summary(fit_motor_ERD2_snare))

fit_motor_ERD2_bw_snare <- uni_lmmelms_fct(x = "ERD2_between", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERD4_between.txt")
print(summary(fit_motor_ERD2_bw_snare))

sink() # to free memory



##### same for wdBlk #####
file = "Results/wdBlk_data_motor.csv"
wdBlk_data <- read.csv(file, sep=',')

# todo: add within and between
# only take alpha and beta erd and delta ers?
fit_motor_wdBlkAll <- lmmelsm(
  list(
    observed ~ deviation,
    location ~ musicality + trial + session + BP_within + BP_between + 
      ERD1_within + ERD1_between + ERD2_within + ERD2_between +
      ERS1_within + ERS1_between + ERS2_within + ERS2_between, #ohne ERS wenn p<0.05
    scale ~ musicality + trial + session + BP_within + BP_between + 
      ERD1_within + ERD1_between + ERD2_within + ERD2_between +
      ERS1_within + ERS1_between + ERS2_within + ERS2_between,
    between ~ musicality + trial + session + BP_within + BP_between + 
      ERD1_within + ERD1_between + ERD2_within + ERD2_between +
      ERS1_within + ERS1_between + ERS2_within + ERS2_between),
  group = subject, data = wdBlk_data, cores=8, iter=10000)
sink("Results/motor/lmmelsm_motor_wdBlk_all.txt")
print(summary(fit_motor_wdBlkAll))


# test univariate models
uni_lmmelms_fct <- function(x, data) lmmelsm(
  list(
    as.formula(paste("observed ~", "deviation")),
    as.formula(paste("location ~", x)),
    as.formula(paste("scale ~", x)),
    as.formula(paste("between ~", x))),
  group = subject, data = data, cores = 8, iter=10000)

fit_motor_musicality_wdBlk <- uni_lmmelms_fct(x = "musicality", data=wdBlk_data)
sink("Results/motor/lmmelsm_motor_wdBlk_musicality.txt")
print(summary(fit_motor_musicality_wdBlk))

fit_motor_trial_wdBlk <- uni_lmmelms_fct(x = "trial", data=wdBlk_data)
sink("Results/motor/lmmelsm_motor_wdBlk_trial.txt")
print(summary(fit_motor_trial_wdBlk))

fit_motor_session_wdBlk <- uni_lmmelms_fct(x = "session", data=wdBlk_data)
sink("Results/motor/lmmelsm_motor_wdBlk_session.txt")
print(summary(fit_motor_session_wdBlk))

fit_motor_BP_wdBlk <- uni_lmmelms_fct(x = "BP_within", data=wdBlk_data)
sink("Results/motor/lmmelsm_motor_wdBlk_BP_within.txt")
print(summary(fit_motor_BP_wdBlk))

fit_motor_BPbw_wdBlk <- uni_lmmelms_fct(x = "BP_between", data=wdBlk_data)
sink("Results/motor/lmmelsm_motor_wdBlk_BP_between.txt")
print(summary(fit_motor_BPbw_wdBlk))

fit_motor_ERS1_wdBlk <- uni_lmmelms_fct(x = "ERS1_within", data=wdBlk_data)
sink("Results/motor/lmmelsm_motor_wdBlk_ERS1_within.txt")
print(summary(fit_motor_ERS1_wdBlk))

fit_motor_ERS1bw_wdBlk <- uni_lmmelms_fct(x = "ERS1_between", data=wdBlk_data)
sink("Results/motor/lmmelsm_motor_wdBlk_ERS1_between.txt")
print(summary(fit_motor_ERS1bw_wdBlk))

fit_motor_ERS2_wdBlk <- uni_lmmelms_fct(x = "ERS2_within", data=wdBlk_data)
sink("Results/motor/lmmelsm_motor_wdBlk_ERS2_within.txt")
print(summary(fit_motor_ERS2_wdBlk))

fit_motor_ERS2bw_wdBlk <- uni_lmmelms_fct(x = "ERS2_between", data=wdBlk_data)
sink("Results/motor/lmmelsm_motor_wdBlk_ERS2_between.txt")
print(summary(fit_motor_ERS2bw_wdBlk))

fit_motor_ERD1_wdBlk <- uni_lmmelms_fct(x = "ERD1_within", data=wdBlk_data)
sink("Results/motor/lmmelsm_motor_wdBlk_ERD3_within.txt")
print(summary(fit_motor_ERD1_wdBlk))

fit_motor_ERD1bw_wdBlk <- uni_lmmelms_fct(x = "ERD1_between", data=wdBlk_data)
sink("Results/motor/lmmelsm_motor_wdBlk_ERD3_between.txt")
print(summary(fit_motor_ERD1bw_wdBlk))

fit_motor_ERD2_wdBlk <- uni_lmmelms_fct(x = "ERD2_within", data=wdBlk_data)
sink("Results/motor/lmmelsm_motor_wdBlk_ERD4_within.txt")
print(summary(fit_motor_ERD2_wdBlk))

fit_motor_ERD2_bw_wdBlk <- uni_lmmelms_fct(x = "ERD2_between", data=wdBlk_data)
sink("Results/motor/lmmelsm_motor_wdBlk_ERD4_between.txt")
print(summary(fit_motor_ERD2_bw_wdBlk))

sink() # to free memory

