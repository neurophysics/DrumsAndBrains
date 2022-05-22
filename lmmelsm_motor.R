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
      ERD3_within + ERD3_between + ERD4_within + ERD4_between +
      ERS1_within + ERS1_between + ERS2_within + ERS2_between,
    scale ~ musicality + trial + session + BP_within + BP_between + 
      ERD3_within + ERD3_between + ERD4_within + ERD4_between +
      ERS1_within + ERS1_between + ERS2_within + ERS2_between,
    between ~ musicality + trial + session + BP_within + BP_between + 
      ERD3_within + ERD3_between + ERD4_within + ERD4_between +
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

fit_motor_musicality <- uni_lmmelms_fct(x = "musicality", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_musicality.txt")
print(summary(fit_motor_musicality))

fit_motor_trial <- uni_lmmelms_fct(x = "trial", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_trial.txt")
print(summary(fit_motor_trial))

fit_motor_session <- uni_lmmelms_fct(x = "session", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_session.txt")
print(summary(fit_motor_session))

fit_motor_BP <- uni_lmmelms_fct(x = "BP_within", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_BP_within.txt")
print(summary(fit_motor_BP))

fit_motor_BPbw <- uni_lmmelms_fct(x = "BP_between", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_BP_between.txt")
print(summary(fit_motor_BPbw))

fit_motor_ERS1 <- uni_lmmelms_fct(x = "ERS1_within", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERS1_within.txt")
print(summary(fit_motor_ERS1))

fit_motor_ERS1bw <- uni_lmmelms_fct(x = "ERS1_between", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERS1_between.txt")
print(summary(fit_motor_ERS1bw))

fit_motor_ERS2 <- uni_lmmelms_fct(x = "ERS2_within", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERS2_within.txt")
print(summary(fit_motor_ERS2))

fit_motor_ERS2bw <- uni_lmmelms_fct(x = "ERS2_between", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERS2_between.txt")
print(summary(fit_motor_ERS2bw))

fit_motor_ERD3 <- uni_lmmelms_fct(x = "ERD3_within", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERD3_within.txt")
print(summary(fit_motor_ERD3))

fit_motor_ERD3bw <- uni_lmmelms_fct(x = "ERD3_between", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERD3_between.txt")
print(summary(fit_motor_ERD3bw))

fit_motor_ERD4 <- uni_lmmelms_fct(x = "ERD4_within", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERD4_within.txt")
print(summary(fit_motor_ERD4))

fit_motor_ERD4_bw <- uni_lmmelms_fct(x = "ERD4_between", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERD4_between.txt")
print(summary(fit_motor_ERD4_bw))

sink() # to free memory
