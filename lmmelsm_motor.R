### LMMELSM: calculates REWB model with scale and variance
# see Devlaminck et al. 2011 - Multisubject learning for 
# common spatial patterns in motor-imagery BCI
# setting cores higher than numbe rof chains makes no sense
# setwd('/Volumes/1TB_SSD/Arbeit/Charite/DrumsAndBrains')

library('LMMELSM')

##### Calculate all snare models and store to Results/motor/lmmelsm_snare_...#####
file = "Results/snare_data_motor.csv"
snare_data <- read.csv(file, sep=',')

fit_motor_snareAll <- lmmelsm(
  list(
    observed ~ deviation,
    location ~ musicality + trial + session + BP + ERD1 + ERD2 + ERD3 + ERD4 +
      ERD5 + ERS1 + ERS2 + ERS3 + ERS4 + ERS5,
    scale ~ musicality + trial + session + BP + ERD1 + ERD2 + ERD3 + ERD4 +
      ERD5 + ERS1 + ERS2 + ERS3 + ERS4 + ERS5,
    between ~ musicality + trial + session + BP + ERD1 + ERD2 + ERD3 + ERD4 +
      ERD5 + ERS1 + ERS2 + ERS3 + ERS4 + ERS5),
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

fit_motor_BP <- uni_lmmelms_fct(x = "BP", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_BP.txt")
print(summary(fit_motor_BP))

fit_motor_ERD1 <- uni_lmmelms_fct(x = "ERD1", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERD1.txt")
print(summary(fit_motor_ERD1))
fit_motor_ERS1 <- uni_lmmelms_fct(x = "ERS1", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERS1.txt")
print(summary(fit_motor_ERS1))

fit_motor_ERD2 <- uni_lmmelms_fct(x = "ERD2", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERD2.txt")
print(summary(fit_motor_ERD2))
fit_motor_ERS2 <- uni_lmmelms_fct(x = "ERS2", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERS2.txt")
print(summary(fit_motor_ERS2))

fit_motor_ERD3 <- uni_lmmelms_fct(x = "ERD3", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERD3.txt")
print(summary(fit_motor_ERD3))
fit_motor_ERS3 <- uni_lmmelms_fct(x = "ERS3", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERS3.txt")
print(summary(fit_motor_ERS3))

fit_motor_ERD4 <- uni_lmmelms_fct(x = "ERD4", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERD4.txt")
print(summary(fit_motor_ERD4))
fit_motor_ERS4 <- uni_lmmelms_fct(x = "ERS4", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERS4.txt")
print(summary(fit_motor_ERS4))

fit_motor_ERD5 <- uni_lmmelms_fct(x = "ERD5", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERD5.txt")
print(summary(fit_motor_ERD5))
fit_motor_ERS5 <- uni_lmmelms_fct(x = "ERS5", data=snare_data)
sink("Results/motor/lmmelsm_motor_snare_ERS5.txt")
print(summary(fit_motor_ERS5))

sink() # to free memory