### LMMELSM: calculates REWB model with scale and variance
# see Devlaminck et al. 2011 - Multisubject learning for 
# common spatial patterns in motor-imagery BCI
# setting cores higher than numbe rof chains makes no sense
# 
# diagnostics:
# high divergences => increase step size (or treedepth)
# high rhat => increase iterations

setwd('/Volumes/1TB_SSD/Arbeit/Charite/DrumsAndBrains')
prob = 0.9984375 #1-0.05/32, because bonferroni with 32 variables
library('LMMELSM')
library('rstan')

file = "Results/snare_data_silence.csv"
snare_data <- read.csv(file, sep=',')

file = "Results/wdBlk_data_silence.csv"
wdBlk_data <- read.csv(file, sep=',')

##### univariate snare models #####
# intercept only
fit_snare_intercept <- lmmelsm(
  list(
    observed ~ deviation),
  group = subject, data = snare_data, cores = 8, iter=5000, warmup=1000,
  control = list(adapt_delta = 0.99, stepsize = 1, max_treedepth = 10))
sink("Results/models/singleVariable/snare_intercept.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_snare_intercept))
save(fit_snare_intercept, file = "Results/models/singleVariable/snare_intercept.RData")

# single variable
uni_lmmelms_fct <- function(x, data) lmmelsm(
  list(
    as.formula(paste("observed ~", "deviation")),
    as.formula(paste("location ~", x)),
    as.formula(paste("scale ~", x)),
    as.formula(paste("between ~", x))),
  group = subject, data = snare_data, cores = 8, iter=5000, warmup=1000,
  control = list(adapt_delta = 0.99, stepsize = 1, max_treedepth = 10))

fit_musicality <- uni_lmmelms_fct(x = "musicality", data=snare_data)
sink("Results/models/singleVariable/snare_musicality_2.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_musicality))
save(fit_musicality, file = "Results/models/singleVariable/snare_musicality_2.RData")

fit_trial <- uni_lmmelms_fct(x = "trial", data=snare_data)
sink("Results/models/singleVariable/snare_trial10k_2.txt")
print('iter=10000, warmup=3000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_trial))
save(fit_trial, file = "Results/models/singleVariable/snare_trial10k_2.RData")

fit_session <- uni_lmmelms_fct(x = "session", data=snare_data)
sink("Results/models/singleVariable/snare_session.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_session))
save(fit_session, file = "Results/models/singleVariable/snare_session.RData")

fit_Snare1_between <- uni_lmmelms_fct(x = "Snare1_between", data=snare_data)
sink("Results/models/singleVariable/snare_Snare1_between.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_Snare1_between))
save(fit_Snare1_between, file = "Results/models/singleVariable/snare_Snare1_between.RData")

fit_Snare2_between <- uni_lmmelms_fct(x = "Snare2_between", data=snare_data)
sink("Results/models/singleVariable/snare_Snare2_between.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_Snare2_between))
save(fit_Snare2_between, file = "Results/models/singleVariable/snare_Snare2_between.RData")

fit_Snare1_within <- uni_lmmelms_fct(x = "Snare1_within", data=snare_data)
sink("Results/models/singleVariable/snare_Snare1_within.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_Snare1_within))
save(fit_Snare1_within, file = "Results/models/singleVariable/snare_Snare1_within.RData")

fit_Snare2_within <- uni_lmmelms_fct(x = "Snare2_within", data=snare_data)
sink("Results/models/singleVariable/snare_Snare2_within.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_Snare2_within))
save(fit_Snare2_within, file = "Results/models/singleVariable/snare_Snare2_within.RData")

fit_WdBlk1_between <- uni_lmmelms_fct(x = "WdBlk1_between", data=snare_data)
sink("Results/models/singleVariable/snare_WdBlk1_between.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_WdBlk1_between))
save(fit_WdBlk1_between, file = "Results/models/singleVariable/snare_WdBlk1_between.RData")

fit_WdBlk2_between <- uni_lmmelms_fct(x = "WdBlk2_between", data=snare_data)
sink("Results/models/singleVariable/snare_WdBlk2_between.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_WdBlk2_between))
save(fit_WdBlk2_between, file = "Results/models/singleVariable/snare_WdBlk2_between.RData")

fit_WdBlk1_within <- uni_lmmelms_fct(x = "WdBlk1_within", data=snare_data)
sink("Results/models/singleVariable/snare_WdBlk1_within.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_WdBlk1_within))
save(fit_WdBlk1_within, file = "Results/models/singleVariable/snare_WdBlk1_within.RData")

fit_WdBlk2_within <- uni_lmmelms_fct(x = "WdBlk2_within", data=snare_data)
sink("Results/models/singleVariable/snare_WdBlk2_within.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_WdBlk2_within))
save(fit_WdBlk2_within, file = "Results/models/singleVariable/snare_WdBlk2_within.RData")
closeAllConnections()


##### univariate wdBlk models #####
# intercept only
fit_wdBlk_intercept <- lmmelsm(
  list(
    observed ~ deviation),
  group = subject, data = wdBlk_data, cores = 8, iter=5000, warmup=1000,
  control = list(adapt_delta = 0.99, stepsize = 1, max_treedepth = 10))
sink("Results/models/singleVariable/wdBlk_intercept.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_wdBlk_intercept))
save(fit_wdBlk_intercept, file = "Results/models/singleVariable/wdBlk_intercept.RData")

# single variable
uni_lmmelms_fct <- function(x, data) lmmelsm(
  list(
    as.formula(paste("observed ~", "deviation")),
    as.formula(paste("location ~", x)),
    as.formula(paste("scale ~", x)),
    as.formula(paste("between ~", x))),
  group = subject, data = wdBlk_data, cores = 8, iter=5000, warmup=1000,
  control = list(adapt_delta = 0.99, stepsize = 1, max_treedepth = 10))

fit_musicality <- uni_lmmelms_fct(x = "musicality", data=wdBlk_data)
sink("Results/models/singleVariable/wdBlk_musicality.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_musicality))
save(fit_musicality, file = "Results/models/singleVariable/wdBlk_musicality.RData")

fit_trial <- uni_lmmelms_fct(x = "trial", data=wdBlk_data)
sink("Results/models/singleVariable/wdBlk_trial0995_10k.txt")
print('iter=10000, warmup=3000, adapt_delta=0.995, stepsize = 1, max_treedepth = 10')
print(summary(fit_trial))
save(fit_trial, file = "Results/models/singleVariable/wdBlk_trial0995_10k.RData")

fit_session <- uni_lmmelms_fct(x = "session", data=wdBlk_data)
sink("Results/models/singleVariable/wdBlk_session.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_session))
save(fit_session, file = "Results/models/singleVariable/wdBlk_session.RData")

fit_WdBlk1_between <- uni_lmmelms_fct(x = "WdBlk1_between", data=wdBlk_data)
sink("Results/models/singleVariable/wdBlk_WdBlk1_between.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_WdBlk1_between))
save(fit_WdBlk1_between, file = "Results/models/singleVariable/wdBlk_WdBlk1_between.RData")

fit_WdBlk2_between <- uni_lmmelms_fct(x = "WdBlk2_between", data=wdBlk_data)
sink("Results/models/singleVariable/wdBlk_WdBlk2_between.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_WdBlk2_between))
save(fit_WdBlk2_between, file = "Results/singleVariable/models/wdBlk_WdBlk2_between.RData")

fit_WdBlk1_within <- uni_lmmelms_fct(x = "WdBlk1_within", data=wdBlk_data)
sink("Results/models/singleVariable/wdBlk_WdBlk1_within.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_WdBlk1_within))
save(fit_WdBlk1_within, file = "Results/singleVariable/models/wdBlk_WdBlk1_within.RData")

fit_WdBlk2_within <- uni_lmmelms_fct(x = "WdBlk2_within", data=wdBlk_data)
sink("Results/models/singleVariable/wdBlk_WdBlk2_within.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_WdBlk2_within))
save(fit_WdBlk2_within, file = "Results/models/singleVariable/wdBlk_WdBlk2_within.RData")


fit_Snare1_between <- uni_lmmelms_fct(x = "Snare1_between", data=wdBlk_data)
sink("Results/models/singleVariable/wdBlk_Snare1_between.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_Snare1_between))
save(fit_Snare1_between, file = "Results/models/singleVariable/wdBlk_Snare1_between.RData")

fit_Snare2_between <- uni_lmmelms_fct(x = "Snare2_between", data=wdBlk_data)
sink("Results/models/singleVariable/wdBlk_Snare2_between.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_Snare2_between))
save(fit_Snare2_between, file = "Results/models/singleVariable/wdBlk_Snare2_between.RData")

fit_Snare1_within <- uni_lmmelms_fct(x = "Snare1_within", data=wdBlk_data)
sink("Results/models/singleVariable/wdBlk_Snare1_within.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_Snare1_within))
save(fit_Snare1_within, file = "Results/models/singleVariable/wdBlk_Snare1_within.RData")

fit_Snare2_within <- uni_lmmelms_fct(x = "Snare2_within", data=wdBlk_data)
sink("Results/models/singleVariable/wdBlk_Snare2_within.txt")
print('iter=5000, warmup=1000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_Snare2_within))
save(fit_Snare2_within, file = "Results/models/singleVariable/wdBlk_Snare2_within.RData")

closeAllConnections() #closes sink so that output is back to console

# (not working) apply for all of wdBlk_data except first two (subject, deviation)
# uni_fits <- lapply(wdBlk_data[,-2:-1], uni_lmmelms_fct, data=wdBlk_data) 


##### snare (duple) All...#####
fit_snareAll <- lmmelsm(
  list(
    observed ~ deviation,
    location ~ musicality + trial + session + 
      # Snare1_between + Snare2_between + 
      # Snare1_within + Snare2_within +
      WdBlk1_between + WdBlk2_between + 
      WdBlk1_within + WdBlk2_within,
    scale ~ musicality + trial + session + 
      # Snare1_between + Snare2_between + 
      # Snare1_within + Snare2_within +      
      WdBlk1_between + WdBlk2_between + 
      WdBlk1_within + WdBlk2_within,
    between ~ musicality + trial + session + 
      # Snare1_between + Snare2_between + 
      # Snare1_within + Snare2_within +
      WdBlk1_between + WdBlk2_between + 
      WdBlk1_within + WdBlk2_within),
  group = subject, data = snare_data, cores=8, iter=25000, warmup=15000,
  # default: adapt_delta = 0.95 (bei lmmelsm, 0.8 stan), stepsize = 1, max_treedepth = 10
  #see http://singmann.org/hierarchical-mpt-in-stan-i-dealing-with-convergent-transitions-via-control-arguments/
  control = list(adapt_delta = 0.99, stepsize = 1, max_treedepth = 10)) 

save(fit_snareAll, file = "Results/models/snare_all_crossOnly.RData")
sink("Results/models/snare_all_crossOnly.txt")
print('iter=25000, warmup=15000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_snareAll))

sink("Results/models/snare_all_crossFreq_bonferroni.txt")
print(summary(fit_snareAll, prob=prob)) 


##### wdBlk (triple) All...#####
fit_wdBlkAll <- lmmelsm(
  list(
    observed ~ deviation,
    location ~ musicality + trial + session + 
      # WdBlk1_between + WdBlk2_between + 
      # WdBlk1_within + WdBlk2_within +
      Snare1_between + Snare2_between + 
      Snare1_within + Snare2_within,
    scale ~ musicality + trial + session + 
      # WdBlk1_between + WdBlk2_between + 
      # WdBlk1_within + WdBlk2_within + 
      Snare1_between + Snare2_between + 
      Snare1_within + Snare2_within,
    between ~ musicality + trial + session + 
      # WdBlk1_between + WdBlk2_between + 
      # WdBlk1_within + WdBlk2_within + 
      Snare1_between + Snare2_between + 
      Snare1_within + Snare2_within),
  group = subject, data = wdBlk_data, cores=8, iter=25000, warmup=15000,
  control = list(adapt_delta = 0.99, stepsize = 1, max_treedepth = 10)) 

save(fit_wdBlkAll, file = "Results/models/wdBlk_all_crossOnly.RData")
sink("Results/models/wdBlk_all_crossOnly.txt")
print('iter=25000, warmup=15000, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_wdBlkAll))

sink("Results/models/wdBlk_all_crossFreq_bonferroni.txt")
print(summary(fit_wdBlkAll, prob=prob))




##### Calculate combined wdblk and snare trials models nd store to Results/models/combined...#####
file = "Results/combined_data_silence.csv"
combined_data <- read.csv(file, sep=',')

fit_combined <- lmmelsm(
  list(
    observed ~ deviation,
    location ~ musicality + trial + session + 
      Snare1_between + Snare2_between + 
      Snare1_within + Snare2_within +
      WdBlk1_between + WdBlk2_between + 
      WdBlk1_within + WdBlk2_within,
    scale ~ musicality + trial + session + 
      Snare1_between + Snare2_between + 
      Snare1_within + Snare2_within +
      WdBlk1_between + WdBlk2_between + 
      WdBlk1_within + WdBlk2_within,
    between ~ musicality + trial + session + 
      Snare1_between + Snare2_between + 
      Snare1_within + Snare2_within +
      WdBlk1_between + WdBlk2_between + 
      WdBlk1_within + WdBlk2_within),
  group = subject, data = combined_data, cores=8, iter=30000, warmup=10000,
  # default: adapt_delta = 0.95 (bei lmmelsm, 0.8 stan), stepsize = 1, max_treedepth = 10
  #see http://singmann.org/hierarchical-mpt-in-stan-i-dealing-with-convergent-transitions-via-control-arguments/
  control = list(adapt_delta = 0.99, stepsize = 1, max_treedepth = 10)) 
save(fit_combined, file = "Results/models/combined_all30k099.RData")
sink("Results/models/combined_all30k099_bonferroni.txt")
prob_combined = 0.999 #1-0.05/48 = 0.9989583333333333, because bonferroni with 4*12 variables
print('iter=?, warmup=?, adapt_delta=0.99, stepsize = 1, max_treedepth = 10')
print(summary(fit_combined, prob=prob)) 
sink("Results/models/combined_all30k099.txt")
print(summary(fit_combined)) 




##### Other models (exclude musicality or session...)#####

fit_snare_woTSidx <- lmmelsm(
  list(
    observed ~ deviation,
    location ~ musicality +
      Snare1_between + Snare2_between + 
      Snare1_within + Snare2_within ,
    scale ~ musicality +
      Snare1_between + Snare2_between + 
      Snare1_within + Snare2_within,
    between ~ musicality + 
      Snare1_between + Snare2_between + 
      Snare1_within + Snare2_within),
  group = subject, data = snare_data, cores=8, iter=10000, warmup=5000,
  # default: adapt_delta = 0.95 (bei lmmelsm, 0.8 stan), stepsize = 1, max_treedepth = 10
  #see http://singmann.org/hierarchical-mpt-in-stan-i-dealing-with-convergent-transitions-via-control-arguments/
  control = list(adapt_delta = 0.99, stepsize = 1, max_treedepth = 10)) 
save(fit_snare_woTSidx, file = "Results/models/fit_snare_woTSidx.RData")
save(fit_snare_woTSidx, file = "Results/models/snare_woTSidx.RData")
sink("Results/models/snare_woTSidx_bonferroni.txt")
print(summary(fit_snare_woTSidx, prob=prob)) 
sink("Results/models/snare_woTSidx.txt")
print(summary(fit_snare_woTSidx)) 

fit_mini_wobetween <- lmmelsm(
  list(
    observed ~ deviation,
    location ~ musicality + trial + session + 
      Snare1_between + Snare2_between + 
      Snare1_within + Snare2_within ,
    scale ~ musicality + trial + session + 
      Snare1_between + Snare2_between + 
      Snare1_within + Snare2_within),
  group=subject, data = snare_data, cores=8, iter=100,chains=1)
c <- mcmc_parcoord_data(as.array(fit_mini_wobetween$fit))
