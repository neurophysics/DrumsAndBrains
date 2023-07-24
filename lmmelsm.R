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


###### small test for fitting absolute and not absolute at once ###### 
file = "Results/snare_data_silence.csv"
snare_data <- read.csv(file, sep=',')
chosen <- sort(sample(c(1:1324),size=100)) #sample 100 lines
small_snare_data <- snare_data[chosen, ] 

fit_test2y <- lmmelsm(
  list(
    observed ~ deviation + log(abs(deviation)),
    location ~ musicality + trial + session + 
      Snare1_between + Snare2_between + 
      Snare1_within + Snare2_within ,
    scale ~ musicality + trial + session + 
      Snare1_between + Snare2_between + 
      Snare1_within + Snare2_within,
    between ~ musicality + trial + session + 
      Snare1_between + Snare2_between + 
      Snare1_within + Snare2_within),
  group = subject, data = small_snare_data, cores=8) 

sink("/Users/carolabothe/Desktop/test.txt")
summary(fit_test2y)

##### Calculate p values #####
# nu is location intercept
# sigma is scale intercept
# mu_/logsd_beta gives wanted aprameters for location and scale model respectively
# zeta (80000,7,2) is between-group scale model
# mu_logsd_betas_random_sigma (80000,2) are random effects
get_samples <- function(lmmelsm_obj){
  # returns data frame with samples for each parameter in location and scale model
  stan_fit <- lmmelsm_obj$fit
  dat <- extract(stan_fit)
  names <- c('musicality', 'trial', 'session', 'EEG1_between', 'EEG2_between',
             'EEG1_within', 'EEG2_within') #todo: directly get names from formula
  location <- data.frame(loc=dat$mu_beta)
  colnames(location) <- sprintf("loc_%s", names)
  location <- cbind(data.frame(loc_intercept=dat$nu),location) #add loc intercept
  scale <- data.frame(loc=dat$logsd_beta)
  colnames(scale) <- sprintf("scale_%s", names)
  scale <- cbind(data.frame(scale_intercept=dat$sigma), scale) #add scale intercept
  group_loc <- data.frame(loc=dat$zeta[,,1])
  colnames(group_loc) <- sprintf("group_loc_%s", names)
  group_scale <- data.frame(loc=dat$zeta[,,2])
  colnames(group_scale) <- sprintf("group_scale_%s", names)
  random_effects <- data.frame(loc=dat$mu_logsd_betas_random_sigma)
  colnames(random_effects) <- c('RE_loc', 'RE_scale')
  
  res <- cbind(location, scale, group_loc, group_scale, random_effects)
  return(res)
}

p_value <- function(sample, name){
  # sample dim should be (80000, 1) and not contain nan
  # H0: µ=0, calculate two-tailed, i.e. p(µ≠0) 
  hist(sample, main=paste(name)) #distribution should be symmetrical - check
  p <- min(mean(sample < 0), 1-mean(sample < 0))*2 #test what percentage is below/above zero, *2 for two-tailed
  return(p)
}

### do the following for each model
# get sample data
load('Results/models/snare_all25k099.RData')
sample_df <- get_samples(fit_snareAll25k)
# calculate p_values
all_p_values <- c()
for(i in 1:ncol(sample_df)) {       # for-loop over columns
  res <- p_value(sample_df[,i], colnames(sample_df)[i])
  all_p_values <- c(all_p_values,res)
}
# check whether histograms are symmetrical!
# store together with column names
sink("Results/models/snare_all25k099_p.txt")
print_df <- data.frame(all_p_values)
rownames(print_df) <- colnames(sample_df)
print(print_df)
closeAllConnections()
#p_value(loc_int) # können Ho dass µ=0 ist nicht ablehnen, weil wahrscheinlichkeit dafür dass µ≠0'nur'67% 
# wenn man summary mit prob=1-p_loc_int nimmt, ist der eine wert beim quantil dann 0, denn ab da ist es significant


# aus p-werten dann false discovery rate berechnen: 
test_df <- read.table("Results/models/snare_all25k_p.txt")

# p-werte aufsteigend sortieren. m=#tests=32, alpha=0.05
# siehe wikipedia benjamin hochberg procedure


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
print(summary(fit_combined, prob=prob)) 
sink("Results/models/combined_all30k099.txt")
print(summary(fit_combined)) 

##### Calculate all snare models and store to Results/models/snare_...#####
file = "Results/snare_data_silence.csv"
snare_data <- read.csv(file, sep=',')

fit_snareAll25k_noMusic <- lmmelsm(
  list(
    observed ~ deviation,
    location ~ trial + session + 
      Snare1_between + Snare2_between + 
      Snare1_within + Snare2_within ,
    scale ~ trial + session + 
      Snare1_between + Snare2_between + 
      Snare1_within + Snare2_within,
    between ~ trial + session + 
      Snare1_between + Snare2_between + 
      Snare1_within + Snare2_within),
  group = subject, data = snare_data, cores=8, iter=25000, warmup=5000,
  # default: adapt_delta = 0.95 (bei lmmelsm, 0.8 stan), stepsize = 1, max_treedepth = 10
  #see http://singmann.org/hierarchical-mpt-in-stan-i-dealing-with-convergent-transitions-via-control-arguments/
  control = list(adapt_delta = 0.99, stepsize = 1, max_treedepth = 10)) 
save(fit_snareAll25k_noMusic, file = "Results/models/snare_awoMusic_25k099.RData")
sink("Results/models/snare_awoMusic_25k099_bonferroni.txt")
print(summary(fit_snareAll25k_noMusic, prob=prob)) 
sink("Results/models/snare_log2_all25k099_noMusic.txt")
print(summary(fit_snareAll25k_noMusic)) 

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

##### test univariate snare models #####
uni_lmmelms_fct <- function(x, data) lmmelsm(
  list(
    as.formula(paste("observed ~", "deviation")),
    as.formula(paste("location ~", x)),
    as.formula(paste("scale ~", x)),
    as.formula(paste("between ~", x))),
  group = subject, data = snare_data, cores = 8, iter=25000, warmup=5000,
  control = list(adapt_delta = 0.99, stepsize = 1, max_treedepth = 10))

fit_musicality <- uni_lmmelms_fct(x = "musicality", data=snare_data)
sink("Results/models/snare_musicality25k0999.txt")
print(summary(fit_musicality))
save(fit_snare_woTSidx, file = "Results/models/snare_musicality25k0999.RData")

fit_trial <- uni_lmmelms_fct(x = "trial", data=snare_data)
sink("Results/models/snare_trial099.txt")
print(summary(fit_trial))

fit_session <- uni_lmmelms_fct(x = "session", data=snare_data)
sink("Results/models/snare_session.txt")
print(summary(fit_session))

fit_Snare1_between <- uni_lmmelms_fct(x = "Snare1_between", data=snare_data)
sink("Results/models/snare_Snare1_between.txt")
print(summary(fit_Snare1_between))

fit_Snare2_between <- uni_lmmelms_fct(x = "Snare2_between", data=snare_data)
sink("Results/models/snare_Snare2_between.txt")
print(summary(fit_Snare2_between))

fit_Snare1_within <- uni_lmmelms_fct(x = "Snare1_within", data=snare_data)
sink("Results/models/snare_Snare1_within.txt")
print(summary(fit_Snare1_within))

fit_Snare2_within <- uni_lmmelms_fct(x = "Snare2_within", data=snare_data)
sink("Results/models/snare_Snare2_within099.txt")
print(summary(fit_Snare2_within))

sink() # to free memory
##### Calculate all wdBlk models and store to Results/models/wdBlk_...#####
file = "Results/wdBlk_data_silence.csv"
wdBlk_data <- read.csv(file, sep=',')

fit_wdBlkAll25k0995 <- lmmelsm(
  list(
    observed ~ deviation,
    location ~ musicality + trial + session + WdBlk1_between + WdBlk2_between + 
      WdBlk1_within + WdBlk2_within,
    scale ~ musicality + trial + session + WdBlk1_between + WdBlk2_between + 
      WdBlk1_within + WdBlk2_within,
    between ~ musicality + trial + session + WdBlk1_between + WdBlk2_between + 
      WdBlk1_within + WdBlk2_within),
  group = subject, data = wdBlk_data, cores=8, iter=25000, warmup=5000,
  control = list(adapt_delta = 0.99, stepsize = 1, max_treedepth = 10)) 

save(fit_wdBlkAll25k0995, file = "Results/models/wdBlk_all25k099_control2.RData")
sink("Results/models/wdBlk_all25k099_bonferroni_control2.txt")
print(summary(fit_wdBlkAll25k0995, prob=prob))
sink("Results/models/wdBlk_all25k099_control2.txt")
print(summary(fit_wdBlkAll25k0995))

##### test univariate wdBlk models #####
uni_lmmelms_fct <- function(x, data) lmmelsm(
  list(
    
    as.formula(paste("observed ~", "deviation")),
    as.formula(paste("location ~", x)),
    as.formula(paste("scale ~", x)),
    as.formula(paste("between ~", x))),
  group = subject, data = wdBlk_data, cores = 8, iter=25000, warmup=5000,
  control = list(adapt_delta = 0.995, stepsize = 1, max_treedepth = 12))

fit_musicality <- uni_lmmelms_fct(x = "musicality", data=wdBlk_data)
sink("Results/models/wdBlk_musicality25k099.txt")
print(summary(fit_musicality))
save(fit_musicality, file = "Results/models/wdBlk_musicality25k099.RData")

fit_trial <- uni_lmmelms_fct(x = "trial", data=wdBlk_data)
sink("Results/models/wdBlk_trial30k0999.txt")
print(summary(fit_trial))
save(fit_trial, file = "Results/models/wdBlk_trial25k0999.RData")

fit_session <- uni_lmmelms_fct(x = "session", data=wdBlk_data)
sink("Results/models/wdBlk_session.txt")
print(summary(fit_session))

fit_wdBlk1_between <- uni_lmmelms_fct(x = "WdBlk1_between", data=wdBlk_data)
sink("Results/models/wdBlk_WdBlk1_between.txt")
print(summary(fit_wdBlk1_between))

fit_wdBlk2_between <- uni_lmmelms_fct(x = "WdBlk2_between", data=wdBlk_data)
sink("Results/models/wdBlk_WdBlk2_between.txt")
print(summary(fit_wdBlk2_between))

fit_wdBlk1_within <- uni_lmmelms_fct(x = "WdBlk1_within", data=wdBlk_data)
sink("Results/models/wdBlk_WdBlk1_within25k0995TD12.txt")
print(summary(fit_wdBlk1_within))

fit_wdBlk2_within <- uni_lmmelms_fct(x = "WdBlk2_within", data=wdBlk_data)
sink("Results/models/wdBlk_WdBlk2_within.txt")
print(summary(fit_wdBlk2_within))

closeAllConnections() #closes sink so that output is back to console

# (not working) apply for all of wdBlk_data except first two (subject, deviation)
# uni_fits <- lapply(wdBlk_data[,-2:-1], uni_lmmelms_fct, data=wdBlk_data) 


##### plot divergences ##### 
# see https://mc-stan.org/bayesplot/articles/visual-mcmc-diagnostics.html?search-input=85
library(bayesplot)
library(ggplot2)
library(ggrastr) #rasterize plot to make it faster

##### snare #####
load('Results/models/snare_all25k099.RData')
np_cp <- nuts_params(fit_snareAll25k$fit) #$fit because that gives you the stan object
posterior_cp <- as.array(fit_snareAll25k$fit)

p <- mcmc_parcoord(posterior_cp, np = np_cp, regex_pars='mu', transform = scale) #regex_pars='mu', scale scales per parameter
p + theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1))# + rasterize(p, dpi=300)
pdf('Results/models/snareAll25k099_mu_scaled.pdf')
print(p)
dev.off()

p <- mcmc_parcoord(posterior_cp, np = np_cp, transform = scale) #regex_pars='mu', scale scales per parameter
p + theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1) + rasterize(p, dpi=300))
pdf('Results/models/snareAll25k099_scaled.pdf')
print(p)
dev.off()

# alternatively try
ggsave('Results...pdf', plot=p)

##### wdBlk #####
load('Results/models/wdBlk_all25k099.RData')
np_cp <- nuts_params(fit_wdBlkAll25k$fit) #$fit because that gives you the stan object
posterior_cp <- as.array(fit_wdBlkAll25k$fit)

pdf('Results/models/wdBlkAll25k099_mu_scaled.pdf')
p <- mcmc_parcoord(posterior_cp, np = np_cp, regex_pars='mu', transform = scale) #regex_pars='mu', scale scales per parameter
p + theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1))# + rasterize(p, dpi=300)
print(p)
dev.off()

pdf('Results/models/wdBlkAll25k099_scaled.pdf')
p <- mcmc_parcoord(posterior_cp, np = np_cp, transform = scale) #regex_pars='mu', scale scales per parameter
p + theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1))
print(p)
dev.off()

##### more diagnoastic plots
#divcergences vs. log-likelihood (if there are divergences in highly likely models, thats no good)
mcmc_nuts_divergence(np_cp,log_posterior(fit_snareAll$fit))

# search for funnels (pair plot for 32 parameters)
mcmc_pairs(posterior_cp, np = np_cp, pars = c("mu_beta","logsd_beta"))
