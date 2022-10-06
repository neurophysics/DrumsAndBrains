### LMMELSM: calculates REWB model with scale and variance
# see Devlaminck et al. 2011 - Multisubject learning for 
# common spatial patterns in motor-imagery BCI
# setting cores higher than numbe rof chains makes no sense
# 
setwd('/Volumes/1TB_SSD/Arbeit/Charite/DrumsAndBrains')
prob = 0.9984375 #1-0.05/32, because bonferroni with 32 variables
library('LMMELSM')

##### Calculate all snare models and store to Results/models/snare_...#####
file = "Results/snare_data_silence.csv"
snare_data <- read.csv(file, sep=',')

fit_snareAll25k <- lmmelsm(
	       list(
	observed ~ deviation,
	location ~ musicality + trial + session + 
	  Snare1_between + Snare2_between + 
	  Snare1_within + Snare2_within ,
	scale ~ musicality + trial + session + 
	  Snare1_between + Snare2_between + 
	  Snare1_within + Snare2_within,
	between ~ musicality + trial + session + 
	  Snare1_between + Snare2_between + 
	  Snare1_within + Snare2_within),
	group = subject, data = snare_data, cores=8, iter=25000, warmup=5000,
  # default: adapt_delta = 0.95 (bei lmmelsm, 0.8 stan), stepsize = 1, max_treedepth = 10
  #see http://singmann.org/hierarchical-mpt-in-stan-i-dealing-with-convergent-transitions-via-control-arguments/
  control = list(adapt_delta = 0.95, stepsize = 1, max_treedepth = 10)) 
save(fit_snareAll25k, file = "Results/models/snare_all25k.RData")
sink("Results/models/snare_all25k_bonferroni.txt")
print(summary(fit_snareAll25k), prob=prob) 
sink("Results/models/snare_all25k.txt")
print(summary(fit_snareAll25k)) 

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
              group = subject, data = data, cores = 8, iter=10000)

fit_musicality <- uni_lmmelms_fct(x = "musicality", data=snare_data)
sink("Results/models/snare_musicality.txt")
print(summary(fit_musicality))

fit_trial <- uni_lmmelms_fct(x = "trial", data=snare_data)
sink("Results/models/snare_trial.txt")
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
sink("Results/models/snare_Snare2_within.txt")
print(summary(fit_Snare2_within))

sink() # to free memory
##### Calculate all wdBlk models and store to Results/models/wdBlk_...#####
file = "Results/wdBlk_data_silence.csv"
wdBlk_data <- read.csv(file, sep=',')

fit_wdBlkAll25k <- lmmelsm(
  list(
    observed ~ deviation,
    location ~ musicality + trial + session + WdBlk1_between + WdBlk2_between + 
       WdBlk1_within + WdBlk2_within,
    scale ~ musicality + trial + session + WdBlk1_between + WdBlk2_between + 
       WdBlk1_within + WdBlk2_within,
    between ~ musicality + trial + session + WdBlk1_between + WdBlk2_between + 
       WdBlk1_within + WdBlk2_within),
  group = subject, data = wdBlk_data, cores=8, iter=25000,
  control = list(adapt_delta = 0.99, stepsize = 1, max_treedepth = 10)) 

save(fit_wdBlkAll25k, file = "Results/models/wdBlk_all25k099.RData")
sink("Results/models/wdBlk_all25k099_bonferroni.txt")
print(summary(fit_wdBlkAll25k, prob=prob))
sink("Results/models/wdBlk_all25k099.txt")
print(summary(fit_wdBlkAll25k))

##### test univariate wdBlk models #####
uni_lmmelms_fct <- function(x, data) lmmelsm(
  list(
    
    as.formula(paste("observed ~", "deviation")),
    as.formula(paste("location ~", x)),
    as.formula(paste("scale ~", x)),
    as.formula(paste("between ~", x))),
  group = subject, data = data, cores = 8, iter=10000)

fit_musicality <- uni_lmmelms_fct(x = "musicality", data=wdBlk_data)
sink("Results/models/wdBlk_musicality.txt")
print(summary(fit_musicality))

fit_trial <- uni_lmmelms_fct(x = "trial", data=wdBlk_data)
sink("Results/models/wdBlk_trial.txt")
print(summary(fit_trial))

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
sink("Results/models/wdBlk_WdBlk1_within.txt")
print(summary(fit_wdBlk1_within))

fit_wdBlk2_within <- uni_lmmelms_fct(x = "WdBlk2_within", data=wdBlk_data)
sink("Results/models/wdBlk_WdBlk2_within.txt")
print(summary(fit_wdBlk2_within))

# (not working) apply for all of wdBlk_data except first two (subject, deviation)
# uni_fits <- lapply(wdBlk_data[,-2:-1], uni_lmmelms_fct, data=wdBlk_data) 


##### plot divergences ##### 
# see https://mc-stan.org/bayesplot/articles/visual-mcmc-diagnostics.html?search-input=85
library(bayesplot)
library(ggplot2)
library(ggrastr) #rasterize plot to make it faster

##### snare #####
load('Results/models/snare_all25k.RData')
np_cp <- nuts_params(fit_snareAll25k$fit) #$fit because that gives you the stan object
posterior_cp <- as.array(fit_snareAll25k$fit)

p <- mcmc_parcoord(posterior_cp, np = np_cp, regex_pars='mu', transform = scale) #regex_pars='mu', scale scales per parameter
p + theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1))# + rasterize(p, dpi=300)
pdf('Results/models/snareAll25k_mu_scaled.pdf')
print(p)
dev.off()

p <- mcmc_parcoord(posterior_cp, np = np_cp, transform = scale) #regex_pars='mu', scale scales per parameter
p + theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1) + rasterize(p, dpi=300))
pdf('Results/models/snareAll25k_scaled.pdf')
print(p)
dev.off()

# alternatively try
ggsave('Results...pdf', plot=p)

##### wdBlk #####
load('Results/models/wdBlk_all25k.RData')
np_cp <- nuts_params(fit_wdBlkAll25k$fit) #$fit because that gives you the stan object
posterior_cp <- as.array(fit_wdBlkAll25k$fit)

pdf('Results/models/wdBlkAll25k_mu_scaled.pdf')
p <- mcmc_parcoord(posterior_cp, np = np_cp, regex_pars='mu', transform = scale) #regex_pars='mu', scale scales per parameter
p + theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1))# + rasterize(p, dpi=300)
print(p)
dev.off()

pdf('Results/models/wdBlkAll25k_scaled.pdf')
p <- mcmc_parcoord(posterior_cp, np = np_cp, transform = scale) #regex_pars='mu', scale scales per parameter
p + theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1))
print(p)
dev.off()

##### more diagnoastic plots
#divcergences vs. log-likelihood (if there are divergences in highly likely models, thats no good)
mcmc_nuts_divergence(np_cp,log_posterior(fit_snareAll$fit))

# search for funnels (pair plot for 32 parameters)
mcmc_pairs(posterior_cp, np = np_cp, pars = c("mu_beta","logsd_beta"))
           