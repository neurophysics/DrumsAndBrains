###########
# loads .RData and analyses lmmelsm models by
# calculating corresponding p-values
# plotting divergences
##########

setwd('/Volumes/1TB_SSD/Arbeit/Charite/DrumsAndBrains')
prob = 0.9984375 #1-0.05/32, because bonferroni with 32 variables
library('LMMELSM')
library('rstan')

# specify model data to be used
model_name = 'singleVariable/wdBlk_WdBlk2_within'
#model_name = 'snare_all25k099'
path_RData = paste('Results/models/', model_name, '.RData', sep='')
loaded_data <-  load(path_RData)
fit_name <- get(loaded_data)

##### Calculate p values #####
get_samples <- function(lmmelsm_obj){
  # returns data frame with samples for each parameter in location and scale model
  stan_fit <- lmmelsm_obj$fit
  dat <- extract(stan_fit)
  name <- sub(".+_(\\D+).*", "\\1", model_name) #thanks chatgpt
  if (name == 'all') {
    names <- c('musicality', 'trial', 'session', 'EEG1_between', 'EEG2_between',
               'EEG1_within', 'EEG2_within')  # todo: directly get names from formula
  } else {
    names <- name
  }
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

result_file_name = paste('Results/models/', model_name, '_p.txt', sep='')
# nu is location intercept
# sigma is scale intercept
# mu_/logsd_beta gives wanted aprameters for location and scale model respectively
# zeta (80000,7,2) is between-group scale model
# mu_logsd_betas_random_sigma (80000,2) are random effects

### do the following for each model
# get data
sample_df <- get_samples(fit_name)

# calculate p_values
all_p_values <- c()
for(i in 1:ncol(sample_df)) {       # for-loop over columns which are variables
  res <- p_value(sample_df[,i], colnames(sample_df)[i])
  all_p_values <- c(all_p_values,res)
}
# check whether histograms are symmetrical!
# store together with column names
sink(result_file_name)
print_df <- data.frame(all_p_values)
rownames(print_df) <- colnames(sample_df)
print(print_df)
closeAllConnections()
print(paste('stored p values in', result_file_name, sep=' '))
#p_value(loc_int) # können Ho dass µ=0 ist nicht ablehnen, weil wahrscheinlichkeit dafür dass µ≠0'nur'67% 
# wenn man summary mit prob=1-p_loc_int nimmt, ist der eine wert beim quantil dann 0, denn ab da ist es significant

# aus p-werten dann false discovery rate berechnen: 
test_df <- read.table("Results/models/snare_all25k_p.txt")

# p-werte aufsteigend sortieren. m=#tests=32, alpha=0.05
# siehe wikipedia benjamin hochberg procedure




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


closeAllConnections() #closes sink so that output is back to console

