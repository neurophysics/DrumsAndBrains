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
model_names_sv <- c(#'singleVariable/snare_intercept',
                    #' 'singleVariable/snare_musicality0995', #3 div
                    #' #'singleVariable/snare_session',
                    #' #'singleVariable/snare_Snare1_between',
                    #' #'singleVariable/snare_Snare1_within',
                    #' #'singleVariable/snare_Snare2_between',
                    #' #'singleVariable/snare_Snare2_within',
                    #' 'singleVariable/snare_trial10k', #high rhats, what do we do with these?
                    #' #'singleVariable/wdBlk_intercept',
                    #' 'singleVariable/wdBlk_musicality', #3 div, looks fine
                    #' #'singleVariable/wdBlk_session',
                    #' 'singleVariable/wdBlk_trial_10k', #45 div
                    #' #'singleVariable/wdBlk_WdBlk1_between',
                    #' 'singleVariable/wdBlk_WdBlk1_within'#3 div, looks ok
                    #' #'singleVariable/wdBlk_WdBlk2_between',
                    #' #'singleVariable/wdBlk_WdBlk2_within',
                    # exermining integrated processing:
                    'singleVariable/wdBlk_Snare2_within',
                    'singleVariable/wdBlk_Snare2_between',
                    'singleVariable/wdBlk_Snare1_within',
                    'singleVariable/wdBlk_Snare1_between',
                    'singleVariable/snare_WdBlk2_within',
                    'singleVariable/snare_WdBlk2_between',
                    'singleVariable/snare_WdBlk1_within',
                    'singleVariable/snare_WdBlk1_between'                    
                    )
model_name = 'singleVariable/wdBlk_Snare1_within'
path_RData = paste('Results/models/', model_name, '.RData', sep='')
loaded_data <-  load(path_RData)
fit_name <- get(loaded_data)

##### plot divergences ##### 
# see https://mc-stan.org/bayesplot/articles/visual-mcmc-diagnostics.html?search-input=85
library(bayesplot)
library(ggplot2)
library(ggrastr) #rasterize plot to make it faster

# each line in the plot represents one iteration so #chains times #iterations in total lines
# x axis is parameter and y axis is its value in that iteration
#for (model_name in model_names_sv) { #intercept needs to be done seperately, because it has less variables!
  print(model_name)
  path_RData = paste('Results/models/', model_name, '.RData', sep='')
  loaded_data <-  load(path_RData)
  fit_name <- get(loaded_data)
  
  #for each iteration in each chain (e.g. 4*24000) the accepted parameter value
  # 96000 = 4 (chains) * 4000 (5000iterations-1000warmup) * 6
  #6: > unique(np_cp[,3])= accept_stat__ stepsize__    treedepth__   n_leapfrog__  divergent__   energy__  # telling which one is divergent so they can get colored
  np_cp <- nuts_params(fit_name$fit) 
  # load the actual iteration data, $fit to asses the underlying stan model:
  #extract iterations*chains*parameter array
  # e.g. 4000*4*2702 so apparently we have 2702 variables???? he confuses variables with #datapoints??
  # but model pars only has 10 variable names that make more or less sense?
  posterior_cp <- as.array(fit_name$fit)  
  
  # get 2702 variable names with 
  #x <- dimnames(posterior_cp)$parameters
  # find out which names the interesting ones have:
  #x[1:10] 
  #x[(length(x)-100):length(x)]
  #mcmc_parcoord(posterior_cp, np = np_cp, pars=c("zeta[1,1]", "zeta[1,2]", "mu_logsd_betas_random_sigma[1]", "mu_logsd_betas_random_sigma[2]", "Omega_eta[1,1]","Omega_mean_logsd[1,1]", "Omega_mean_logsd[2,2]"))
  # nu[1] and sigma[1] have one value each - intercepts for loc and scale (fits values so far)
  # eta_logsd has 1324 values = #trials - would be faster if we leave them out somewehere earlier but what are they?
  # mu_beta und logsd_beta should be loc and scale (values fit)
  # "zeta[1,1]" "zeta[1,2]" concluding from the value and position/order that its Between-group scale model mu and logsd
  # mu_random has 20 values (1 for each subject)
  # mlogsd_random has 20 values (1 for each subject)
  # "mu_logsd_betas_random_sigma[1]" and "mu_logsd_betas_random_sigma[2]" concluding from the value and position/order that its RE mu and logsd
  # "Omega_eta[1,1]"                
  # "Omega_mean_logsd[1,1]" to [2,2] table on the bottom for RE correlations 
  #"lp__" 
  
  internal_var_names = c('nu[1]', 'sigma[1]', # intercept loc, intercept scale
                         'mu_beta[1,1]', 'logsd_beta[1,1]', #variable (e.g. musicality) loc, variable scale
                         'zeta[1,1]', 'zeta[1,2]', #between-group loc and scale
                         'mu_logsd_betas_random_sigma[1]', 'mu_logsd_betas_random_sigma[2]') #RE for loc and scale
  var_names = c('intercept_loc', 'intercept_scale', 'variable_loc','variable_scale','between-group_loc','between-group_scale','RE_loc','RE_scale')
  
  # without scale to compare with actual values
  p <- mcmc_parcoord(posterior_cp, 
                     np = np_cp, 
                     pars=internal_var_names) 
  p + theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1))# + rasterize(p, dpi=300)
  pdf(paste('Results/models/', model_name, '.pdf', sep=''))
  print(p + 
          scale_x_discrete(labels=var_names) + 
          theme(axis.text.x = element_text(angle=45, vjust=1.0, hjust=1)))
  dev.off()
  
  # and one to see divergence issues
  p_scaled <- mcmc_parcoord(posterior_cp, 
                            np = np_cp,
                            pars= internal_var_names,
                            transform = scale) #scale scales per parameter
  # can be further customized with ggplot2 using mcmc_parcoord_data
  pdf(paste('Results/models/', model_name, '_scaled.pdf', sep=''))
  print(p_scaled + 
          scale_x_discrete(labels=var_names) + 
          theme(axis.text.x = element_text(angle=45, vjust=1.0, hjust=1)) # rasterize(p, dpi=300)
  )
  dev.off()


# PLOT FOR INTERESTING VALUE ONE LINE PER CHAIN PVER ITERATIONS AND see if it starts jumping:
#traceplot(fit_name$fit,pars=c('mu_beta[1,1]')) #variable location
#traceplot(fit_name$fit,pars=c('zeta[1,1]')) #between group loc

color_scheme_set("mix-brightblue-gray")

file_name <- paste('Results/models/', model_name, '_traceplot_varloc.pdf', sep='')
trace_plot <- mcmc_trace(posterior_cp, pars = 'mu_beta[1,1]', np = np_cp) + 
  xlab("Post-warmup iteration")
ggsave(file_name, trace_plot, width = 12, height = 3) 

file_name <- paste('Results/models/', model_name, '_traceplot_bwgrouploc.pdf', sep='')
trace_plot <- mcmc_trace(posterior_cp, pars = 'zeta[1,1]', np = np_cp) + 
  xlab("Post-warmup iteration")
ggsave(file_name, trace_plot, width = 12, height = 3) 
# see https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/Diagnosing_biased_Inference_with_Divergences.html

##### more diagnoastic plots##### 
#divcergences vs. log-likelihood (if there are divergences in highly likely models, thats no good)
mcmc_nuts_divergence(np_cp,log_posterior(fit_snareAll$fit))

# search for funnels (pair plot for 32 parameters)
mcmc_pairs(posterior_cp, np = np_cp, pars = c("mu_beta","logsd_beta"))

closeAllConnections() #closes sink so that output is back to console

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
  
  # intercept - location is nu, scale is sigma
  loc_intercept <- data.frame(loc_intercept=dat$nu)
  scale_intercept <- data.frame(scale_intercept=dat$sigma)
  
  # Random Effects - mu_logsd_betas_random_sigma (80000,2) 
  random_effects <- data.frame(loc=dat$mu_logsd_betas_random_sigma)
  colnames(random_effects) <- c('RE_loc', 'RE_scale')
  if (names == 'intercept'){ #other format
    res <- cbind(loc_intercept, scale_intercept, random_effects)
  }else{
    # Parameters -  mu_/logsd_beta for location/scale
    location <- data.frame(loc=dat$mu_beta)
    colnames(location) <- sprintf("loc_%s", names)
    scale <- data.frame(loc=dat$logsd_beta)
    colnames(scale) <- sprintf("scale_%s", names)
    
    #Between-group scale model - zeta (80000,7,2)
    group_loc <- data.frame(loc=dat$zeta[,,1])
    colnames(group_loc) <- sprintf("group_loc_%s", names)
    group_scale <- data.frame(loc=dat$zeta[,,2])
    colnames(group_scale) <- sprintf("group_scale_%s", names)
    res <- cbind(loc_intercept, location, scale_intercept, scale, group_loc, group_scale, random_effects)
  }
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
# get data
for (model_name in model_names_sv) {
  path_RData = paste('Results/models/', model_name, '.RData', sep='')
  loaded_data <-  load(path_RData)
  fit_name <- get(loaded_data)
  result_file_name = paste('Results/models/', model_name, '_p.txt', sep='')
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
  }
#p_value(loc_int) # können Ho dass µ=0 ist nicht ablehnen, weil wahrscheinlichkeit dafür dass µ≠0'nur'67% 
# wenn man summary mit prob=1-p_loc_int nimmt, ist der eine wert beim quantil dann 0, denn ab da ist es significant


# aus p-werten dann false discovery rate berechnen: 
test_df <- read.table("Results/models/snare_all25k_p.txt")

# p-werte aufsteigend sortieren. m=#tests=32, alpha=0.05
# siehe wikipedia benjamin hochberg procedure





closeAllConnections() #closes sink so that output is back to console
