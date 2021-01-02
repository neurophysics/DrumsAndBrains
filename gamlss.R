# install.packages('gamlss')
# install.packages('reticulate')
# install.packages('abind)

# set constants
result_folder = '/Volumes/1TB_SSD/Arbeit/Charite/DrumsAndBrains/Results'
data_folder = '/Volumes/1TB_SSD/Arbeit/Charite/DrumsAndBrains/Data'
N_subjects = 20
N_comp_ssd = 2 #first two
N_comp_freq = 2 #snare and wdblk


##### read our data #####
# use python in R
library(reticulate) 
np <- import("numpy")

# read F_SSD, the SSD for each subject and sort by subject
## get freq bins und snare/wdBlk freq indices (same for all subjects)
f <- np$load(file.path(result_folder, 'S01', 'prepared_FFTSSD.npz')) [['f']] 
i_sn_freq <- which(abs(f-7/6)<0.000001)
i_wb_freq <- which(abs(f-7/4)<0.000001)
## load file and initialize F_SSD list
F_SSD_file <- np$load(file.path(result_folder, 'F_SSD.npz')) 
F_SSD <- vector("list", length(N_subjects))
N_trials <- c()
## loop over subjects, sort and store needed F_SSD parts in list
for (subject in 1:N_subjects){
  arr <- F_SSD_file$f[[sprintf('arr_%d', subject-1)]]
  s <- abs(arr[1,1,1,1])+1 #4th dim is coded to give subject/sort number, python indices start with 0
  arr <- drop(arr) #deletes 4th dimension
  arr <- arr[1:2,c(i_sn_freq,i_wb_freq),] #only need first two SSD components and two relevant frequencies
  F_SSD[[s]] <- abs(arr) #eeg is complex
  #print(sum(is.nan(F_SSD[[s]])))
  #print(is.numeric(F_SSD[[s]]))
  N_trials <- c(N_trials, dim(arr)[3])
}
# calculate mean and sd over all subjects
## concatenate to one big array
library(abind)
F_SSD_concat <- abind(F_SSD, along = 3)
#F_SSD_concat <- array(F_SSD, dim=c(2,2,sum(N_trials)))
F_SSD_mean <- apply(F_SSD_concat, c(1,2), mean)
F_SSD_sd <- apply(F_SSD_concat, c(1,2), mean)

# divide F_SSD into conditions and read behavioral data
F_SSD_snare <- vector("list", length(N_subjects))
F_SSD_wdBlk <- vector("list", length(N_subjects))
N_trials_wb <- c() #stores number of valid wdBlk trials for each subject
N_trials_sn <- c() #stores number of valid snare trials for each subject
trial_id <- c(1:75)
for (subject in 1:(N_subjects+1)){ # +1 for originally we had 21 subjects
  ## get right subject index (no eeg data für subject 11) => use i_subject to assess new data structure but subject to load data
  if (subject==11) next 
  else if (subject<11) i_subject <- subject
  else i_subject <- subject-1 
  
  ## get valid EEG trials in listening window (not noisy)
  inlier_file <- np$load(file.path(result_folder, sprintf('S%02d', subject), 'eeg_results.npz'))
  snareInlier <- inlier_file$f[['snareInlier']]
  wdBlkInlier <- inlier_file$f[['wdBlkInlier']]

  ### read, calculate z-score and divide F_SSD into snare and woodblock,reject invalid trials
  F_SSD_subj <- sweep(sweep(F_SSD[[i_subject]], MARGIN=c(1,2), F_SSD_mean, FUN="-"), MARGIN=c(1,2), F_SSD_sd, FUN="/")
  behavior_file <- np$load(file.path(result_folder, sprintf('S%02d', subject), 'behavioural_results.npz'),
                           allow_pickle = T, encoding='latin1') 
  snareCue_times <- behavior_file$f[['snareCue_times']]
  snareCue_times <- snareCue_times[snareInlier]
  wdBlkCue_times <- behavior_file$f[['wdBlkCue_times']]
  wdBlkCue_times <- wdBlkCue_times[wdBlkInlier]
  type_df <- data.frame(rep(FALSE,length(snareCue_times)),snareCue_times)
  names(type_df) <- c('type_index','cueTime')
  type_df2 <- data.frame(rep(TRUE,length(wdBlkCue_times)), wdBlkCue_times)
  names(type_df2) <- c('type_index','cueTime')
  type_df <- rbind(type_df, type_df2)
  trial_type <- type_df[order(type_df$cueTime),]$type_index
  F_SSD_wdBlk[[i_subject]] <- F_SSD_subj[,,trial_type]
  F_SSD_snare[[i_subject]] <- F_SSD_subj[,,!trial_type]
  
  ## read behavioral data
  snareDev <- behavior_file$f[["snare_deviation"]] #deviation
  #snareDev <- snareDev[snareInlier]
  wdBlkDev <- behavior_file$f[["wdBlk_deviation"]]
  #wdBlkDev <- wdBlkDev[wdBlkInlier]
  wdBlkDevToClock <- behavior_file$f[["wdBlkCue_DevToClock"]] #to get sessions
  snareDevToClock <- behavior_file$f[["snareCue_DevToClock"]]

  ### flatten list and use session index instead
  N_sessions <- length(snareDevToClock)
  snare_session <- c()
  wdBlk_session <- c()
  for (i in 1:N_sessions){
    snare_session <- c(snare_session, rep(i,length(snareDevToClock[[i]])))
    wdBlk_session <- c(wdBlk_session, rep(i,length(wdBlkDevToClock[[i]])))
  }
  
  ### combine each to data frame
  snare_df <- data.frame(rep(subject, length(snareDev)), snare_session, trial_id, snareDev)
  names(snare_df) <- c('subject_ID','session','trial', 'dev')
  wdBlk_df <- data.frame(rep(subject, length(wdBlkDev)), wdBlk_session, trial_id, wdBlkDev)
  names(wdBlk_df) <- c('subject_ID','session','trial', 'dev')
  
  ### reject invalid eeg trials 
  snare_df <- snare_df[snareInlier,]
  wdBlk_df <- wdBlk_df[wdBlkInlier,]
  
  ### reject outlier by taking range median ± 1.5*IQR 
  lb <- median(snare_df$dev, na.rm=T) - 1.5*IQR(snare_df$dev, na.rm=T)
  ub <- median(snare_df$dev, na.rm=T) + 1.5*IQR(snare_df$dev, na.rm=T)
  ind_sn <- which(snare_df$dev>lb & snare_df$dev<ub)
  snare_df <- snare_df[ind_sn,]
  F_SSD_snare[[i_subject]] <- F_SSD_snare[[i_subject]][,,ind_sn]
  N_trials_sn <- c(N_trials_sn, length(ind_sn))
  
  lb <- median(wdBlk_df$dev, na.rm=T) - 1.5*IQR(wdBlk_df$dev, na.rm=T)
  ub <- median(wdBlk_df$dev, na.rm=T) + 1.5*IQR(wdBlk_df$dev, na.rm=T)
  ind_wb <- which(wdBlk_df$dev>lb & wdBlk_df$dev<ub)
  wdBlk_df <- wdBlk_df[ind_wb,]
  F_SSD_wdBlk[[i_subject]] <- F_SSD_wdBlk[[i_subject]][,,ind_wb]
  N_trials_wb <- c(N_trials_wb, length(ind_wb))
  
  ### combine to single data frame with index 0 for snare and 1 for wdBlk (type_index)
  type_index <- c(rep(0,length(snare_df[,1])),rep(1,length(wdBlk_df[,2])))
  behavior_subj <- data.frame(type_index, rbind(snare_df, wdBlk_df))
  behavior_subj$dev <- behavior_subj$dev - mean(behavior_subj$dev, na.rm=TRUE) # normalize subjects to have 0 mean each
  if (i_subject==1) behavior_df <- behavior_subj
  else behavior_df <- rbind(behavior_df, behavior_subj)
  
  # check dimensions
  #print(length(ind_wb))
  #print(length(wdBlk_df$subject_ID==i_subject))
  #print(dim(F_SSD_wdBlk[[i_subject]]))
}

# split into conditions
behavior_df_snare <- split(behavior_df, behavior_df$type_index)[[1]]
behavior_df_wdBlk <- split(behavior_df, behavior_df$type_index)[[2]]

# read additional subject info (handedness and musical score)
addInfo <- read.csv(file.path(data_folder,'additionalSubjectInfo.csv'), sep=';')
music_total_score <- addInfo$MusicQualification + addInfo$MusicianshipLevel + addInfo$TrainingYears
music_z_score <- (music_total_score - mean(music_total_score)) / sd(music_total_score)
music_z_score <- c(music_z_score[1:10], music_z_score[12:21]) #del subject 11 for we dont have corresponding eeg data

# remove unused objects
rm('F_SSD_file','eeg_file','snareInlier','wdBlkInlier','F_SSD_subj','file','snareCue_times','wdBlkCue_times',
   'type_df','type_df2','trial_type','behavior','snareDev','wdBlkDev','wdBlkDevToClock',
   'snareDevToClock','snare_df', 'wdBlk_df','behavior_subj','addInfo', 'music_total_score')

##### check normality of response variable #####

# hist and qq plot for first 4 subjects
for (i in 1:4){
  par(mfrow=c(2,2))
  hist(behavior_df_snare[behavior_df_snare$subject_ID==i,4], breaks=20, main=sprintf('Subject %d, Snare', i), xlab='Deviation from Cue') 
  hist(behavior_df_wdBlk[behavior_df_wdBlk$subject_ID==i,4], breaks=20, main=sprintf('Subject %d, WdBlk', i), xlab='Deviation from Cue') 
  qqnorm(behavior_df_snare[behavior_df_snare$subject_ID==i,4], main=sprintf('Subject %d, Snare', i))
  qqline(behavior_df_snare[behavior_df_snare$subject_ID==i,4])
  qqnorm(behavior_df_wdBlk[behavior_df_wdBlk$subject_ID==i,4], main=sprintf('Subject %d, WdBlk', i))
  qqline(behavior_df_wdBlk[behavior_df_wdBlk$subject_ID==i,4])
  #mtext(sprintf('Subject %d', i), side = 3, line = -1, outer = TRUE)
}
# hist and qq plot for all subjects, save in Results folder
pdf(file=file.path(result_folder,'gamlss_NormalityDev.pdf'))
par(mfrow=c(2,2))
hist(behavior_df_snare$dev, breaks=20, main='All Subjects Snare', xlab='Deviation from Cue')
hist(behavior_df_wdBlk$dev, breaks=20, main='All Subjects WdBlk', xlab='Deviation from Cue')
qqnorm(behavior_df_snare$dev, main='All Subjects Snare')
qqline(behavior_df_snare$dev)
qqnorm(behavior_df_wdBlk$dev, main='All Subjects WdBlk')
qqline(behavior_df_wdBlk$dev)
dev.off()
# can be left like this because we have RE on variance

# Kolmogorov-Smirnov Test
data <- behavior_df_snare[behavior_df_snare$subject_ID==1,4]
ks.test(data, y='pnorm', mean(data, na.rm=T), sd(data, na.rm=T), alternative = 'two.sided')
ks.test(behavior_df_wdBlk$dev, y='pnorm', mean(behavior_df_wdBlk$dev, na.rm=T), sd(behavior_df_wdBlk$dev, na.rm=T), alternative = 'two.sided')
## single subjects seem to be normally dist, but not all together

##### create design matrix for snare #####
intercept <- rep(1,sum(N_trials_sn))
beta1 <- matrix(nrow=N_comp_ssd*N_comp_freq, ncol=sum(N_trials_sn))
beta2 <- matrix(nrow=N_comp_ssd*N_comp_freq, ncol=sum(N_trials_sn))
beta3 <- c()
ypsilon0 <- matrix(nrow=N_subjects, ncol=sum(N_trials_sn))
ypsilon1 <- matrix(0L, nrow=N_subjects*N_comp_ssd*N_comp_freq, ncol=sum(N_trials_sn))
'TODO: 1,2,3,4 hardcoden in abhängigkeit von N_comp_ssd (aufdröseln in N_freq und N_SSD?'
trial_index <- c(0,cumsum(N_trials_sn))
for (i in 1:N_subjects) {
  ## calculate needed indices
  i1 <- 1
  i2 <- N_comp_ssd
  i3 <- 1+N_comp_freq
  i4 <- N_comp_ssd*N_comp_freq
  ## beta1 contains the 4 (2 SSDs for 2 freq) trial averaged components for every subject
  snare_Tmean <- apply(F_SSD_snare[[i]], c(1,2), mean) #trials are 3rd dim (keep first two)
  beta1[i1:i2, (trial_index[i]+1):trial_index[i+1]] <- F_SSD_snare[[i]][,1,] - snare_Tmean[,1] #snare freq
  beta1[i3:i4, (trial_index[i]+1):trial_index[i+1]] <- F_SSD_snare[[i]][,2,] - snare_Tmean[,2] #wdblk freq
  
  ## beta2 contains the 4 (2 SSDs for 2 freq) trial averages for every subject (repeated N_trials times)
  beta2[i1:i2, (trial_index[i]+1):trial_index[i+1]] <- rep(snare_Tmean[,1], N_trials_sn[i])
  beta2[i3:i4, (trial_index[i]+1):trial_index[i+1]] <- rep(snare_Tmean[,2], N_trials_sn[i])
  
  ## beta3 contains the musical z_score of each proband repeated N_trials_sn times
  beta3 <- c(beta3, rep(music_z_score[i], N_trials_sn[i]))

  ## ypsilon0 contains a 1 if subject row and column are the same, 0 else
  yps0 <- rep(0, N_subjects)
  yps0[i] <- 1 # 1 at subject column
  ypsilon0[i,] <- rep(yps0, N_trials_sn) #repeat 1 for number of trials in each row
  
  ## ypsilon1 contains x_it-mean(x_i) if subject row and column are the same, 0 else
  ypsilon1[(1+(i-1)*i4):(2+(i-1)*i4), (trial_index[i]+1):trial_index[i+1]] <- F_SSD_snare[[i]][,1,] - snare_Tmean[,1]
  ypsilon1[(3+(i-1)*i4):(4+(i-1)*i4), (trial_index[i]+1):trial_index[i+1]] <- F_SSD_snare[[i]][,2,] - snare_Tmean[,2]
}
design_mat_snare <- rbind(intercept, beta1, beta2, beta3, ypsilon0, ypsilon1, behavior_df_snare$session, behavior_df_snare$trial)
snare_data <- data.frame(t(rbind(behavior_df_snare$dev, beta1, beta2, beta3, ypsilon0, ypsilon1, behavior_df_snare$session, behavior_df_snare$trial))) #dont need intercept for its added automatically
dim(design_mat_snare)

##### create design matrix for wdBlk #####
intercept <- rep(1,sum(N_trials_wb))
beta1 <- matrix(nrow=N_comp_ssd*N_comp_freq, ncol=sum(N_trials_wb))
beta2 <- matrix(nrow=N_comp_ssd*N_comp_freq, ncol=sum(N_trials_wb))
beta3 <- c()
ypsilon0 <- matrix(nrow=N_subjects, ncol=sum(N_trials_wb))
ypsilon1 <- matrix(0L, nrow=N_subjects*N_comp_ssd*N_comp_freq, ncol=sum(N_trials_wb))
'TODO: 1,2,3,4 hardcoden in abhängigkeit von N_comp_ssd (aufdröseln in N_freq und N_SSD?'
trial_index <- c(0,cumsum(N_trials_wb))
for (i in 1:N_subjects) {
  ## calculate needed indices
  i1 <- 1
  i2 <- N_comp_ssd
  i3 <- 1+N_comp_freq
  i4 <- N_comp_ssd*N_comp_freq
  ## beta1 contains the 4 (2 SSDs for 2 freq) trial averaged components for every subject
  wdBlk_Tmean <- apply(F_SSD_wdBlk[[i]], c(1,2), mean) #trials are 3rd dim (keep first two)
  beta1[i1:i2, (trial_index[i]+1):trial_index[i+1]] <- F_SSD_wdBlk[[i]][,1,] - wdBlk_Tmean[,1] #wdBlk freq
  beta1[i3:i4, (trial_index[i]+1):trial_index[i+1]] <- F_SSD_wdBlk[[i]][,2,] - wdBlk_Tmean[,2] #wdblk freq
  
  ## beta2 contains the 4 (2 SSDs for 2 freq) trial averages for every subject (repeated N_trials times)
  beta2[i1:i2, (trial_index[i]+1):trial_index[i+1]] <- rep(wdBlk_Tmean[,1], N_trials_wb[i])
  beta2[i3:i4, (trial_index[i]+1):trial_index[i+1]] <- rep(wdBlk_Tmean[,2], N_trials_wb[i])
  
  ## beta3 contains the musical z_score of each proband repeated N_trials_wb times
  beta3 <- c(beta3, rep(music_z_score[i], N_trials_wb[i]))
  
  ## ypsilon0 contains a 1 if subject row and column are the same, 0 else
  yps0 <- rep(0, N_subjects)
  yps0[i] <- 1 # 1 at subject column
  ypsilon0[i,] <- rep(yps0, N_trials_wb) #repeat 1 for number of trials in each row
  
  ## ypsilon1 contains x_it-mean(x_i) if subject row and column are the same, 0 else
  ypsilon1[(1+(i-1)*i4):(2+(i-1)*i4), (trial_index[i]+1):trial_index[i+1]] <- F_SSD_wdBlk[[i]][,1,] - wdBlk_Tmean[,1]
  ypsilon1[(3+(i-1)*i4):(4+(i-1)*i4), (trial_index[i]+1):trial_index[i+1]] <- F_SSD_wdBlk[[i]][,2,] - wdBlk_Tmean[,2]
}
design_mat_wdBlk <- rbind(intercept, beta1, beta2, beta3, ypsilon0, ypsilon1, behavior_df_wdBlk$session, behavior_df_wdBlk$trial)
wdBlk_data <- data.frame(t(rbind(behavior_df_wdBlk$dev, beta1, beta2, beta3, ypsilon0, ypsilon1, behavior_df_wdBlk$session, behavior_df_wdBlk$trial))) #dont need intercept for its added automatically
dim(design_mat_wdBlk)


##### OLS ######
formula_snare <- as.formula(snare_data)
ols_snare <- lm(formula_snare, data = snare_data)
summary(ols_snare) #NA? sum(is.na.data.frame(snare_data))=0, 
# intercept, beta2, session and trial index some v0 and one v1 significant
# beta1 and beta3 not significant
# NA for S15-20 in v0 S2 and S20 in v1 (14 not defined because of singularities)
# alias(ols_snare) shows dependencies
# intercept: subjects are 0.02 s too early

# wenn abs(dev): negative komponente heißt besser (zweite ssd bei snare größer => reduziert abweichung)
# also, around 40% of variance explained
p <- predict(ols_snare)
cor(p,snare_data$V1) #0.46

formula_wdBlk <- as.formula(wdBlk_data)
ols_wdBlk <- lm(formula_wdBlk, data = wdBlk_data, na.action=na.exclude)
summary(ols_wdBlk)
# Intercept, beta2, beta3, v0 (except NA and S10), session and trial index significant
# beta1 and v1 not significant
# NA for S15-20 in v0, S2 and S20 in v1
p <- predict(ols_wdBlk)
cor(p,wdBlk_data$V1) #0.38


##### TODO #####
#3 trial index/session index ergänzen? bräuchten noch FE und RE (könnte systematischen einfluss haben der unterschiedlich stark in jedem probandinnen)
#4 gamlss (wollen mean und variance fitten von dev)
#5 gamlss und regression vergleiche, das gleiche nochmal für wbre und wdblk zusammen 
# (wenn unterschied nicht groß vllt nur regression)


##### notes #####
# snare_deviation contains nan which are not corresponding to Inlier... keep for now, take care with mean!
# explanatory variables: type_index, subject_ID, music_z_score, addInfo$LQ

#R cheatsheet
# first array slice: a[1, , ]
# get positions: snare_index <- which(trial_type %in% 0)
#  TypeError: 'BagObj' object is not subscriptable => falsche indizierung i.e. [[]] insteaf od [] (or set pickle to true)
# NaN operations: add na.remove=T

##### GAMLSS #####
library(gamlss)
# example
data(abdom)
dim(abdom)
mod<-gamlss(y~pb(x),sigma.fo=~pb(x),family=BCT, data=abdom, method=mixed(1,20))
plot(mod)
summary(mod)
rm(mod)

# snare
y = behavior_df_snare$dev
mu = mean(y, na.rm = T)
sigma = sd(y, na.rm = T)
family_obj = dNO(x=y, mu=mu, sigma=sigma)
mod_snare<-gamlss(formula_snare, sigma.fo=formula_snare, family=NO, data=snare_data, method=mixed(1,20))
# family: d, p, q and r functions for density (pdf), distribution function (cdf), quantile function and random generation function
# todo: change family to family_obj (error: invalid object argument), NO length 26, family_obj 1259
# what are the link functions?
plot(mod_snare)
summary(mod_snare)

# wdBlk
y = behavior_df_wdBlk$dev
mod_wdBlk<-gamlss(formula_wdBlk, sigma.fo=formula_wdBlk, family=NO, data=wdBlk_data, method=mixed(1,100))
plot(mod_wdBlk)
summary(mod_wdBlk)
