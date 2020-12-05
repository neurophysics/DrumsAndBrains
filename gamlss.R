# install.packages('gamlss')
# install.packages('reticulate')

library(gamlss)
# example
data(abdom)
dim(abdom)
mod<-gamlss(y~pb(x),sigma.fo=~pb(x),family=BCT, data=abdom, method=mixed(1,20))
plot(mod)
summary(mod)
rm(mod)

# set constants
result_folder = '/Volumes/1TB_SSD/Arbeit/Charite/DrumsAndBrains/Results'
data_folder = '/Volumes/1TB_SSD/Arbeit/Charite/DrumsAndBrains/Data'
N_subjects = 2
N_trials = 75
N_comp = 4


# read our data
library(reticulate) # use python in R
np <- import("numpy")

## read y (absolute latency)
for (subject in 1:N_subjects) {
  if (subject==11) next #no data für subject 11
  behavior <- np$load(file.path(result_folder, sprintf('S%02d', subject), 'behavioural_results.npz'), 
                      encoding='latin1', allow_pickle=TRUE)
  snareDev <- behavior$f[["snareCue_DevToClock"]] 
  wdBlkDev <- behavior$f[['wdBlkCue_DevToClock']]
  
  ### flatten list and use session index instead
  N_sessions <- length(snareDev)
  snare_session <- c()
  wdBlk_session <- c()
  for (i in 1:N_sessions){
    snare_session <- c(snare_session, rep(i,length(snareDev[[i]])))
    wdBlk_session <- c(wdBlk_session, rep(i,length(wdBlkDev[[i]])))
  }
  ### combine each to data frame
  snareDev <- unlist(snareDev)
  wdBlkDev <- unlist(wdBlkDev)
  snare_df <- data.frame(rep(subject, length(snareDev)), snare_session, snareDev)
  names(snare_df) <- c('subject_ID','session','dev')
  wdBlk_df <- data.frame(rep(subject, length(wdBlkDev)), snare_session, wdBlkDev)
  names(wdBlk_df) <- c('subject_ID','session','dev')
  
  ### combine to single data frame with index 0 for snare and 1 for wdBlk (type_index)
  type_index <- c(rep(0,length(snare_df[,1])),rep(1,length(wdBlk_df[,2])))
  behavior_subj <- data.frame(type_index, rbind(snare_df, wdBlk_df))
  behavior_subj$dev <- behavior_subj$dev - mean(behavior_subj$dev) # normalize subjects to have 0 mean each
  if (subject==1) behavior_df <- behavior_subj
  else behavior_df <- rbind(behavior_df, behavior_subj)
}

## read additional subject info (handedness and musical score)
addInfo <- read.csv(file.path(data_folder,'additionalSubjectInfo.csv'), sep=';')
music_total_score <- addInfo$MusicQualification + addInfo$MusicianshipLevel + addInfo$TrainingYears
music_z_score <- (music_total_score - mean(music_total_score)) / sd(music_total_score)
music_z_score <- c(music_z_score[1:10], music_z_score[12:21]) #del subject 11 for we dont have corresponding eeg data

## read F_SSD, the SSD for each subject
#?? how to store this: as df that would be 6001*2 SSDs rows and then a column for each subject?
F_SSD_file <- np$load(file.path(result_folder, 'F_SSD.npz')) 
F_SSD_file$files #contains arr_0 to arr_19
F_SSD <- vector("list", length(N_subjects))
for (subject in 1:N_subjects){
  if (subject<=10){
    F_SSD_subj <- F_SSD_file$f[[sprintf('arr_%d', subject-1)]] #python has 0 index
    F_SSD[[subject]] <- F_SSD_subj[1:2,,] # only need first two SSD Components
  }
  else if(subject==11) next #no data for subjcet 11
  else{
    F_SSD_subj <- F_SSD_file$f[[sprintf('arr_%d', subject)]] #no index is right     
    F_SSD[[subject]] <- F_SSD_subj[1:2,,] # only need first two SSD Components  
  }   
}
length(F_SSD) # list of N_subjects elements
dim(F_SSD[[1]]) #each element a about (2,6001,148) shaped array
#TODO: sollten wir die shape mit NaNs auffüllen, damit das richtig in die Matrix passt? 
#brauchen maske mit welcher trial ist snare, welcher wdblk, welcher NaN
# => em Ende eine F_SSD Liste für Snare und eine für WdBlk (siehe Design Matrix)


#check normality of response variable

## split into conditions
behavior_snare <- split(behavior_df, type_index)[[1]]
behavior_wdBlk <- split(behavior_df, type_index)[[2]]

## substract mean deviation
behavior_snare$dev <- behavior_snare$dev-mean(behavior_snare$dev)
behavior_wdBlk$dev <- behavior_wdBlk$dev-mean(behavior_wdBlk$dev)

## qq plot
qqnorm(behavior_snare$dev)
qqline(behavior_snare$dev)
qqnorm(behavior_wdBlk$dev)
qqline(behavior_wdBlk$dev)


# create design matrix
design_mat <- data.frame(rbind(rep(1,N_subjects*N_trials)))
beta1 <- matrix(nrow=N_comp, ncol=N_subjects*N_trials)
beta2 <- matrix(nrow=N_comp, ncol=N_subjects*N_trials)
beta3 <- c()
ypsilon0 <- matrix(nrow=N_subjects, ncol=N_subjects*N_trials)
ypsilon1 <- matrix(0L, nrow=N_subjects*N_comp, ncol=N_subjects*N_trials)
'TODO: 1,2,3,4 hardcoden in abhängigkeit von N_Comp (aufdröseln in N_freq und N_SSD?'
for (i in 1:N_subjects) {
  ## beta1 contains the 4 (2 SSDs for 2 freq) trial averaged components for every subject
  snare_Tmean <- sapply(F_SSD_snare[[i]], 3, mean)
  wdBlk_Tmean <- sapply(F_SSD_wdBlk[[i]], 3, mean)
  beta1[1:2, 1+(i-1)*N_trials:i*N_trials] <- F_SSD_snare[[i]] - snare_Tmean #trials are 3rd dim
  beta1[3:4, 1+(i-1)*N_trials:i*N_trials] <- F_SSD_wdBlk[[i]] - wdBlk_Tmean
  
  ## beta2 contains the 4 (2 SSDs for 2 freq) trial averages for every subject (repeated N_trials times)
  beta2[1:2, 1+(i-1)*N_trials:i*N_trials] <- rep(snare_Tmean, N_trials)
  beta2[3:4, 1+(i-1)*N_trials:i*N_trials] <- rep(wdBlk_Tmean, N_trials)
  
  ## beta3 contains the musical z_score of each proband repeated N_trials times
  beta3 <- c(beta3, rep(music_z_score[i], N_trials))
  
  ## ypsilon0 contains a 1 if subject row and column are the same, 0 else
  yps0 <- rep(0, N_subjects)
  yps0[i] <- 1
  ypsilon0[i,] <- rep(yps0, each=N_trials)
  
  ## ypsilon1 contains x_it-mean(x_i) if subject row and column are the same, 0 else
  ypsilon1[1+(i-1)*N_comp:2+(i-1)*N_comp, 1+(i-1)*N_trials:i*N_trials] <- F_SSD_snare[[i]] - snare_Tmean #trials are 3rd dim
  ypsilon1[3+(i-1)*N_comp:4+(i-1)*N_comp, 1+(i-1)*N_trials:i*N_trials] <- F_SSD_wdBlk[[i]] - wdBlk_Tmean
}
design_mat <- rbind(design_mat, beta1, beta2, beta3, ypsilon0, ypsilon1)


#TODO
#1 designmatrix erstellen (110x1500 df)
#2 OLS fit ohne gamlss (mixed RE anf FE model)
#3 gamlss

# explanatory variables: type_index, subject_ID, music_z_score, addInfo$LQ
mod<-gamlss(behavior_df$absDev~pb(x),sigma.fo=~pb(x), family=Normal, data=data, method=mixed(1,20))

#R cheatsheet
# first array slice: a[1, , ]
