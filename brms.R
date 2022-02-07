### test brms package with our data
# see Williams et al. 2019 paper

# todo:
# skripte für daten durchlaufen lassen
library(brms)
snare_dat <- read.csv("Results/snare_data_both.csv", sep=' ')

#might not need this, instead standardise everything except subject
for (i in 5:12){ 
  colnames(snare_dat)[i] = gsub('_', '', colnames(snare_dat)[i]) #delete '_'
}
# define the model 

#gunnar
deviation ~ 1 + trial + session + 
  (1|c|subject) #random intercept,
sigma ~ 1 + session + trial + musicality + Snare1 + Snare2 
  + WdBlk1 + WdBlk2 + Snare1within + Snare2within 
  + WdBlk1within + WdBlk2within
# => deviation can be negative so being more consistent is most interesting. 
# within subject: add (deviation | subject) ?
# bf short for brmsformula
b_mod2 <- brmsformula( # all effects are assumed to vary across grouping (subject) except musicality, right?
    precision ~ musicality + (1 + session + trial + Snare1 + Snare2 
              + WdBlk1 + WdBlk2 + Snare1within + Snare2within 
              + WdBlk1within + WdBlk2within | subject), 
    sigma ~ musicality + (1 + session + trial + Snare1 + Snare2
              + WdBlk1 + WdBlk2 + Snare1within + Snare2within 
              + WdBlk1within + WdBlk2within | subject),
              nl = FALSE)
prior2 <- get_prior(b_mod2, data=snare_dat, family=gaussian()) #get priors
# error: only supported prior for correlation matrices is the 'lkj' prior => TODO: find that prior and change it
prior2$prior[1:length((prior2))] <- "normal(0,20)" #set all priors to gaussian
# rathe student t instead of normal because it is less peaky
# e.g. student(3,0,1) mean 3 degrees of freedem

b_fit2 = brm(b_mod2, prior = prior2, data = snare_dat,
             control = list(adapt_delta = .9999, max_treedepth = 12),
             chains = 4, iter = 10000,
             inits = 0, cores = 8)
summary(b_fit2)


# closer to example
b_mod3 <- brmsformula( # all effects are assumed to vary across grouping (subject) except musicality, right?
    precision ~  1 + musicality + session + trial + Snare1 + Snare2 
                 + WdBlk1 + WdBlk2 + Snare1within + Snare2within 
                 + WdBlk1within + WdBlk2within,
    session ~ 1 + (1|c|subject),
    trial ~ 1 + (1|c|subject),
    Snare1 ~ 1 + (1|c|subject),
    Snare2 ~ 1 + (1|c|subject),
    WdBlk1 ~ 1 + (1|c|subject),
    Snare1within ~ 1 + (1|c|subject),
    Snare2within ~ 1 + (1|c|subject),
    WdBlk1within ~ 1 + (1|c|subject),
    WdBlk2within ~ 1 + (1|c|subject),
    sigma ~ musicality + (1 + session + trial + Snare1 + Snare2
                        + WdBlk1 + WdBlk2 + Snare1within + Snare2within 
                        + WdBlk1within + WdBlk2within | subject),
  nl = TRUE)

###### define priors ############
# chose Gaussian with high variance (uninformative)
prior3 = c(set_prior("normal(0, 20)", nlpar = "musicality"), 
           set_prior("normal(0, 20)", nlpar = "Snare1"),
           set_prior("normal(0, 20)", nlpar = "Snare2"),
           set_prior("normal(0, 20)", nlpar = "WdBlk1"),
           set_prior("normal(0, 20)", nlpar = "WdBlk2"),
           set_prior("normal(0, 20)", nlpar = "Snare1within"),
           set_prior("normal(0, 20)", nlpar = "Snare2within"),
           set_prior("normal(0, 20)", nlpar = "WdBlk1within"),
           set_prior("normal(0, 20)", nlpar = "WBlk2within"),
           set_prior("normal(0, 20)", class = "sd", nlpar = "musicality"), 
           set_prior("normal(0, 20)", class = "sd", nlpar = "Snare1"),
           set_prior("normal(0, 20)", class = "sd", nlpar = "Snare2"),
           set_prior("normal(0, 20)", class = "sd", nlpar = "WdBlk1"),
           set_prior("normal(0, 20)", class = "sd", nlpar = "WdBlk2"),
           set_prior("normal(0, 20)", class = "sd", nlpar = "Snare1within"),
           set_prior("normal(0, 20)", class = "sd", nlpar = "Snare2within"),
           set_prior("normal(0, 20)", class = "sd", nlpar = "WdBlk1within"),
           set_prior("normal(0, 20)", class = "sd", nlpar = "WBlk2within"))

##### fit the model #####
b_fit3 = brm(b_mod3, prior = prior3, data = snare_dat,
             control = list(adapt_delta = .9999, max_treedepth = 12),
             chains = 4, iter = 10000,
             inits = 0, cores = 8)
summary(b_fit3)


# example from paper
dat <- read.csv("dat.csv")

# define the model 
b_mod1 <- bf(recall ~ betaMu + (alphaMu - betaMu) * exp(-exp(gammaMu) * trial), #asymptotic function (see Model specification page 1972)
             betaMu ~ 1 + (1|c|subject), #eigenes intercept + RE mit intercept
             alphaMu ~ 1 + (1|c|subject),
             gammaMu ~ 1 + (1|c|subject),
             nl = TRUE) +
          nlf(sigma ~ betaSc + (alphaSc - betaSc) * exp(-exp(gammaSc) * trial),
              alphaSc ~ 1 + (1|c|subject), 
              betaSc ~ 1 + (1|c|subject),
              gammaSc ~ 1 +  (1|c|subject)) # variance is non linear 

###### define priors ############
prior1 <- c(set_prior("normal(25, 2)", nlpar = "betaMu"),
           set_prior("normal(8, 2)",  nlpar = "alphaMu"),
           set_prior("normal(-1, 2)",  nlpar = "gammaMu"),
           set_prior("normal(0, 1)", nlpar = "betaSc"),
           set_prior("normal(1, 1)",  nlpar = "alphaSc"),
           set_prior("normal(0, 1)",  nlpar = "gammaSc"),
           set_prior("student_t(3, 0, 5)" , class = "sd", nlpar = "alphaMu"),
           set_prior("student_t(3, 0, 5)" , class = "sd", nlpar = "betaMu"),
           set_prior("student_t(3, 0, 5)" , class = "sd", nlpar = "gammaMu"),
           set_prior("student_t(3, 0, 1)" , class = "sd", nlpar = "alphaSc"),
           set_prior("student_t(3, 0, 1)" , class = "sd", nlpar = "betaSc"),
           set_prior("student_t(3, 0, 1)" , class = "sd", nlpar = "gammaSc")
           )

## fit the model
b_fit1 <- brm(b_mod1, prior = prior1, data = dat,
             control = list(adapt_delta = .9999, max_treedepth = 12),
             chains = 4, iter = 10000,
             inits = 0, cores = 8)
summary(b_fit1)
