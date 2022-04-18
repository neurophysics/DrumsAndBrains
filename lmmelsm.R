### test brms package with our data
# see Williams et al. 2019 paper

# todo:
# skripte für daten durchlaufen lassen
library('LMMELSM')

snare_dat <- read.csv("Results/snare_data_listen.csv", sep=',')

fit <- lmmelsm(
	       list(
	observed ~ deviation,
	location ~ musicality + trial + session + Snare1_between + Snare2_between + Snare3_between + Snare1_within + Snare2_within + Snare3_within,
	scale ~ musicality + trial + session + Snare1_between + Snare2_between + Snare3_between + Snare1_within + Snare2_within + Snare3_within,
	between ~ musicality + trial + session + Snare1_between + Snare2_between + Snare3_between + Snare1_within + Snare2_within + Snare3_within),
	group = subject, snare_dat, cores=8, iter=10000, control=list(adapt_delta=0.99))
