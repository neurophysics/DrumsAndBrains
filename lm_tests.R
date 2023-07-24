#works:
x <- rnorm(15)
y <- 5 + rnorm(15)
model <- lm(y ~ 1 + x, data=data.frame(x,y))
x_new <- seq(-3, 3, 1)
new <- data.frame(x=x_new)
predict(model, newdata=new)

# this also works
x <- array(rnorm(40), dim=c(10,4))
y <- 5 + rnorm(10)
model <- lm(y ~ 1 + X1 + X2 + X3 + X4, data=data.frame(x))
x_new <- array(seq(-3, 3, 0.25), dim=c(7,4))
predict(model, newdata=data.frame(x_new))

# and this
x <- array(rnorm(40), dim=c(10,4))
y <- 5 + rnorm(10)
model <- lm(formula(data.frame(y,x)), data=data.frame(x))
x_new <- array(seq(-3, 3, 0.25), dim=c(7,4))
predict(model, newdata=data.frame(x_new))

# but this doesnt
x <- array(rnorm(40), dim=c(10,4))
y <- 5 + rnorm(10)
model <- lm(y ~ 1 + x, data=data.frame(x))
x_new <- array(seq(-3, 3, 0.25), dim=c(7,4))
predict(model, newdata=data.frame(x_new))





x <- data.frame(array(rnorm(20), dim=c(10,2)))
y <- 5 + rnorm(10)
model <- lm(y ~ 1 + x, data=data.frame(y,x))
x_new <- data.frame(array(seq(-3, 3, 0.5), dim=c(7,2)), dimnames = list(NULL, seq(1,2)))
new <- data.frame(x=x_new)
predict(
  model, newdata=data.frame(array(
    seq(-3, 3, 0.5), 
    dim=c(7,2), 
    dimnames = NULL)
    ))
  )

data.frame(matrix(ncol=3,nrow=0, dimnames=list(NULL, c("name", "age", "gender"))))



