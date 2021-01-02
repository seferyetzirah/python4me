---
title: "R Notebook"
output: html_notebook
---

install.packages("plotly")
install.packages("heatmaply")
install.packages("ggcorrplot")

install.packages('installr'); install.Rtools()
install.packages('Rtools')

install.packages.2 <- function (pkg) if (!require(pkg)) install.packages(pkg);
install.packages.2('devtools')
# make sure you have Rtools installed first! if not, then run:

library("heatmaply")

spdata <- read.csv("C:/stockpile/multivar.csv", stringsAsFactors=FALSE, header = TRUE)
attach(spdata)

colnames(spdata)  
ncol(spdata)
n <- nrow(spdata)
str(spdata)
xvar <-spdata[,3:26]
xvar
yvar <-spdata[,27]
yvar 
attach(spdata)

pairs(xvar+yvar, data=spdata)
plot(yvar)


df  <-normalize(spdata)
df

heatmaply(df)

  

