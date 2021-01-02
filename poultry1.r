
install.packages("plotly")
install.packages("heatmaply")
install.packages("ggcorrplot")

install.packages('installr'); install.Rtools()
install.packages('Rtools')

install.packages.2 <- function (pkg) if (!require(pkg)) install.packages(pkg);
install.packages.2('devtools')
# make sure you have Rtools installed first! if not, then run:

library("heatmaply")

spdata1 <- read.csv("C:/stockpile/ptvn1.csv", stringsAsFactors=FALSE, header = TRUE)
attach(spdata1)
class(spdata1)
spdata1

spdata1$ï..P.NH3
spdata1$PH2.DM.TVN

spdata1[spdata1$ï..P.NH3> 67,]

str(spdata1)
colnames(spdata1)  
ncol(spdata1)
nrow(spdata1)

df  <-normalize(spdata1)
df

heatmaply(df)

heatmaply_cor(
  cor(spdata1),
  xlab = "spdata1$ï..P.NH3",
  ylab = "spdata1$PH2.DM.TVN",
  k_col = 2,
  k_row = 2
)


model <- lm(spdata1$ï..P.NH3 ~  spdata1$PH2.DM.TVN,data =spdata)
model
summary(model)

xvar <-spdata1$ï..P.NH3
xvar
yvar <-spdata1$PH2.DM.TVNyvar 
attach(spdata1)

pairs(xvar+yvar, data=spdata1)
plot(yvar)
