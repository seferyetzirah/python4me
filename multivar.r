
install.packages("plotly")
install.packages("heatmaply")
install.packages("ggcorrplot")

install.packages('installr'); install.Rtools()
install.packages('Rtools')

install.packages.2 <- function (pkg) if (!require(pkg)) install.packages(pkg);
install.packages.2('devtools')
# make sure you have Rtools installed first! if not, then run:

library("heatmaply")

spdata <- read.csv("C:/stockpile/mv.csv", stringsAsFactors=FALSE, header = TRUE)
attach(spdata)
class(spdata)
spdata

spdata$HAY.FAT

spdata[spdata$HAY.FAT > 42,]

str(spdata)
colnames(spdata)  
ncol(spdata)
nrow(spdata)

df  <-normalize(spdata)
df

heatmaply(df)

heatmaply_cor(
  cor(spdata),
  xlab = "spdata",
  ylab = "spdata$PH2.CELLULOSE",
  k_col = 2,
  k_row = 2
)


model <- lm(spdata$ï..PH2.CELLULOSE ~  spdata$HAY.FAT + spdata$HAY.WAC5 + spdata$HAY.WAC24 + spdata$HAY.N 
+ spdata$HAY.RESISTANCE + spdata$HAY.STRUCTURE + spdata$HAY.HEMI + spdata$HAY.CELLULOSE + spdata$PRESP.FAT 
+ spdata$PRESP.WAC5+ spdata$PRESP.WAC24+ spdata$PRESP.N + spdata$PRESP.RESISTANCE + spdata$PRESP.STRUCTURE 
+spdata$PRESP.HEMI +spdata$PRESP.CELLULOSE+ spdata$SP.FAT + spdata$SP.WAC5 +spdata$SP.WAC24 + spdata$SP.N 
+ spdata$SP.RESISTANCE +spdata$SP.STRUCTURE +spdata$SP.HEMI +spdata$SP.CELLULOSE,data =spdata)
model
summary(model)

xvar <-spdata[,2:25]
xvar
yvar <-spdata[,1]
yvar 
attach(spdata)

pairs(xvar+yvar, data=spdata)
plot(yvar)
