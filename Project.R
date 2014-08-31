##############################################################
####                                              ############
####  NORMALIZE FEATURES!!!!!!!!!!!!!!!!!!        ############
####                                              ############
##############################################################
###### give a try to Neural Networks, SVM, Random Forests



library
(devtools)
library(zoo)
library(xts)
library(Quandl)
library(quantmod)
Quandl.auth("mCDHcSdN9mQ_Hubid1Uq")
############ setting english as time language
Sys.setlocale("LC_TIME", "C")


###### getting data from the DOW JONES Index
sp = Quandl("YAHOO/INDEX_GSPC", type="xts", start_date="2010-01-01", end_date="2014-08-15")
#head(sp)

########### new script version importing  data as xts
names(sp) <- paste(c(names(sp)[1:5],'AdjClose'), '.SP500', sep='')
sp$Delta.SP500 <- Delt(sp$AdjClose.SP500)


#X11()
#plot(sp$Delta.SP500)


############## old version as data.frame 
# names(sp) <- c(names(sp)[1:6],'AdjClose')
# sp['Date'] <- as.Date(strptime(sp[,1], "%Y-%m-%d"))
# sp$DeltaPc <- c(-diff(sp$AdjClose,1),NA)/c(sp$AdjClose[2:nrow(sp)],NA)
# sp$DeltaPc[nrow(sp)] <- mean(sp$DeltaPc[1:(nrow(sp)-1)])
# 
# for (name in names(sp)){
#   print(name)
#   print(sum(is.na(sp[,name])))  
# }
# 
# plot(sp$Date,sp$AdjClose, type='l')
# 
# sp[which.max(sp$DeltaPc),]$Date
# abline(v=sp[which.max(sp$DeltaPc),]$Date)
# 
# sp[which.min(sp$AdjClose),]$Date
# abline(v=sp[which.min(sp$AdjClose),]$Date, col='red')
# 
# plot(sp$Date,sp$DeltaPc)

##########################################################################
#function to interpolate zero point in vector with mean of the k neighbours
interp <- function(vec.with.zero, k=3) {
  stopifnot(min(vec.with.zero) == 0)
  while (min(vec.with.zero) == 0) {
   ind <- which.min(vec.with.zero)
   begin <- ind - floor(k/2)
   end <- ind + floor(k/2)
   selected.points <- vec.with.zero[ c(begin:(ind-1),(ind+1):end) ]
   vec.with.zero[ind] <- mean(selected.points)
  }
  return(vec.with.zero)
}


############ OPEC oil price as xts
oilOpec = Quandl("OPEC/ORB", type="xts", start_date="2010-01-01", end_date="2014-08-15")
names(oilOpec) <- 'oilOPEC.Value'
sum(is.na(oilOpec))
#head(oilOpec)
#min(oilOpec)
#X11()
#plot(oilOpec$oilOPEC.Value)

#oilOpec['Date'] <- as.Date(strptime(oilOpec[,1], "%Y-%m-%d"))
#plot(oilOpec$Date,oilOpec$Value, type='l')

############ natural gas price
gas = Quandl("OFDP/FUTURE_NG1", type="xts", start_date="2010-01-01", end_date="2014-08-15")
names(gas) <- paste(names(gas), '.gas', sep='')
gas$Open.gas <- interp(gas$Open.gas)
#sum(is.na(gas))
#min(gas)
#X11()
#plot(gas$Open.gas)

#names(gas)
#dim(gas)
#summary(gas$Open)

#gas[which.min(gas$Open),'Date']
#gas[which.min(gas$High),'Date']

#gas['Date'] <- as.Date(strptime(gas[,1], "%Y-%m-%d"))
# X11()
# par(mfrow=c(2,2))
# for (name in names(gas)[2:5]){
#   print(name)
#   plot(gas$Date,gas[,name], type='l', ylab=name)  
# }
# 
# gas$Open.gas <- interp(gas$Open.gas)
# 
# plot(gas$Date,gas$Open, type='l')

############ corn price
corn = Quandl("OFDP/FUTURE_C1", type='xts', start_date="2010-01-01", end_date="2014-08-15")
names(corn) <- paste(names(corn), '.corn', sep='')
corn$Open.corn <- interp(corn$Open.corn)
#sum(is.na(corn))
#min(corn$Open.corn)
#plot(corn$Open.corn)
#names(corn)
#dim(corn)
#summary(corn)
# plot(corn$Date,corn$Open, type='l')
# corn['Date'] <- as.Date(strptime(corn[,1], "%Y-%m-%d"))
# corn$Open <- interp(corn$Open)
# plot(corn$Date,corn$Open, type='l')

########### hong kong stock
hkong = Quandl("YAHOO/INDEX_HSI", type="xts", start_date="2010-01-01", end_date="2014-08-15")
#head(hkong)
names(hkong) <- paste(c(names(hkong)[1:5],'AdjClose'), '.hkong', sep='')
hkong$Delta.hkong <- Delt(hkong$AdjClose.hkong)
#sum(is.na(hkong))
#X11()
#plot(hkong$Delta.hkong)

########### nikkei Tokyo Japan
nikkei = Quandl("YAHOO/INDEX_N225", type="xts", start_date="2010-01-01", end_date="2014-08-15")
#head(nikkei)
names(nikkei) <- paste(c(names(nikkei)[1:5],'AdjClose'), '.nikkei', sep='')
nikkei$Delta.nikkei <- Delt(nikkei$AdjClose.nikkei)
#X11()
#plot(nikkei$AdjClose)
#plot(nikkei$Delta.nikkei)

########### frankfurt Index
frankfurt = Quandl("YAHOO/INDEX_GDAXI", type="xts", start_date="2010-01-01", end_date="2014-08-15")
#head(frankfurt)
names(frankfurt) <- paste(c(names(frankfurt)[1:5],'AdjClose'), '.frankfurt', sep='')
frankfurt$Delta.frankfurt <- Delt(frankfurt$AdjClose.frankfurt)
#X11()
#plot(frankfurt$AdjClose)
#plot(frankfurt$Delta.frankfurt)

########### gold price
gold = Quandl("BOE/XUDLGPS", type="xts", start_date="2010-01-01", end_date="2014-08-15")
#head(gold)
names(gold) <- 'Value.gold'
#X11()
gold$Value.gold <- interp(gold$Value.gold)
#plot(gold$Value.gold)

########### euro vs usd
euro = Quandl("QUANDL/USDEUR", type="xts", start_date="2010-01-01", end_date="2014-08-15")
#head(euro)
names(euro)[1] <- 'Rate.euro'
#X11()
#euro$Rate.euro <- interp(euro$Rate.euro)
#plot(euro$Rate.euro)

########### aus vs usd
aus = Quandl("QUANDL/USDAUD", type="xts", start_date="2010-01-01", end_date="2014-08-15")
#head(aus)
names(aus)[1] <- 'Rate.aus'
#X11()
#aus$Rate.aus <- interp(aus$Rate.aus)
#plot(aus$Rate.aus)

########### Paris Index
paris = Quandl("YAHOO/INDEX_FCHI", type="xts", start_date="2010-01-01", end_date="2014-08-15")
#head(paris)
names(paris) <- paste(c(names(paris)[1:5],'AdjClose'), '.paris', sep='')
paris$Delta.paris <- Delt(paris$AdjClose.paris)
#X11()
#plot(paris$AdjClose)
#plot(paris$Delta.paris)

########### Dow Jones Index
djia = Quandl("YAHOO/INDEX_DJI", type="xts", start_date="2010-01-01", end_date="2014-08-15")
#head(djia)
names(djia) <- paste(c(names(djia)[1:5],'AdjClose'), '.djia', sep='')
djia$Delta.djia <- Delt(djia$AdjClose.djia)
#X11()
#plot(djia$AdjClose)
#plot(djia$Delta.djia)


########### 5 year US Treasury YTM
treasury5 = Quandl("YAHOO/INDEX_FVX", type="xts", start_date="2010-01-01", end_date="2014-08-15")
#head(treasury5)
#sum(is.na(treasury5))
names(treasury5) <- paste(c(names(treasury5)[1:5],'AdjClose'), '.treasury5', sep='')
treasury5$Delta.treasury5 <- Delt(treasury5$AdjClose.treasury5)
#X11()
#plot(treasury5$AdjClose)
#plot(treasury5$Delta.treasury5)


################### combining data
finance <- cbind(sp$Delta.SP500, sp[,1:5], #
                 djia$Delta.djia,
                 treasury5$Delta.treasury5,#
                 hkong$Delta.hkong, #
                 frankfurt$Delta.frankfurt, #
                 paris$Delta.paris,#
                 nikkei$Delta.nikkei,#
                 oilOpec$oilOPEC.Value,#
                 gas$Open.gas,#
                 corn$Open.corn,
                 gold$Value.gold,#
                 #euro$Rate.euro,
                 aus$Rate.aus
                 )

dim(finance)
head(finance, 50)
class(finance)
sum(is.na(finance))/sum(!is.na(finance))
for (name  in names(finance)) {
  finance[,name][is.na(finance[,name])] <- mean(finance[,name], na.rm = TRUE)
  print(sum(is.na(col)))
}

finance <- data.frame(date=index(finance), coredata(finance))
ncol(finance)
### remember that the dirst column is dates
finance[,2:ncol(finance)] <- scale(finance[,2:ncol(finance)])
finance <- finance[,2:ncol(finance)]
dim(finance)

#for (j  in 1:18) {
# print(class(finance[,j]))
#}

### turning delta S&P into factors for classification
finance$Delta.SP500 <- as.factor(ifelse(finance$Delta.SP500 >= 0, 'Up','Down'))
table(finance$Delta.SP500)
######## function for train and test selection
# we assume that the predictor is always the first column
selectTrainTest <- function(dataset, region, percentage) {
  split <- floor(dim(dataset)*percentage)[1]
  if (region == 'end') {
    train <- dataset[1:split,]
    test <- dataset[(split+1):nrow(dataset),]
    return(list(train=train, test=test))
  }
  else if (region == 'begin') {
    newsplit <- nrow(dataset) - split
    train <- dataset[(newsplit+1):nrow(dataset),]
    test <- dataset[1:newsplit,]
    return(list(train=train, test=test))
  }
} 

tt <- selectTrainTest(finance, 'end', 0.8)
X11()
pairs(finance[,2:ncol(finance)])
############### first random forests
set.seed(1)
library(randomForest)

rf <- randomForest(Delta.SP500~., tt$train)
#rf$importance
#sum(diag(rf$confusion[1:2,1:2]))/sum(rf$confusion[1:2,1:2])

prediction <- predict(rf, tt$test)
tb <- table(tt$test[,1],prediction)
sum(diag(tb))/sum(tb)

###################### support vector machine
library(e1071)
tune.out = tune(svm, Delta.SP500~., data=tt$train, kernel="radial",
                    ranges = list(cost = c(1, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1),
                                    gamma = c(0.001, 0.005, 0.007, 0.009, 0.1, 0.2)))
summary(tune.out)
#sv <- svm(Delta.SP500~., trainFin, kernel='radial') 
ypred = predict(tune.out$best.model,tt$test)
#names(sv)
#sv$index
tb <- table(tt$test[,1],ypred)
sum(diag(tb))/sum(tb)

################### neural networks
#install.packages('nnet')
library(nnet)
nn = nnet(Delta.SP500~., data=tt$train,size=20,maxit=10000,decay=.001)
tb <- table(tt$test[,1], predict(nn, newdata=tt$test, type="class"))
sum(diag(tb))/sum(tb)

install.packages('RSNNS')
library(RSNNS)
#demo(iris)


################### claudia
install.library('ggplot2')
library(ggplot2)
k <- 3
x <- seq(-2*k, 2*k, 0.1)
y <- seq(-2*k, 2*k, 0.1)
y[(y>-k & y<k)] <- 0
y[y <= -k] <- y[y <= -k] + k
y[y >= k] <- y[y >= k] - k


df <- as.data.frame(cbind(x,y))
g <- ggplot(df, aes(x = x, y = y)) + geom_line(size=2) 
g <- g + theme(axis.ticks = element_blank(), axis.text.x = element_blank(), axis.text.y = element_blank())
#g <- g + geom_hline(aes(yintercept=0), colour="#990000", linetype="dashed")
#g <- g + geom_vline(aes(yintercept=0), colour="#990000", linetype="dashed")
g <- g + geom_hline(aes(yintercept=0), colour="#990000")
g <- g + geom_vline(aes(yintercept=0), colour="#990000")
g <- g + theme(axis.title.x = element_blank()) + theme(axis.title.y = element_blank())
g <- g + annotate("text", x=0.5, y=2.8, label="G[k](u)", size = 12,parse=TRUE)
g <- g + annotate("text", x=6.2, y=-0.2, label="u", size = 12,parse=TRUE)
g <- g + annotate("text", x=3, y=-0.2, label="k", size = 12,parse=TRUE)
g <- g + annotate("text", x=-2.9, y=-0.2, label="-k", size = 12,parse=TRUE)
ggsave("gku.pdf")
print(g)

##########
# #########        FIRST TWITTER FINANCE DATA
# ####################################################################################################
# ####################################################################################################
# ####################################################################################################
# ######## dealing with twitter data
# stocksTwitter <- read.table('stock_symbol_keywords.tsv', sep='\t')
# names(stocksTwitter) <- c('Time', 'TickerSymbol', 'TweetID', 'KeyWords')
# library(stringi)
# stocksTwitter$TickerSymbol <-stri_trans_tolower(stocksTwitter$TickerSymbol)
# stocksTwitter$Time <- as.character(stocksTwitter$Time)
# timeConvert <- function(charTime) {
#   charTime <- strsplit(charTime,'')
#   d <- unlist(charTime)
#   return(sprintf("%s/%s/%s %s:%s:%s", paste(d[1:4],collapse=''),
#           paste(d[5:6],collapse=''),
#           paste(d[7:8],collapse=''),
#           paste(d[9:10],collapse=''),
#           paste(d[11:12],collapse=''),
#           paste(d[13:14],collapse='')))
# }
# 
# stocksTwitter$Time <- lapply(stocksTwitter$Time, timeConvert)
# stocksTwitter$Time <- unlist(stocksTwitter$Time)
# stocksTwitter$Time <- strptime(stocksTwitter$Time, "%Y/%m/%d %H:%M:%S")
# stocksTwitter$Time[1]
# stocksTwitter$Time[nrow(stocksTwitter)]




rfr <- 0.025
mr <- 0.08
betas <- c(1.4,2,0.7)
ret <- rfr + (mr-rfr)*betas
retP <- (sum(1/3*ret))*100
retP


tot <- 400*82 + 300*46 + 300*124
SF <- (300*124*100)/tot
SF


###########################
prop <- c(0.3,0.5,0.2)

X <- c(12,7,0)
Y <- c(7,6,5)
Z <- c(-1,0,6)

rX <- sum(X*prop)
varX <- sum(X^2*prop)-sum(X*prop)^2
stdX <- sqrt(varX)

rY <- sum(Y*prop)
varY <- sum(Y^2*prop)-sum(Y*prop)^2
stdY <- sqrt(varY)

rZ <- sum(Z*prop)
varZ <- sum(Z^2*prop)-sum(Z*prop)^2
stdZ <- sqrt(varZ)


covXY <- sum(prop*((X - rX)*(Y - rY)))
covXY
covXZ <- sum(prop*((X - rX)*(Z - rZ)))
covXZ
covYZ <- sum(prop*((Y - rY)*(Z - rZ)))
covYZ



por1 <- c(0.2,0.4,0.4)
por2 <- c(0.34,0.33,0.33)
por3 <- c(0.5,0.25,0.25)
por4 <- c(0.15,0.7,0.15)
variances <- c(varX, varY, varZ)

#varPor1 <- sum(por1^2*variances) + 2*por1[1]*por1[2]*covXY + 2*por1[1]*por1[3]*covXZ + 2*por1[2]*por1[3]*covYZ
varPor1 <- por1[1]^2*varX + por1[2]^2*varY + por1[3]^2*varZ + 2*por1[1]*por1[2]*covXY + 2*por1[1]*por1[3]*covXZ + 2*por1[2]*por1[3]*covYZ
varPor1

varPor2 <- por2[1]^2*varX + por2[2]^2*varY + por2[3]^2*varZ + 2*por2[1]*por2[2]*covXY + 2*por2[1]*por2[3]*covXZ + 2*por2[2]*por2[3]*covYZ
varPor2

varPor3 <- por3[1]^2*varX + por3[2]^2*varY + por3[3]^2*varZ + 2*por3[1]*por3[2]*covXY + 2*por3[1]*por3[3]*covXZ + 2*por3[2]*por3[3]*covYZ
varPor3

varPor4 <- por4[1]^2*varX + por4[2]^2*varY + por4[3]^2*varZ + 2*por4[1]*por4[2]*covXY + 2*por4[1]*por4[3]*covXZ + 2*por4[2]*por4[3]*covYZ
varPor4



