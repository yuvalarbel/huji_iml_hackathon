dt=read.csv("C:\\Users\\User\\Downloads\\waze_data.csv")

#Splitting train and test

n=dim(dt)[1]

ntrain = round(0.6*n)
ndev=round(0.2*n)
ntest=n-ntrain-ndev
ixtrain = sample.int(n,ntrain)

#All indices not used for train
ixtestdev=seq.int(n)[-ixtrain]

#Indices to be used for dev, test will be the rest
ixdev=sample(ixtestdev,ndev)
ixtest=seq.int(n)[-c(ixdev,ixtrain)]

dt.train=dt[ixtrain,]
dt.dev = dt[ixdev,]
dt.test = dt[ixtest,]


write.csv(dt.train,"C:\\Users\\User\\Desktop\\New folder\\waze_data_train.csv", row.names = TRUE)
write.csv(dt.dev,"C:\\Users\\User\\Desktop\\New folder\\waze_data_dev.csv", row.names = TRUE)
write.csv(dt.test,"C:\\Users\\User\\Desktop\\New folder\\waze_data_test.csv", row.names = TRUE)
