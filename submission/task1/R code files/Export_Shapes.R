dt=read.csv("C:\\Users\\User\\Downloads\\waze_data.csv")
library(dummies)
#Splitting train and test

dt=dt.train
locs=dt[,c(17,18,19)]
hist(locs$update_date,breaks=200)

val=''
dt.yes.city=dt[!is.element(dt$linqmap_city, val),]
loc.yes.city=dt.yes.city[,c(18,19)]

write.csv(loc.yes.city,"C:\\Users\\User\\Desktop\\IMLHack\\locations_no_city.csv")
