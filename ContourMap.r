
#http://stat405.had.co.nz/ggmap.pdf
df<-read.csv('C:\\Users\\samee\\Desktop\\CSCI 565 DM\\DM Project\\train.csv',header=TRUE, sep=',')
# dfTest<-read.csv('C:\\Users\\samee\\Desktop\\CSCI 565 DM\\DM Project\\test.csv',header=TRUE, sep=',')
# df<-rbind(df,dfTest)

sf <- c(left = -122.516441, bottom = 37.706027, right = -122.37276, top = 37.841171)
library(ggmap)
map <- get_map(sf, zoom = 13,source="google",maptype="roadmap")
dfFiltered <- subset(df,
  -122.516441 <= X & X <= -122.37276 &
   37.706027 <= Y & Y <=  37.841171 &
    Category %in% c("DRUNKENNESS","DRIVING UNDER THE INFLUENCE")
)
dfFiltered$Dates<- as.POSIXct(as.character(dfFiltered$Dates), format="%Y-%m-%d %H:%M:%S")
dfFiltered$Hour<-as.POSIXlt(dfFiltered$Dates)$hour
dfFiltered$Year<-as.POSIXlt(dfFiltered$Dates)$year
ggmap(map) + stat_density2d(aes(x=X,y=Y,fill= ..level..),size=2, bins=7,alpha=0.5,data=dfFiltered,geom="polygon")+
labs(title = "Crime Density for Drinking Crimes in San Francisco")+
scale_fill_gradient(low = "yellow",high= "red")

#,"ASSAULT","BURGLARY","DRUG/NARCOTIC","KIDNAPPING","ROBBERY","VEHICLE THEFT","VANDALISM","SEX OFFENSES FORCIBLE")
#Plot graph#ggmap(map) + geom_point(aes(x=X,y=Y,color=Category),size = 0.5, alpha = 0.5,data=dfFiltered) + geom_point(size=3,alpha=0.3)
ggsave("plotDrinkCrimes.png",dpi = 500,width=10,height=7)



dfNight<-subset(dfFiltered,0 <= Hour & Hour < 4)
dfNight$timeSlot<-"0-4"
dfMorning<-subset(dfFiltered,4 <= Hour & Hour < 8)
dfMorning$timeSlot<-"4-8"
dfAfternoon<-subset(dfFiltered,	8 <= Hour & Hour < 12)
dfAfternoon$timeSlot<-"8-12"
dfevening<-subset(dfFiltered,	12 <= Hour & Hour < 16)
dfevening$timeSlot<-"12-16"
df16<-subset(dfFiltered,	16 <= Hour & Hour < 20)
df16$timeSlot<-"16-20"
df20<-subset(dfFiltered,	20 <= Hour & Hour < 24)
df20$timeSlot<-"20-24"


dfFiltered<-rbind(dfNight,dfMorning,dfAfternoon,dfevening,df16,df20)

#df$timeSlot<-ifelse((dfFiltered$Hour>=0 && dfFiltered$Hour < 6), "0-6",ifelse((dfFiltered$Hour>=6 && dfFiltered$Hour < 12),"6-12",ifelse((dfFiltered$Hour>=12 && dfFiltered$Hour < 18),"12-18","18-24")))
# else if(dfFiltered$Hour>=6 && dfFiltered$Hour < 12){dfFiltered$timeSlot<-"6-12"}
# else if(dfFiltered$Hour>=12 && dfFiltered$Hour < 18){dfFiltered$timeSlot<-"12-18"}
# else if(dfFiltered$Hour>=18 && dfFiltered$Hour < 24){dfFiltered$timeSlot<-"18-24"}

ggmap(map) + stat_density2d(aes(x=X,y=Y,fill= ..level..),size=2, bins=7,alpha=0.5,data=dfFiltered,geom="polygon")+
labs(title = "Crime Density of of Assault by Time Slots in San Francisco")+
scale_fill_gradient(low = "yellow",high= "red")+facet_wrap(~ timeSlot )
ggsave("plotDrinkTimewise.png",dpi = 500,width=10,height=7)