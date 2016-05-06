
#@Title: Implementation of various classifiers for SF Crime Classification

print "importing required libraries ..."
import os 
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from pandas.tools.plotting import scatter_matrix
from sklearn.svm import SVC, LinearSVC
import time
from datetime import datetime
import thread
import matplotlib.pyplot as plt

#https://www.youtube.com/watch?v=7gAZoK6kGhM

def getData():
	print "reading file ..."
	data=pd.read_csv('../train.csv')   #, compression="infer
  	return data

def mapAddr(x):
	x=str(x)
	temp=''.join([c for c in x if (c.isupper() or c=='/')])
	return temp

def splitDate(x):
	dateObject=datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
	time=dateObject.hour
	day=dateObject.day
	month=dateObject.month
	year=dateObject.year
	return time,day,month,year

def get_season(x):
    season=0
    if (x in [5, 6, 7]):
        season=1 #summer
    if (x in [8, 9, 10]):
        season=2 #fall
    if (x in [11, 0, 1]):
        season=3 #winter
    if (x in [2, 3, 4]):
        season=4 #spring
    return season


def preprocessData(trainDF):
	start=time.time()
	y=trainDF.Category.values
	print "pre processing data ..."
	dictData={}
	days={}
	temp=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
	for i in temp:
		days[i]=temp.index(i)
	dictData["DayOfWeek"]=days

	pdDist={}
	temp=sorted(trainDF.PdDistrict.unique().tolist())
	for i in temp:
		pdDist[i]=temp.index(i)
	dictData["PdDistrict"]=pdDist

	# trainDF['intersection']=trainDF['Address'].apply(lambda x: 1 if "/" in x else 0)
	trainDF.Address= trainDF.Address.apply(mapAddr)

	# print trainDF.Address
	addr={}
	temp=sorted(trainDF.Address.unique().tolist())
	for i in temp:
		addr[i]=temp.index(i)
	dictData["Address"]=addr
	# print dictData["Address"]

	cat={}
	temp=sorted(trainDF.Category.unique().tolist())
	for i in temp:
		cat[i]=temp.index(i)
	dictData["Category"]=cat

	xcol={}
	trainDF.X=pd.cut(trainDF.X,70)
	temp=sorted(trainDF.X.unique())
	for i in temp:
		xcol[i]=temp.index(i)
	dictData["X"]=xcol

	ycol={}
	trainDF.Y=pd.cut(trainDF.Y,70)
	temp=sorted(trainDF.Y.unique())
	for i in temp:
		ycol[i]=temp.index(i)
	dictData["Y"]=ycol
	

	print"\tmapping DayOfWeek"
	trainDF.DayOfWeek=trainDF.DayOfWeek.map(days)
	print"\tmapping PdDistrict"
	trainDF.PdDistrict=trainDF.PdDistrict.map(pdDist)
	print"\tmapping Address"
	trainDF.Address=trainDF.Address.map(addr)
	# print trainDF.Address
	print"\tsplitting Date"
	trainDF["time"],trainDF["day"],trainDF["month"],trainDF["year"]=zip(*trainDF["Dates"].apply(splitDate))
	print "\tsplitting X"
	trainDF.X=trainDF.X.map(xcol)
	print "\tsplitting Y"
	trainDF.Y=trainDF.Y.map(ycol)

	trainDF.Category=trainDF.Category.map(cat)
	# print trainDF.X.unique()
	# trainDF["season"]=trainDF['month'].apply(get_season)
	# plt.hist(trainDF.X,bins=20)
	# plt.axis(-123,-119)
	# fig=plt.gcf()
	# plt.savefig("hist.png")
	trainDF=trainDF.drop(['Dates','Descript','Resolution','Category'],axis=1)
	# corr=trainDF.corr(method="pearson")
	# print corr
	trainDF=trainDF.drop(['time','Y','X','year'],axis=1)
	# print "\tnormalizing values in data frame"
	# for col in trainDF.columns.tolist():
	# 	trainDF[col]=(trainDF[col] - trainDF[col].mean())/trainDF[col].std(ddof=0)
	# x=trainDF.values
	# minMaxScalar=preprocessing.MinMaxScalar()
	# x_scaled = minMaxScalar.fit_transform(x)
	# trainDF= pandas.DataFrame(x_scaled)

	# print "\tplotting scatter plot"
	# scatter=scatter_matrix(trainDF)
	# plt.savefig("scatterPlot.png")
	# print "\tscatter plot saved."
	# trainDF=trainDF.drop([],axis=1)

	end=time.time()
	print trainDF.head()
	print"Columns considered for classification: "
	print list(trainDF.columns.values)
	print "\tTime taken for pre processing :"+str(end-start)
	return trainDF,y

def svm(trainDF,y):
	start=time.time()
	print "#"*70
	print "Using Support Vector Machines Classifier"
	print "#"*70
	trainingVectors=trainDF.as_matrix()
	clf =SVC(C=1, kernel = 'linear', gamma=1, verbose= False, probability=False)
	print "training classifier ..."
	clf.fit(trainDF,y)
	print "performing cross fold validation ..."
	predicted = cross_validation.cross_val_predict(clf, trainDF, y, cv=5)
	print "accuracy score: ", metrics.accuracy_score(y, predicted)
	print "precision score: ", metrics.precision_score(y, predicted,average='weighted')
	# print "recall score: ", metrics.recall_score(y, predicted,average='weighted')
	end=time.time()
	print "Execution time for classifier: "+str(end-start)
	print "#"*70
	return

def decisionTree(trainDF,y):
	start=time.time()
	print "#"*70
	print "Using Decision Tree Classifier"
	print "#"*70
	trainingVectors=trainDF.as_matrix()
	clf=DecisionTreeClassifier(criterion="entropy")
	print "training classifier ..."
	clf.fit(trainDF,y)
	print "performing cross fold validation ..."
	predicted = cross_validation.cross_val_predict(clf, trainDF, y, cv=5)
	print "accuracy score: ", metrics.accuracy_score(y, predicted)
	print "precision score: ", metrics.precision_score(y, predicted,average='weighted')
	# print "recall score: ", metrics.recall_score(y, predicted,average='weighted')
	end=time.time()
	print "Execution time for classifier: "+str(end-start)
	print "#"*70
	return

def randomForest(trainDF,y):
	start=time.time()
	print "#"*70
	print "Using Random Forest Classifier"
	print "#"*70
	trainingVectors=trainDF.as_matrix()
	clf=RandomForestClassifier(n_estimators=2,criterion="entropy")
	print "training classifier ..."
	clf.fit(trainDF,y)
	print "performing cross fold validation ..."
	predicted = cross_validation.cross_val_predict(clf, trainDF, y, cv=5)
	print "accuracy score: ", metrics.accuracy_score(y, predicted)
	print "precision score: ", metrics.precision_score(y, predicted,average='weighted')
	# print "recall score: ", metrics.recall_score(y, predicted,average='weighted')
	end=time.time()
	print "Execution time for classifier: "+str(end-start)
	print "#"*70
	return

def nBModelling(trainDF,y):
	start=time.time()
	print "#"*70
	print "Using Naive Bayes classifier"
	print "#"*70
	trainingVectors=trainDF.as_matrix()
	clf = GaussianNB()
	print "training classifier ..."
	clf.fit(trainDF,y)
	print "performing cross fold validation ..."
	predicted = cross_validation.cross_val_predict(clf, trainDF, y, cv=5)
	print "accuracy score: ", metrics.accuracy_score(y, predicted)
	print "precision score: ", metrics.precision_score(y, predicted,average='weighted')
	# print "recall score: ", metrics.recall_score(y, predicted,average='weighted')
	# print "classification_report: \n ", metrics.classification_report(y, predicted)
	# print "confusion_matrix:\n ", metrics.confusion_matrix(y, predicted)
	end=time.time()
	print "Execution time for classifier: "+str(end-start)
	print "#"*70

	return
	

def main():
	trainDF=getData()
	trainDF,y=preprocessData(trainDF)
	
	nBModelling(trainDF,y)
	decisionTree(trainDF,y)
	randomForest(trainDF,y)
	# svm(trainDF,y)
	return
	

if __name__=="__main__":
	main()
