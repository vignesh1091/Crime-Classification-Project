import pandas as pd
import time
from datetime import datetime
from sklearn import metrics
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC, LinearSVC
import gzip

def getData():
  print "reading file ..."
  data=pd.read_csv('../train.csv')
  test=pd.read_csv('../test.csv')
  return data,test

def preprocessTest(trainDF):
	print "pre processing data ..."
  	start=time.time()
	return trainDF

def decisionTree(trainDF,y,xHat):
  start=time.time()
  print "#"*70
  print "Using Decision Tree Classifier"
  print "#"*70
  trainingVectors=trainDF.as_matrix()
  # clf=DecisionTreeClassifier(criterion="entropy")
  clf = MultinomialNB()
  print "training classifier ..."
  clf.fit(trainDF,y)
  print "predicting classes for test data"
  # xHat.drop(['Id'])
  yHat=clf.predict_proba(xHat)
  print"yhat"
  print yHat[0]
  end=time.time()
  print "Execution time for classifier: "+str(end-start)
  print "#"*70
  return yHat,clf


def daytime(x):
  # eMorning=0
  # morning=0
  # afternoon=0
  # evening=0
  # night=0
  a=0
  b=0
  c=0
  d=0
  e=0
  f=0
  # if (x in [4,5,6,7]):
  #   eMorning=1
  # if (x in [8,9,10,11]):
  #   morning=1
  # if (x in [12,13,14,15,16]):
  #   afternoon=1
  # if (x in [17,18,19,20,21,22,23,0,1,2,3]):
  #   night=1
  if (x in [4,5,6,7]):
    a=1
  if (x in [8,9,10,11]):
    b=1
  if (x in [12,13,14,15]):
    c=1
  if (x in [16,17,18,19]):
    d=1
  if (x in [20,21,22,23]):
    e=1
  if (x in [0,1,2,3]):
    f=1
  return a,b,c,d,e,f

def splitDate(x):
  dateObject=datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
  time=dateObject.hour
  day=dateObject.day
  month=dateObject.month
  year=dateObject.year
  return time,day,month,year

def preprocessData(trainDF):
  print "pre processing data ..."
  start=time.time()
  cols=trainDF.columns.values.tolist()
  if ('Category'in cols):
  	trainDF=trainDF.drop(['Category','Descript','Resolution'],axis=1)
  df=pd.DataFrame()
  print"seperating districts"
  df=pd.get_dummies(trainDF['PdDistrict'],prefix='pD')
  trainDF=pd.concat([trainDF,df],axis=1)
  df=pd.DataFrame()
  print "seperating days of week"
  df=pd.get_dummies(trainDF['DayOfWeek'],prefix='day')
  trainDF=pd.concat([trainDF,df],axis=1)
  print "seperating time"
  trainDF["time"],trainDF["day"],trainDF["month"],trainDF["year"]=zip(*trainDF["Dates"].apply(splitDate))
  print "getting part of day"
  trainDF["a"],trainDF["b"],trainDF["c"],trainDF["d"],trainDF["e"],trainDF["f"]=zip(*trainDF["time"].apply(daytime))
  print"generating extra feature Awake"
  # trainDF["summer"],trainDF["fall"],trainDF["winter"],trainDF["spring"]=zip(*trainDF["month"].apply(getSeason))
  print"generating extra feature Awake"
  trainDF["Awake"]=trainDF["time"].apply(lambda x: 1 if (x==0 or (x>=8 and x<=23)) else 0)
  print"generating extra feature intersection"
  trainDF['intersection']=trainDF['Address'].apply(lambda x: 1 if "/" in x else 0)

  print "descretizing X"
  xcol={}
  trainDF.X=pd.cut(trainDF.X,60)
  temp=sorted(trainDF.X.unique())
  for i in temp:
    xcol[i]=temp.index(i)
  trainDF.X=trainDF.X.map(xcol)
  df=pd.DataFrame()
  df=pd.get_dummies(trainDF['X'],prefix='X')
  trainDF=pd.concat([trainDF,df],axis=1)

  print "descretizing Y"
  ycol={}
  trainDF.Y=pd.cut(trainDF.Y,100)
  temp=sorted(trainDF.Y.unique())
  for i in temp:
    ycol[i]=temp.index(i)
  trainDF.Y=trainDF.Y.map(ycol)
  df=pd.DataFrame()
  df=pd.get_dummies(trainDF['Y'],prefix='Y')
  trainDF=pd.concat([trainDF,df],axis=1)

  print"dropping unnecessary values"
  trainDF=trainDF.drop(['DayOfWeek','PdDistrict','Address','time','day','year','month','Dates','X','Y'],axis=1)
  print trainDF.head()
  end=time.time()
  return trainDF

def main():
  trainDF,testDF=getData()
  y=trainDF.Category.values
  idList=testDF.Id.tolist()
  testDF=testDF.drop(['Id'],axis=1)
  trainDF=preprocessData(trainDF)
  testDF=preprocessData(testDF)
  predicted,clf=decisionTree(trainDF,y,testDF)
  submission = pd.DataFrame(predicted,columns=clf.classes_)
  submission['Id']=idList
  cols=submission.columns.tolist()
  cols=cols[-1:]+cols[:-1]
  submission=submission[cols]
  print submission.head()
  submission.to_csv(open('RF.csv','wt'),index=False)
  print "submission file created"
  return


if __name__=="__main__":
  main()