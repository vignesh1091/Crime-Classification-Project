import pandas as pd
import time
from datetime import datetime
from sklearn import metrics
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

def getData():
  print "reading file ..."
  data=pd.read_csv('../train.csv')
  return data


def nBModelling(trainDF,y):
  start=time.time()
  print "#"*70
  print "Using Naive Bayes classifier"
  print "#"*70
  trainingVectors=trainDF.as_matrix()
  clf = MultinomialNB()
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


def svm(trainDF,y):
  start=time.time()
  print "#"*70
  print "Using Support Vector Machines Classifier"
  print "#"*70
  trainingVectors=trainDF.as_matrix()
  clf =OneVsRestClassifier(SVC(C=0.5, kernel = 'linear', gamma='auto', verbose= False, probability=False))
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
  clf=RandomForestClassifier(n_estimators=2,criterion="gini")
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

def splitDate(x):
  dateObject=datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
  time=dateObject.hour
  day=dateObject.day
  month=dateObject.month
  year=dateObject.year
  return time,day,month,year

def getSeason(x):
    summer=0
    fall=0
    winter=0
    spring=0
    if (x in [5, 6, 7]):
        summer=1
    if (x in [8, 9, 10]):
        fall=1
    if (x in [11, 0, 1]):
        winter=1
    if (x in [2, 3, 4]):
        spring=1
    return summer, fall, winter, spring

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

def preprocessData(trainDF):
  print "pre processing data ..."
  start=time.time()
  y=trainDF.Category.values
  trainDF=trainDF.drop(['Descript','Resolution'],axis=1)
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

  # df=pd.DataFrame()
  # print "seperating year"
  # df=pd.get_dummies(trainDF['year'],prefix='yr')
  # trainDF=pd.concat([trainDF,df],axis=1)

  print "getting parts of day"
  trainDF["a"],trainDF["b"],trainDF["c"],trainDF["d"],trainDF["e"],trainDF["f"]=zip(*trainDF["time"].apply(daytime))
  print"generating extra feature Awake"
  trainDF["Awake"]=trainDF["time"].apply(lambda x: 1 if (x==0 or (x>=8 and x<=23)) else 0)
  # trainDF["half"]=trainDF["day"].apply(lambda x: 1 if (x>=15) else 0)
  print"generating extra feature intersection"
  trainDF['intersection']=trainDF['Address'].apply(lambda x: 1 if "/" in x else 0)

  cat={}
  temp=sorted(trainDF.Category.unique().tolist())
  for i in temp:
    cat[i]=temp.index(i)
  trainDF.Category=trainDF.Category.map(cat)

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
  trainDF.Y=pd.cut(trainDF.Y,50)
  temp=sorted(trainDF.Y.unique())
  for i in temp:
    ycol[i]=temp.index(i)
  trainDF.Y=trainDF.Y.map(ycol)
  df=pd.DataFrame()
  df=pd.get_dummies(trainDF['Y'],prefix='Y')
  trainDF=pd.concat([trainDF,df],axis=1)
  # print trainDF.head()
  print"dropping unnecessary values"
  trainDF=trainDF.drop(['Dates','Address'],axis=1)
  # print trainDF.head()
  # corr=trainDF.corr(method="pearson")
  # print corr
  trainDF=trainDF.drop(['Category','DayOfWeek','PdDistrict','time','day','month','X','Y','year','Awake'],axis=1)
  print trainDF.head()
  print trainDF.columns.tolist()
  end=time.time()
  return trainDF,y

def main():
  trainDF=getData()
  trainDF,y=preprocessData(trainDF)
  nBModelling(trainDF,y)
  decisionTree(trainDF,y)
  randomForest(trainDF,y)
  # svm(trainDF,y)
  return
#trainDF=trainDF.drop(['Category','DayOfWeek','PdDistrict','time','day','year','month','X','Y'],axis=1)
#0.224845082678
#0.231554275445

  

if __name__=="__main__":
  main()
