import time
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import Lab5

def labelencoding(values):
    categories=sorted(set(values))
    return [categories.index(v) for v in values]

def onehot(values):
    categories=sorted(set(values))
    return [[1 if v==c else 0 for c in categories] for v in values],categories

def impute(values,method="median"):
    series=pd.Series(values)
    if method=="mean":
        filler=series.mean()
    elif method=="median":
        filler=series.median()
    elif method=="mode":
        filler=series.mode()[0]
    else:
        raise ValueError("unknown method: "+str(method))
    return list(series.fillna(filler))

def preparedata(data,method="median"):
    new=data.drop(columns=["ID","Dt_Customer","Z_CostContact","Z_Revenue"])
    new["Education"]=labelencoding(list(new["Education"]))
    matrix,cols=onehot(list(new["Marital_Status"]))
    new=new.drop(columns=["Marital_Status"])
    for i,c in enumerate(cols):
        new["Marital_"+c]=[row[i] for row in matrix]
    for column in new.columns:
        if new[column].isna().sum()>0:
            new[column]=impute(list(new[column]),method)
    labels=list(new["Response"])
    features=new.drop(columns=["Response"])
    return features.values.tolist(),labels

def distance(a,b,metric="euclidean",p=3):
    power={"manhattan":1,"euclidean":2,"minkowski":p}[metric]
    return sum(abs(x-y)**power for x,y in zip(a,b))**(1/power)

def bubblesort(items):
    items=list(items)
    for i in range(len(items)):
        for j in range(len(items)-1-i):
            if items[j+1]<items[j]:
                items[j],items[j+1]=items[j+1],items[j]
    return items

def insertionsort(items):
    items=list(items)
    for i in range(1,len(items)):
        current=items[i]
        j=i-1
        while j>=0 and current<items[j]:
            items[j+1]=items[j]
            j-=1
        items[j+1]=current
    return items

def mergesort(items):
    if len(items)<=1:
        return list(items)
    middle=len(items)//2
    left,right=mergesort(items[:middle]),mergesort(items[middle:])
    merged=[]
    while left and right:
        merged.append(left.pop(0) if left[0]<=right[0] else right.pop(0))
    return merged+left+right

def sortdistances(items,algorithm="merge"):
    return {"bubble":bubblesort,"insertion":insertionsort,"merge":mergesort}[algorithm](items)

def neighbours(train,labels,x,k=3,metric="euclidean",algorithm="merge"):
    items=[(distance(train[i],x,metric),i,labels[i]) for i in range(len(train))]
    return sortdistances(items,algorithm)[:k]

def vote(neigh,weighted=False):
    totals={}
    for d,i,label in neigh:
        totals[label]=totals.get(label,0)+(1/(d+1e-9) if weighted else 1)
    best=max(totals.values())
    winners=[label for label in totals if totals[label]==best]
    for d,i,label in neigh:
        if label in winners:
            return label

class KNN:
    def __init__(self,k=3,metric="euclidean",algorithm="merge",weighted=False):
        self.k,self.metric,self.algorithm,self.weighted=k,metric,algorithm,weighted

    def fit(self,train,labels):
        self.train,self.labels=train,labels
        return self

    def predict(self,test):
        return [vote(neighbours(self.train,self.labels,x,self.k,self.metric,self.algorithm),self.weighted) for x in test]

    def score(self,test,labels):
        predictions=self.predict(test)
        return sum(p==l for p,l in zip(predictions,labels))/len(labels)

def accuracyvsk(train,trainlabels,test,testlabels,ks,weighted=False):
    sorted_lists=[neighbours(train,trainlabels,x,max(ks)) for x in test]
    own=[np.mean([vote(n[:k],weighted)==l for n,l in zip(sorted_lists,testlabels)]) for k in ks]
    package=[KNeighborsClassifier(n_neighbors=k,weights="distance" if weighted else "uniform").fit(train,trainlabels).score(test,testlabels) for k in ks]
    return own,package

def genaiknn(train,trainlabels,test,k=3):
    train,test,trainlabels=np.array(train),np.array(test),np.array(trainlabels)
    dists=np.sqrt(((test[:,None,:]-train[None,:,:])**2).sum(axis=2))
    nearest=trainlabels[np.argsort(dists,axis=1,kind="stable")[:,:k]]
    return [np.bincount(row).argmax() for row in nearest]

def evaluate(name,predictfn,test,testlabels,runs=10):
    start=time.perf_counter()
    for i in range(runs):
        predictions=predictfn()
    elapsed=(time.perf_counter()-start)/runs
    return {"version":name,"accuracy":accuracy_score(testlabels,predictions),
            "precision":precision_score(testlabels,predictions,zero_division=0),
            "recall":recall_score(testlabels,predictions,zero_division=0),
            "f1":f1_score(testlabels,predictions,zero_division=0),"time":elapsed}

class Tests(unittest.TestCase):
    def test_labelencoding(self):
        self.assertEqual(labelencoding(["b","a","b"]),[1,0,1])
        self.assertEqual(Lab5.labelencoding(["b","a","b"])[0],[1,0,1])

    def test_onehot(self):
        self.assertEqual(onehot(["x","y"])[0],[[1,0],[0,1]])
        self.assertEqual(Lab5.onehot(["x","y"])[0],[[1,0],[0,1]])

    def test_impute(self):
        self.assertEqual(impute([1,None,3],"mean"),[1,2,3])
        self.assertEqual(impute([1,None,3,10],"median"),[1,3,3,10])
        self.assertEqual(impute([1,None,1,2],"mode"),[1,1,1,2])
        self.assertEqual(Lab5.impute([1,None,3],"mean"),[1,2,3])

    def test_distance(self):
        self.assertEqual(distance([0,0],[3,4]),5)
        self.assertEqual(distance([0,0],[3,4],"manhattan"),7)
        self.assertEqual(Lab5.calcdistance([0,0],[3,4]),5)
        with self.assertRaises(KeyError):
            distance([0],[1],"cosine")

    def test_sorting(self):
        items=[(3,0,1),(1,1,0),(2,2,1),(1,3,1)]
        expected=[(1,1,0),(1,3,1),(2,2,1),(3,0,1)]
        for algorithm in ["bubble","insertion","merge"]:
            self.assertEqual(sortdistances(items,algorithm),expected)
            self.assertEqual(Lab5.sortdistances([list(i) for i in items],algorithm),[list(e) for e in expected])

    def test_neighbours_tie(self):
        train=[[1],[1],[5]]
        self.assertEqual([n[1] for n in neighbours(train,[0,1,0],[1],2)],[0,1])

    def test_vote_tie(self):
        self.assertEqual(vote([(1,0,"a"),(2,1,"b")]),"a")
        self.assertEqual(vote([(1,0,"a"),(0.5,1,"b"),(3,2,"a")],weighted=True),"b")
        self.assertEqual(Lab5.majorityvote([[1,0,"a"],[2,1,"b"]])[0],"a")

    def test_knn(self):
        model=KNN(k=1).fit([[0],[10]],[0,1])
        self.assertEqual(model.predict([[1],[9]]),[0,1])
        self.assertEqual(model.score([[1],[9]],[0,1]),1)
        self.assertEqual(Lab5.classify([[0],[10]],[0,1],[1],k=1)[0],0)

    def test_genaiknn(self):
        self.assertEqual(genaiknn([[0],[10]],[0,1],[[1],[9]],k=1),[0,1])


if __name__ == "__main__":
    data=pd.read_excel("Lab Session Data (1).xlsx",sheet_name="marketing_campaign")
    features,labels=preparedata(data)
    xtrain,xtest,ytrain,ytest=train_test_split(features,labels,test_size=0.3,random_state=0)

    unittest.main(argv=["Lab6"],exit=False,verbosity=1)

    ks=list(range(1,16))
    own,package=accuracyvsk(xtrain,ytrain,xtest,ytest,ks)
    weightedown,weightedpackage=accuracyvsk(xtrain,ytrain,xtest,ytest,ks,weighted=True)
    plt.plot(ks,own,"o-",label="own kNN")
    plt.plot(ks,package,"s--",label="sklearn kNN")
    plt.plot(ks,weightedown,"^-",label="own weighted kNN")
    plt.plot(ks,weightedpackage,"v--",label="sklearn weighted kNN")
    plt.xlabel("k"); plt.ylabel("test accuracy"); plt.legend()
    plt.savefig("Lab6_accuracy_vs_k.png")

    own=KNN(k=3).fit(xtrain,ytrain)
    sk=KNeighborsClassifier(n_neighbors=3).fit(xtrain,ytrain)
    results=pd.DataFrame([
        evaluate("own",lambda:own.predict(xtest),xtest,ytest),
        evaluate("sklearn",lambda:list(sk.predict(xtest)),xtest,ytest),
        evaluate("genai",lambda:genaiknn(xtrain,ytrain,xtest,3),xtest,ytest)])
    print(results.to_string(index=False))
    plt.show()
