import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV

def probabilities(values):
    counts={}
    for v in values:
        counts[v]=counts.get(v,0)+1
    return [c/len(values) for c in counts.values()]

def entropy(values):
    return -sum(p*math.log2(p) for p in probabilities(values))

def gini(values):
    return 1-sum(p*p for p in probabilities(values))

def equalwidthbinning(values,bins=4):
    low,high=min(values),max(values)
    width=(high-low)/bins
    if width==0:
        return [0 for v in values]
    return [min(int((v-low)/width),bins-1) for v in values]

def equalfrequencybinning(values,bins=4):
    order=sorted(range(len(values)),key=lambda i:values[i])
    result=[0]*len(values)
    size=len(values)/bins
    for rank,i in enumerate(order):
        result[i]=min(int(rank/size),bins-1)
    return result

def binning(values,bins=4,method="width"):
    if method=="width":
        return equalwidthbinning(values,bins)
    if method=="frequency":
        return equalfrequencybinning(values,bins)
    raise ValueError("unknown binning method: "+str(method))

def informationgain(feature,labels):
    total=entropy(labels)
    groups={}
    for f,l in zip(feature,labels):
        groups.setdefault(f,[]).append(l)
    remainder=sum(len(g)/len(labels)*entropy(g) for g in groups.values())
    return total-remainder

def rootnode(features,labels):
    gains={name:informationgain(values,labels) for name,values in features.items()}
    best=max(gains,key=gains.get)
    return best,gains

def majority(labels):
    counts={}
    for l in labels:
        counts[l]=counts.get(l,0)+1
    return max(counts,key=counts.get)

def buildtree(features,labels,depth=0,maxdepth=3):
    if len(set(labels))==1 or len(features)==0 or depth==maxdepth:
        return majority(labels)
    best,gains=rootnode(features,labels)
    if gains[best]==0:
        return majority(labels)
    node={"feature":best,"children":{},"default":majority(labels)}
    for value in set(features[best]):
        rows=[i for i in range(len(labels)) if features[best][i]==value]
        subfeatures={n:[v[i] for i in rows] for n,v in features.items() if n!=best}
        sublabels=[labels[i] for i in rows]
        node["children"][value]=buildtree(subfeatures,sublabels,depth+1,maxdepth)
    return node

def predict(tree,sample):
    while isinstance(tree,dict):
        value=sample[tree["feature"]]
        if value not in tree["children"]:
            return tree["default"]
        tree=tree["children"][value]
    return tree

def treetext(tree,indent=""):
    if not isinstance(tree,dict):
        return indent+"-> class "+str(tree)+"\n"
    text=""
    for value,child in tree["children"].items():
        text+=indent+tree["feature"]+" = bin "+str(value)+"\n"
        text+=treetext(child,indent+"    ")
    return text

def preparedata(data):
    new=data.drop(columns=["ID","Dt_Customer","Z_CostContact","Z_Revenue"])
    new["Education"]=pd.factorize(new["Education"])[0]
    new["Marital_Status"]=pd.factorize(new["Marital_Status"])[0]
    new["Income"]=new["Income"].fillna(new["Income"].median())
    labels=list(new["Response"])
    features=new.drop(columns=["Response"])
    return features,labels

def binfeatures(features,bins=4,method="width"):
    result={}
    for name in features.columns:
        values=list(features[name])
        result[name]=values if len(set(values))<=bins else binning(values,bins,method)
    return result


if __name__ == "__main__":
    data=pd.read_excel("Lab Session Data (1).xlsx",sheet_name="marketing_campaign")
    features,labels=preparedata(data)

    print("A1 entropy:",entropy(labels))
    print("A2 gini:",gini(labels))

    binned=binfeatures(features)
    best,gains=rootnode(binned,labels)
    print("A3 root node:",best)

    print("A4 bins:",binning(list(features["Income"]),4,"frequency")[:10])

    tree=buildtree(binned,labels,maxdepth=3)
    print("A5 tree:\n"+treetext(tree))

    clf=DecisionTreeClassifier(max_depth=3,criterion="entropy").fit(features,labels)
    plt.figure(figsize=(16,8))
    plot_tree(clf,feature_names=list(features.columns),class_names=["0","1"],filled=True)
    plt.savefig("Lab7_tree.png")

    X=features[["Income","MntWines"]]
    clf2=DecisionTreeClassifier(max_depth=3).fit(X,labels)
    xs=[X["Income"].min()+i*(X["Income"].max()-X["Income"].min())/200 for i in range(201)]
    ys=[X["MntWines"].min()+i*(X["MntWines"].max()-X["MntWines"].min())/200 for i in range(201)]
    grid=pd.DataFrame([[x,y] for y in ys for x in xs],columns=["Income","MntWines"])
    Z=[[0]*201 for i in range(201)]
    for i,p in enumerate(clf2.predict(grid)):
        Z[i//201][i%201]=p
    plt.figure()
    plt.contourf(xs,ys,Z,alpha=0.3)
    plt.scatter(X["Income"],X["MntWines"],c=labels,s=5)
    plt.xlabel("Income"); plt.ylabel("MntWines"); plt.title("A7 Decision boundary")
    plt.savefig("Lab7_boundary.png")

    params={"max_depth":[2,3,5,8,None],"criterion":["gini","entropy"],"min_samples_leaf":[1,5,20]}
    search=GridSearchCV(DecisionTreeClassifier(),params,cv=5).fit(features,labels)
    print("A8 best params:",search.best_params_)
    plt.show()
