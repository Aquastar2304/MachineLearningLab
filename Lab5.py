# Lab 05 - A1
# Modular design for a k-Nearest Neighbours classifier.
# Dataset: "Lab Session Data (1).xlsx", sheet "marketing_campaign".
# Class label: Response (already binary, 0 / 1).

import pandas as pd

# ---------- A1 (a) Encoding : convert categorical data ----------
# Reused from Lab3.py

def labelencoding(values):
    """Map each distinct category to an integer 0..n-1 (for ordinal features)."""
    categories=sorted(set(values))
    mapping={category:index for index,category in enumerate(categories)}
    return [mapping[v] for v in values], mapping

def onehot(values):
    """One column per category, 1 in the matching column (for nominal features)."""
    categories=sorted(set(values))
    encoded=[]
    for v in values:
        row=[1 if v==category else 0 for category in categories]
        encoded.append(row)
    return encoded,categories

def encodedataset(data):
    """Education -> label encoding (ordinal), Marital_Status -> one-hot (nominal)."""
    new=data.drop(columns=["Dt_Customer"])          # date column, not numeric
    new["Education"],edumap=labelencoding(list(new["Education"]))
    matrix,cols=onehot(list(new["Marital_Status"]))
    new=new.drop(columns=["Marital_Status"])
    for i,c in enumerate(cols):
        new["Marital_"+c]=[row[i] for row in matrix]
    return new,edumap,cols

# ---------- A1 (b) Data imputation : fill missing values ----------

def mean(values):
    present=[v for v in values if not pd.isna(v)]
    return sum(present)/len(present)

def median(values):
    present=sorted(v for v in values if not pd.isna(v))
    middle=len(present)//2
    if len(present)%2==1:
        return present[middle]
    return (present[middle-1]+present[middle])/2

def mode(values):
    present=[v for v in values if not pd.isna(v)]
    counts={}
    for v in present:
        counts[v]=counts.get(v,0)+1
    best=max(counts.values())
    return sorted(v for v in counts if counts[v]==best)[0]     # smallest value wins a tie

def impute(values,method="median"):
    """Replace missing values in one column with the chosen central tendency."""
    if method=="mean":
        filler=mean(values)
    elif method=="median":
        filler=median(values)
    elif method=="mode":
        filler=mode(values)
    else:
        raise ValueError("unknown imputation method: "+str(method))
    return [filler if pd.isna(v) else v for v in values]

def imputedataset(data,method="median"):
    """Impute every column that has missing values."""
    new=data.copy()
    for column in new.columns:
        if new[column].isna().sum()>0:
            new[column]=impute(list(new[column]),method)
    return new

# ---------- A1 (c) Distance calculation : metric is a config parameter ----------

def minkowsi(a,b,p):
    total=0
    for i in range(len(a)):
        total+=abs(a[i]-b[i])**p
    return total**(1/p)

def calcdistance(a,b,metric="euclidean",p=3):
    """metric = "manhattan" (p=1), "euclidean" (p=2) or "minkowski" (any p)."""
    if metric=="manhattan":
        return minkowsi(a,b,1)
    if metric=="euclidean":
        return minkowsi(a,b,2)
    if metric=="minkowski":
        return minkowsi(a,b,p)
    raise ValueError("unknown metric: "+str(metric))

# ---------- A1 (d) Sorting : 3 algorithms, selected by a config parameter ----------
# Every item sorted below is [distance, trainindex, label].

def before(x,y):
    """True if x belongs before y : smaller distance first, ties broken by train index."""
    if x[0]!=y[0]:
        return x[0]<y[0]
    return x[1]<y[1]

def bubblesort(items):
    items=[list(item) for item in items]
    n=len(items)
    for i in range(n):
        for j in range(n-1-i):
            if before(items[j+1],items[j]):
                items[j],items[j+1]=items[j+1],items[j]
    return items

def insertionsort(items):
    items=[list(item) for item in items]
    for i in range(1,len(items)):
        current=items[i]
        j=i-1
        while j>=0 and before(current,items[j]):
            items[j+1]=items[j]
            j-=1
        items[j+1]=current
    return items

def mergesort(items):
    if len(items)<=1:
        return [list(item) for item in items]
    middle=len(items)//2
    left=mergesort(items[:middle])
    right=mergesort(items[middle:])
    merged=[]
    i=j=0
    while i<len(left) and j<len(right):
        if before(right[j],left[i]):
            merged.append(right[j])
            j+=1
        else:
            merged.append(left[i])
            i+=1
    return merged+left[i:]+right[j:]

def sortdistances(items,algorithm="merge"):
    if algorithm=="bubble":
        return bubblesort(items)
    if algorithm=="insertion":
        return insertionsort(items)
    if algorithm=="merge":
        return mergesort(items)
    raise ValueError("unknown sorting algorithm: "+str(algorithm))

# ---------- A1 (e) Identify the k nearest neighbours ----------

def getneighbours(traindata,trainlabels,testvector,k=3,metric="euclidean",p=3,algorithm="merge"):
    """Return the k nearest training patterns as [distance, trainindex, label].
    Tie breaking: patterns at an equal distance are ordered by their training
    index (see before()), so the same k neighbours are always selected."""
    items=[]
    for i in range(len(traindata)):
        distance=calcdistance(traindata[i],testvector,metric,p)
        items.append([distance,i,trainlabels[i]])
    return sortdistances(items,algorithm)[:k]

# ---------- A1 (f) Class evaluation and assignment : majority vote ----------

def majorityvote(neighbours):
    """Assign the class held by most neighbours.
    Tie breaking: if two classes have the same count, the class of the nearest
    neighbour among the tied classes wins (neighbours are already sorted)."""
    counts={}
    for neighbour in neighbours:
        counts[neighbour[2]]=counts.get(neighbour[2],0)+1
    best=max(counts.values())
    winners=[label for label in counts if counts[label]==best]
    if len(winners)==1:
        return winners[0],counts
    for neighbour in neighbours:
        if neighbour[2] in winners:
            return neighbour[2],counts

def classify(traindata,trainlabels,testvector,k=3,metric="euclidean",p=3,algorithm="merge"):
    """Full kNN pipeline for one test pattern: distances -> sort -> neighbours -> vote."""
    neighbours=getneighbours(traindata,trainlabels,testvector,k,metric,p,algorithm)
    label,counts=majorityvote(neighbours)
    return label,neighbours,counts

# ---------- Put (a) and (b) together into a feature matrix ----------

def preparedata(data,method="median"):
    """Encode, drop non-informative columns, impute, split off the class label."""
    encoded,edumap,maritalcols=encodedataset(data)
    encoded=encoded.drop(columns=["ID","Z_CostContact","Z_Revenue"])   # id + 2 constant columns
    encoded=imputedataset(encoded,method)
    labels=list(encoded["Response"])
    features=encoded.drop(columns=["Response"])
    return features.values.tolist(),labels,list(features.columns),edumap,maritalcols


if __name__ == "__main__":
    data=pd.read_excel("Lab Session Data (1).xlsx",sheet_name="marketing_campaign")
    print("raw shape:",data.shape)
    print("missing values before imputation:")
    print(data.isna().sum()[lambda s:s>0])

    #A1 (a) + (b)
    features,labels,columns,edumap,maritalcols=preparedata(data,"median")
    print("education label map:",edumap)
    print("marital one-hot columns:",maritalcols)
    print("feature matrix:",len(features),"rows x",len(columns),"columns")
    print("missing values after imputation:",sum(1 for row in features for v in row if pd.isna(v)))
    print("class distribution:",{c:labels.count(c) for c in sorted(set(labels))})

    #A1 (c) distance with each metric
    veca=features[0]
    vecb=features[1]
    print("manhattan:",calcdistance(veca,vecb,"manhattan"))
    print("euclidean:",calcdistance(veca,vecb,"euclidean"))
    print("minkowski p=3:",calcdistance(veca,vecb,"minkowski",3))

    # small train / test slice so bubble and insertion sort stay quick
    traindata=features[:200]
    trainlabels=labels[:200]
    testdata=features[200:205]
    testlabels=labels[200:205]

    #A1 (d) the 3 sorting algorithms must give the same ordering
    items=[[calcdistance(traindata[i],testdata[0],"euclidean"),i,trainlabels[i]]
           for i in range(len(traindata))]
    for algorithm in ["bubble","insertion","merge"]:
        result=sortdistances(items,algorithm)
        print(algorithm,"sort - 3 smallest distances:",[round(r[0],3) for r in result[:3]],
              "same as merge:",result==sortdistances(items,"merge"))

    #A1 (e) + (f)
    for i in range(len(testdata)):
        label,neighbours,counts=classify(traindata,trainlabels,testdata[i],k=3)
        print("test",i,"neighbours (distance,index,label):",
              [[round(n[0],3),n[1],n[2]] for n in neighbours],
              "votes:",counts,"predicted:",label,"actual:",testlabels[i])
