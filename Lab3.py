import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial.distance
import numpy as np

#A2
def labelencoding(values):
    categories=sorted(set(values))
    mapping={category:index for index,category in enumerate(categories)}
    return [mapping[v] for v in values], mapping
    
def onehot(values):
    categories=sorted(set(values))
    encoded=[]
    for v in values:
        row=[1 if v==category else 0 for category in categories]
        encoded.append(row)
    return encoded,categories
    
#A3
def encodedataset(data):
    new=data.drop(columns=["Dt_Customer"])          
    new["Education"],edumap=labelencoding(list(new["Education"]))
    matrix,cols=onehot(list(new["Marital_Status"]))
    new=new.drop(columns=["Marital_Status"])
    for i,c in enumerate(cols):
        new["Marital_"+c]=[row[i] for row in matrix]
    return new,edumap,cols

#A4
def minkowsi(a,b,p):
    total=0
    for i in range(len(a)):
        total+=abs(a[i]-b[i])**p
    return total**(1/p)

#A5
def distance(a,b,maxp):
    plist=[i for i in range(1,maxp+1)]
    dlist=[minkowsi(a,b,p) for p in plist]
    return plist,dlist

#A6
def compare(a,b,maxp):
    rows=[]
    for i in range(1,maxp+1):
        mine=minkowsi(a,b,i)
        pkg=scipy.spatial.distance.minkowski(a,b,i)
        rows.append([i,mine,pkg,abs(mine-pkg)])
    return rows

#A7
def dotproduct(a,b):
    total=0
    for i in range(len(a)):
        total+=a[i]*b[i]
    return total
    
def norm(a):
    return dotproduct(a,a)**0.5             

#A8
def mean(data):
    return sum(data)/len(data)

def variance(data):
    m=mean(data)
    return sum((x-m)**2 for x in data)/len(data)
    
def std(data):
    return variance(data)**0.5
    
def matrixstats(matrix):
    means=[]
    variances=[]
    stds=[]
    for i in range(len(matrix[0])):
        column=[row[i] for row in matrix]
        means.append(mean(column))
        variances.append(variance(column))
        stds.append(std(column))
    return means,variances,stds

#A9
def comparestats(matrix):
    mymeans,myvars,mystds=matrixstats(matrix)
    npmeans=np.mean(matrix,axis=0)          
    npstds=np.std(matrix,axis=0)
    rows=[]
    for i in range(len(mymeans)):
        rows.append([mymeans[i],npmeans[i],abs(mymeans[i]-npmeans[i]),
                     mystds[i],npstds[i],abs(mystds[i]-npstds[i])])
    return rows

#A10
def histdata(feature,buckets):
    counts,edges=np.histogram(feature,bins=buckets)     
    return counts,edges,mean(feature),variance(feature)

if __name__ == "__main__":
    education = ["Graduation", "PhD", "Master", "Graduation", "Basic"]
    labels,labelmap=labelencoding(education)
    print(labels)
    print(labelmap)
    onehotmatrix,col=onehot(education)
    print(onehotmatrix)
    print(col)

    data=pd.read_excel("Lab Session Data (1).xlsx",sheet_name="marketing_campaign")
    encoded,edumap,maritalcols=encodedataset(data)
    print(edumap)
    print(maritalcols)
    print(data.shape)
    print(encoded.shape)
    print(encoded.head())

    v1=[1,2,3]
    v2=[4,6,8]
    print(minkowsi(v1,v2,1))
    print(minkowsi(v1,v2,2))
    print(minkowsi(v1,v2,3))

    vectors=encoded.fillna(0)
    veca=list(vectors.iloc[0])
    vecb=list(vectors.iloc[1])
    plist,dlist=distance(veca,vecb,10)
    for p,d in zip(plist,dlist):
        print("p=",p,"d=",d)
    plt.plot(plist,dlist,marker="o")
    plt.show()

    for row in compare(veca,vecb,10):
        print("p=",row[0],"mine=",row[1],"scipy=",row[2],"diff=",row[3])

    print(dotproduct(veca,vecb))
    print(np.dot(veca,vecb))
    print(norm(veca))
    print(np.linalg.norm(veca))
    print(norm(vecb))
    print(np.linalg.norm(vecb))

    matrix=vectors.values.tolist()
    means,variances,stds=matrixstats(matrix)
    print(means)
    print(variances)
    print(stds)

    #A9
    for name,row in zip(vectors.columns,comparestats(matrix)):
        print(name,"| mean mine=",row[0],"numpy=",row[1],"diff=",row[2],
              "| std mine=",row[3],"numpy=",row[4],"diff=",row[5])

    #A10
    feature=list(vectors["MntWines"])
    counts,edges,fmean,fvar=histdata(feature,10)
    for i in range(len(counts)):
        print("range",edges[i],"to",edges[i+1],"count=",counts[i])
    print("mean=",fmean,"variance=",fvar)
    plt.hist(feature,bins=10)
    plt.xlabel("MntWines")
    plt.ylabel("Frequency")
    plt.show()