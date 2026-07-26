#Question A1
'''import numpy as np
import pandas as pd

def load(file):
    data=pd.read_excel(file)
    features=data.iloc[:,1:4].to_numpy()
    y=data.iloc[:,4].to_numpy()

    return features,y

def rank(x):
    return np.linalg.matrix_rank(x)

def cost(x,y):
    return np.linalg.pinv(x) @ y

def main():
    file="Lab Session Data (1).xlsx"
    x,y= load(file)
    print(x)
    print(y)

    ranks=rank(x)
    print(ranks)

    costs=cost(x,y)
    print(costs)

if __name__=="__main__":
    main()'''


#Question A3
'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load(file):
    data=pd.read_excel(file,sheet_name="IRCTC Stock Price")
    return data
    
def my_mean(x):
    sum=0.0
    for i in x:
        sum+=i
    return sum/len(x)
    
def my_var(x):
    m=my_mean(x)
    sum=0.0
    for i in x:
        sum=sum+(i-m)**2
    return sum/len(x)
    
def lossprob(x):
    losses=list(filter(lambda i: i<0,x))
    return len(losses)/len(x)
    
def condition(x):
    wed=x[x["Day"]=="Wed"]
    wed_profit=wed[wed["Chg%"]>0]
    
    return len(wed_profit)/len(wed)
    
def scatterplot(x):
    plt.scatter(x["Day"],x["Chg%"])
    plt.xlabel("Day")
    plt.ylabel("Chg%")
    plt.title("Day & Chg%")
    plt.show()
    
def main():
    file="Lab Session Data (1).xlsx"
    data=load(file)
    price=data["Price"].to_numpy()
    chg=data["Chg%"].to_numpy()
    ownmean=my_mean(price)
    print(ownmean)
    ownvar=my_var(price)
    print(ownvar)
    npmean=np.mean(price)
    print(npmean)
    npvar=np.var(price)
    print(npvar)
    wed=data[data["Day"]=="Wed"]["Price"].to_numpy()
    print(wed)
    ownwedmean=my_mean(wed)
    print(ownwedmean)
    april=data[data["Month"]=="Apr"]["Price"].to_numpy()
    print(april)
    ownaprilmean=my_mean(april)
    print(ownaprilmean)
    loss=lossprob(chg)
    print(loss)
    wed_profit=len((data[(data["Day"]=="Wed") & (data["Chg%"]>0)]))/len(data)
    print(wed_profit)
    cond=condition(data)
    print(cond)
    scatterplot(data)
    
if __name__=="__main__":
    main()'''


#Question A4
'''import numpy as np
import pandas as pd

def load(file):
    data=pd.read_excel(file,sheet_name="thyroid0387_UCI")
    data=data.replace("?",np.nan)
    numeric=["age","TSH","T3","TT4","T4U","FTI","TBG"]
    for column in numeric:
        data[column]=pd.to_numeric(data[column],errors="coerce")

    return data,numeric

def datatypes(data,numeric):
    types={}
    for column in data.columns:
        values=data[column].dropna().unique()
        if column=="Record ID":
            types[column]="identifier"
        elif column in numeric:
            types[column]="numeric (ratio)"
        elif len(values)==2:
            types[column]="binary (nominal)"
        else:
            types[column]="categorical (nominal)"

    return types

def encoding_scheme(data,numeric):
    scheme={}
    for column in data.columns:
        if column in numeric or column=="Record ID":
            continue
        if len(data[column].dropna().unique())<=2:
            scheme[column]="Label encoding"
        else:
            scheme[column]="One-Hot encoding"

    return scheme

def data_range(data,numeric):
    ranges={}
    for column in numeric:
        ranges[column]=(data[column].min(),data[column].max())

    return ranges

def missing_values(data):
    return data.isna().sum()

def outliers(data,numeric):
    counts={}
    for column in numeric:
        values=data[column].dropna()
        q1=values.quantile(0.25)
        q3=values.quantile(0.75)
        iqr=q3-q1
        counts[column]=((values<q1-1.5*iqr)|(values>q3+1.5*iqr)).sum()

    return counts

def mean_variance(data,numeric):
    stats={}
    for column in numeric:
        values=data[column].dropna()
        stats[column]=(values.mean(),values.var(),values.std())

    return stats

def main():
    file="Lab Session Data (1).xlsx"
    data,numeric=load(file)
    print(data.shape)

    #datatype of each attribute
    for column,kind in datatypes(data,numeric).items():
        print(column,":",kind)

    #encoding scheme for the categorical attributes
    for column,scheme in encoding_scheme(data,numeric).items():
        print(column,":",scheme)

    #data range of the numeric attributes
    for column,limits in data_range(data,numeric).items():
        print(column,":",limits[0],"to",limits[1])

    #missing values in each attribute
    print(missing_values(data))

    #outliers in the numeric attributes
    for column,count in outliers(data,numeric).items():
        print(column,":",count)

    #mean, variance and standard deviation of the numeric attributes
    for column,stats in mean_variance(data,numeric).items():
        print(column,":",stats[0],stats[1],stats[2])

if __name__=="__main__":
    main()'''


#Question A5
'''import numpy as np
import pandas as pd

def load(file):
    data=pd.read_excel(file,sheet_name="thyroid0387_UCI")
    data=data.replace("?",np.nan)

    return data

def binary_data(data):
    columns=[]
    for column in data.columns:
        values=set(data[column].dropna().unique())
        if values<={"t","f"}:
            columns.append(column)
    binary=data[columns].replace({"t":1,"f":0})

    return binary.to_numpy()

def counts(v1,v2):
    f11=int(np.sum((v1==1)&(v2==1)))
    f00=int(np.sum((v1==0)&(v2==0)))
    f10=int(np.sum((v1==1)&(v2==0)))
    f01=int(np.sum((v1==0)&(v2==1)))

    return f11,f00,f10,f01

def jaccard(v1,v2):
    f11,f00,f10,f01=counts(v1,v2)
    if f01+f10+f11==0:
        return 0.0

    return f11/(f01+f10+f11)

def smc(v1,v2):
    f11,f00,f10,f01=counts(v1,v2)

    return (f11+f00)/(f00+f01+f10+f11)

def main():
    file="Lab Session Data (1).xlsx"
    data=load(file)

    binary=binary_data(data)
    v1=binary[0]
    v2=binary[1]
    print(v1)
    print(v2)

    f11,f00,f10,f01=counts(v1,v2)
    print(f11,f00,f10,f01)

    print(jaccard(v1,v2))
    print(smc(v1,v2))

if __name__=="__main__":
    main()'''


#Question A6
'''import numpy as np
import pandas as pd

def load(file):
    data=pd.read_excel(file,sheet_name="thyroid0387_UCI")
    data=data.replace("?",np.nan)

    return data

def full_data(data):
    numeric=["age","TSH","T3","TT4","T4U","FTI","TBG"]
    frame=data.drop(columns=["Record ID"]).copy()
    for column in numeric:
        frame[column]=pd.to_numeric(frame[column],errors="coerce")
    frame=frame.replace({"t":1,"f":0})
    frame=pd.get_dummies(frame,columns=["sex","referral source","Condition"])
    frame=frame.fillna(0)

    return frame.to_numpy(dtype=float)

def cosine(v1,v2):
    norm=np.linalg.norm(v1)*np.linalg.norm(v2)
    if norm==0:
        return 0.0

    return float(np.dot(v1,v2)/norm)

def main():
    file="Lab Session Data (1).xlsx"
    data=load(file)

    full=full_data(data)
    print(full[0])
    print(full[1])

    print(cosine(full[0],full[1]))

if __name__=="__main__":
    main()'''


#Question A7
'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load(file):
    data=pd.read_excel(file,sheet_name="thyroid0387_UCI")
    data=data.replace("?",np.nan)

    return data

def binary_data(data):
    columns=[]
    for column in data.columns:
        values=set(data[column].dropna().unique())
        if values<={"t","f"}:
            columns.append(column)
    binary=data[columns].replace({"t":1,"f":0})

    return binary.to_numpy()

def full_data(data):
    numeric=["age","TSH","T3","TT4","T4U","FTI","TBG"]
    frame=data.drop(columns=["Record ID"]).copy()
    for column in numeric:
        frame[column]=pd.to_numeric(frame[column],errors="coerce")
    frame=frame.replace({"t":1,"f":0})
    frame=pd.get_dummies(frame,columns=["sex","referral source","Condition"])
    frame=frame.fillna(0)

    return frame.to_numpy(dtype=float)

def counts(v1,v2):
    f11=int(np.sum((v1==1)&(v2==1)))
    f00=int(np.sum((v1==0)&(v2==0)))
    f10=int(np.sum((v1==1)&(v2==0)))
    f01=int(np.sum((v1==0)&(v2==1)))

    return f11,f00,f10,f01

def jaccard(v1,v2):
    f11,f00,f10,f01=counts(v1,v2)
    if f01+f10+f11==0:
        return 0.0

    return f11/(f01+f10+f11)

def smc(v1,v2):
    f11,f00,f10,f01=counts(v1,v2)

    return (f11+f00)/(f00+f01+f10+f11)

def cosine(v1,v2):
    norm=np.linalg.norm(v1)*np.linalg.norm(v2)
    if norm==0:
        return 0.0

    return float(np.dot(v1,v2)/norm)

def similarity_matrix(vectors,measure):
    n=len(vectors)
    matrix=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            matrix[i][j]=measure(vectors[i],vectors[j])

    return matrix

def heatmap(matrix,title):
    sns.heatmap(matrix,annot=True,fmt=".2f")
    plt.title(title)
    plt.show()

def main():
    file="Lab Session Data (1).xlsx"
    data=load(file)

    binary=binary_data(data)[:20]
    full=full_data(data)[:20]

    jc=similarity_matrix(binary,jaccard)
    sm=similarity_matrix(binary,smc)
    cos=similarity_matrix(full,cosine)
    print(jc)
    print(sm)
    print(cos)

    heatmap(jc,"Jaccard Coefficient")
    heatmap(sm,"Simple Matching Coefficient")
    heatmap(cos,"Cosine Similarity")

if __name__=="__main__":
    main()'''


#Question A8
'''import numpy as np
import pandas as pd

def load(file):
    data=pd.read_excel(file,sheet_name="thyroid0387_UCI")
    data=data.replace("?",np.nan)
    numeric=["age","TSH","T3","TT4","T4U","FTI","TBG"]
    for column in numeric:
        data[column]=pd.to_numeric(data[column],errors="coerce")

    return data,numeric

def has_outliers(values):
    q1=values.quantile(0.25)
    q3=values.quantile(0.75)
    iqr=q3-q1

    return bool(((values<q1-1.5*iqr)|(values>q3+1.5*iqr)).any())

def impute(data,numeric):
    filled=data.copy()
    used={}
    for column in filled.columns:
        if filled[column].isna().sum()==0:
            continue
        if column in numeric:
            values=filled[column].dropna()
            if has_outliers(values):
                filled[column]=filled[column].fillna(values.median())
                used[column]="median"
            else:
                filled[column]=filled[column].fillna(values.mean())
                used[column]="mean"
        else:
            filled[column]=filled[column].fillna(filled[column].mode()[0])
            used[column]="mode"

    return filled,used

def main():
    file="Lab Session Data (1).xlsx"
    data,numeric=load(file)

    #missing values before imputation
    print(data.isna().sum())

    filled,used=impute(data,numeric)

    #central tendency used for each attribute
    for column,method in used.items():
        print(column,":",method)

    #missing values after imputation
    print(filled.isna().sum().sum())

if __name__=="__main__":
    main()'''


#Question A9
'''import numpy as np
import pandas as pd

def load(file):
    data=pd.read_excel(file,sheet_name="thyroid0387_UCI")
    data=data.replace("?",np.nan)
    numeric=["age","TSH","T3","TT4","T4U","FTI","TBG"]
    for column in numeric:
        data[column]=pd.to_numeric(data[column],errors="coerce")

    return data,numeric

def has_outliers(values):
    q1=values.quantile(0.25)
    q3=values.quantile(0.75)
    iqr=q3-q1

    return bool(((values<q1-1.5*iqr)|(values>q3+1.5*iqr)).any())

def impute(data,numeric):
    filled=data.copy()
    for column in numeric:
        values=filled[column].dropna()
        if has_outliers(values):
            filled[column]=filled[column].fillna(values.median())
        else:
            filled[column]=filled[column].fillna(values.mean())

    return filled

def min_max(values):
    return (values-values.min())/(values.max()-values.min())

def z_score(values):
    return (values-values.mean())/values.std()

def normalize(data,numeric):
    scaled=data.copy()
    used={}
    for column in numeric:
        if has_outliers(scaled[column]):
            scaled[column]=z_score(scaled[column])
            used[column]="Z-score (outliers present)"
        else:
            scaled[column]=min_max(scaled[column])
            used[column]="Min-Max (no outliers)"

    return scaled,used

def main():
    file="Lab Session Data (1).xlsx"
    data,numeric=load(file)
    filled=impute(data,numeric)

    #range of the numeric attributes before scaling
    print(filled[numeric].describe())

    scaled,used=normalize(filled,numeric)

    #normalization technique used for each attribute
    for column,method in used.items():
        print(column,":",method)

    #range of the numeric attributes after scaling
    print(scaled[numeric].describe())

if __name__=="__main__":
    main()'''
