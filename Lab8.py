import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

def summation(weights,inputs):
    return weights[0]+np.dot(weights[1:],inputs)

def step(x):
    return 1 if x>0 else 0

def bipolarstep(x):
    return 1 if x>0 else (0 if x==0 else -1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanhfn(x):
    return np.tanh(x)

def relu(x):
    return x if x>0 else 0

def leakyrelu(x,alpha=0.01):
    return x if x>0 else alpha*x

def comparator(targets,outputs):
    return sum((t-o)**2 for t,o in zip(targets,outputs))

def trainperceptron(X,y,weights,lr,activation,maxepochs=1000,tol=0.002):
    weights=np.array(weights,dtype=float)
    errors=[]
    for epoch in range(maxepochs):
        outputs=[]
        for xi,target in zip(X,y):
            out=activation(summation(weights,xi))
            outputs.append(out)
            delta=target-out
            weights[0]+=lr*delta
            weights[1:]+=lr*delta*np.array(xi,dtype=float)
        error=comparator(y,outputs)
        errors.append(error)
        if error<=tol:
            break
    return weights,errors

def predictperceptron(X,weights,activation):
    return [activation(summation(weights,xi)) for xi in X]

def pseudoinverse(X,y):
    A=np.hstack([np.ones((len(X),1)),np.array(X,dtype=float)])
    return np.linalg.pinv(A)@np.array(y,dtype=float)

def backprop(X,y,lr=0.05,hidden=2,outputs=1,maxepochs=1000,tol=0.002):
    rng=np.random.default_rng(0)
    X=np.array(X,dtype=float)
    y=np.array(y,dtype=float).reshape(len(X),outputs)
    V=rng.uniform(-0.05,0.05,(X.shape[1],hidden))
    bv=rng.uniform(-0.05,0.05,hidden)
    W=rng.uniform(-0.05,0.05,(hidden,outputs))
    bw=rng.uniform(-0.05,0.05,outputs)
    errors=[]
    for epoch in range(maxepochs):
        h=sigmoid(X@V+bv)
        o=sigmoid(h@W+bw)
        error=np.sum((y-o)**2)
        errors.append(error)
        if error<=tol:
            break
        do=o*(1-o)*(y-o)
        dh=h*(1-h)*(do@W.T)
        W+=lr*h.T@do
        bw+=lr*do.sum(axis=0)
        V+=lr*X.T@dh
        bv+=lr*dh.sum(axis=0)
    return (V,bv,W,bw),errors

def mlpgate(X,y):
    model=MLPClassifier(hidden_layer_sizes=(4,),activation="logistic",solver="lbfgs",max_iter=5000,random_state=0)
    model.fit(X,y)
    return model.score(X,y)


if __name__ == "__main__":
    AND_X=[[0,0],[0,1],[1,0],[1,1]]
    AND_Y=[0,0,0,1]
    XOR_X=[[0,0],[0,1],[1,0],[1,1]]
    XOR_Y=[0,1,1,0]
    W0=[10,0.2,-0.75]
    LR=0.05

    w,errors=trainperceptron(AND_X,AND_Y,W0,LR,step)
    print("A2 epochs:",len(errors),"weights:",w)
    plt.figure(); plt.plot(errors); plt.xlabel("Epoch"); plt.ylabel("SSE"); plt.title("A2 AND gate - Step")
    plt.savefig("Lab8_A2.png")

    for name,fn in [("bipolar step",bipolarstep),("sigmoid",sigmoid),("relu",relu)]:
        _,e=trainperceptron(AND_X,AND_Y,W0,LR,fn)
        print("A3",name,"epochs:",len(e),"final error:",round(e[-1],4))

    rates=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    iterations=[len(trainperceptron(AND_X,AND_Y,W0,r,step)[1]) for r in rates]
    print("A4 iterations:",iterations)
    plt.figure(); plt.plot(rates,iterations,marker="o"); plt.xlabel("Learning rate"); plt.ylabel("Iterations")
    plt.title("A4 Convergence vs learning rate"); plt.savefig("Lab8_A4.png")

    for name,fn in [("step",step),("bipolar step",bipolarstep),("sigmoid",sigmoid),("relu",relu)]:
        _,e=trainperceptron(XOR_X,XOR_Y,W0,LR,fn)
        print("A5 XOR",name,"epochs:",len(e),"final error:",round(e[-1],4))

    customers=np.array([[20,6,2,386],[16,3,6,289],[27,6,2,393],[19,1,2,110],[24,4,2,280],
                        [22,1,5,167],[15,4,2,271],[18,4,2,274],[21,1,4,148],[16,2,4,198]],dtype=float)
    labels=[1,1,1,0,1,0,1,1,0,0]
    scaled=customers/customers.max(axis=0)
    cw,cerrors=trainperceptron(scaled,labels,[0.0]*5,0.5,sigmoid,maxepochs=2000)
    preds=[1 if p>=0.5 else 0 for p in predictperceptron(scaled,cw,sigmoid)]
    print("A6 epochs:",len(cerrors),"accuracy:",sum(p==t for p,t in zip(preds,labels))/len(labels))

    pw=pseudoinverse(scaled,labels)
    ppreds=[1 if v>=0.5 else 0 for v in np.hstack([np.ones((len(scaled),1)),scaled])@pw]
    print("A7 pseudo-inverse accuracy:",sum(p==t for p,t in zip(ppreds,labels))/len(labels))

    _,e=backprop(AND_X,AND_Y)
    print("A8 AND backprop epochs:",len(e),"final error:",round(e[-1],4))

    _,e=backprop(XOR_X,XOR_Y)
    print("A9 XOR backprop epochs:",len(e),"final error:",round(e[-1],4))

    twoout=lambda y:[[1,0] if v==0 else [0,1] for v in y]
    _,e=backprop(AND_X,twoout(AND_Y),outputs=2)
    print("A10 AND 2-output epochs:",len(e),"final error:",round(e[-1],4))
    _,e=backprop(XOR_X,twoout(XOR_Y),outputs=2)
    print("A10 XOR 2-output epochs:",len(e),"final error:",round(e[-1],4))

    print("A11 MLP AND accuracy:",mlpgate(AND_X,AND_Y))
    print("A11 MLP XOR accuracy:",mlpgate(XOR_X,XOR_Y))

    data=pd.read_excel("Lab Session Data (1).xlsx",sheet_name="marketing_campaign")
    data=data.drop(columns=["ID","Dt_Customer","Z_CostContact","Z_Revenue"])
    data["Education"]=pd.factorize(data["Education"])[0]
    data["Marital_Status"]=pd.factorize(data["Marital_Status"])[0]
    data["Income"]=data["Income"].fillna(data["Income"].median())
    y=data["Response"]
    X=data.drop(columns=["Response"])
    X=(X-X.mean())/X.std()
    project=MLPClassifier(hidden_layer_sizes=(32,16),max_iter=500,random_state=0).fit(X,y)
    print("A12 project MLP accuracy:",project.score(X,y))
    plt.show()
