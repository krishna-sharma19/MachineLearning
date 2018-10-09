import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import scipy.sparse as sparse
from sklearn.metrics import mean_squared_error
from math import sqrt

def coordinateDescent(X,y, origW):

    def calculatePR(w, wPred):
        tp, tn, fp, fn = 0, 0, 0, 0
        for orig, pred in zip(w, wPred):
            if pred != 0:
                if orig != 0:
                    tp += 1
                else:
                    fp += 1
            else:
                if orig != 0:
                    fn += 1
                else:
                    tn += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return precision, recall

    def plotPR(p,r,l):
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()

        ax1.plot(l, p, 'g-')
        ax2.plot(l, r, 'b-')

        ax1.set_xlabel('Lambda')
        ax1.set_ylabel('Precision', color='g')
        ax2.set_ylabel('Recall', color='b')

        plt.show()

    d,n = X.shape
    delta = 1
    step = 0
    objective = []

    lembda = calLambdaMax(X,y)
    r,rprev,b,bprev = 0,0,0,0 #TODO
    wprev = np.zeros((d, 1))
    a = 2 * np.sum(X ** 2, axis=1)
    c = np.zeros((d, 1))
    w=np.ones((d,1))
    precision, recall,lembdas = [],[],[]

    # while abs((w-wprev).sum()) > delta:
    #     step += 1
    #     if step%10==0:
    #         lembda /= 2
    #

    while step < 10:
        step += 1
        lembda /= 2
        rprev = y - np.matmul(np.transpose(X),w) - b
        bprev = b
        b += np.sum(rprev)/ n
        r = rprev + bprev - b
        wprev = w.copy()

        prec,rec = calculatePR(origW,w)
        precision.append(prec)
        recall.append(rec)
        lembdas.append(lembda)

        for k in range(d):
            tempW = w.copy()

            c[k] = 2* np.dot(X[k],r) + 2*np.dot(X[k],X[k])*w[k]

            ck = c[k]
            ak = a[k]

            if ck < -lembda:
                w[k] = (ck+lembda)/ak
            elif ck > lembda:
                w[k] = (ck - lembda)/ak
            else:
                w[k]  = 0

            currR = r.sum()
            objective.append(currR)
            r += np.matmul(np.transpose(X),w) - np.matmul(np.transpose(X),tempW)
            newR = r.sum()

            if abs(currR-newR) < 0.01:
                break

    print(precision, recall, lembdas)
    plotPR(precision[::-1], recall[::-1], lembdas)
    origW = [True if w >0 else  False for w in origW ]
    predW = [True if w >0 else  False for w in w ]

    precision, recall, _ = precision_recall_curve(origW, predW)
    print(recall)
    return w


def coordinateDescentSparse(X,y, origW):

    def calculatePR(w, wPred):
        tp, tn, fp, fn = 0, 0, 0, 0
        for orig, pred in zip(w, wPred):
            if pred != 0:
                if orig != 0:
                    tp += 1
                else:
                    fp += 1
            else:
                if orig != 0:
                    fn += 1
                else:
                    tn += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return precision, recall

    def plotPR(p,r,l):
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()

        ax1.plot(l, p, 'g-')
        ax2.plot(l, r, 'b-')

        ax1.set_xlabel('Lambda')
        ax1.set_ylabel('Precision', color='g')
        ax2.set_ylabel('Recall', color='b')

        plt.show()

    d,n = X.shape
    delta = 1
    step = 0
    objective = []

    lembda = calLambdaMaxS(X,y)
    print(lembda)
    r,rprev,b,bprev = 0,0,0,0 #TODO
    wprev = np.zeros((d, 1))
    Xnew = X.copy()
    Xnew.data **= 2
    a = 2 * Xnew.sum(axis=1)
    c = np.zeros((d, 1))
    w = np.ones((d,1))
    precision, recall,lembdas = [],[],[]

    # while abs((w-wprev).sum()) > delta:
    #     step += 1
    #     if step%10==0:
    #         lembda /= 2
    #

    while step < 10:
        step += 1
        lembda /= 2
        rprev = y - X.transpose().dot(w) - b
        bprev = b
        b += rprev.sum()/ n
        r = rprev + bprev - b
        wprev = w.copy()

        for k in range(d):
            tempW = w.copy()
            c[k] = 2* (X[k].dot(sparse.csr_matrix(r) )).data[0] + 2* X[k].dot(X[k].transpose()).data[0] * w[k]

            ck = c[k]
            ak = a[k]

            if ck < -lembda:
                w[k] = (ck+lembda)/ak
            elif ck > lembda:
                w[k] = (ck - lembda)/ak
            else:
                w[k] = 0

            currR = r.sum()
            objective.append(currR)
            r += X.transpose().dot(w) - X.transpose().dot(tempW)
            newR = r.sum()

            if abs(currR-newR) < 0.01:
                break

        prec,rec = calculatePR(origW,w)
        precision.append(prec)
        recall.append(rec)
        lembdas.append(lembda)

    print(w)
    print(precision, recall, lembdas)
    plotPR(precision[::-1], recall[::-1], lembdas)
    origW = [True if w >0 else  False for w in origW ]
    predW = [True if w >0 else  False for w in w ]

    precision, recall, _ = precision_recall_curve(origW, predW)
    print(recall)
    return w


def coordinateDescentWine(X,y):

    d,n = X.shape
    delta = 1
    step = 0
    objective = []

    lembda = calLambdaMaxS(X,y)
    print("lemda max",lembda)
    r,rprev,b,bprev = 0,0,0,0 #TODO
    wprev = np.zeros((d, 1))
    Xnew = X.copy()
    Xnew.data **= 2
    a = 2 * Xnew.sum(axis=1)
    c = np.zeros((d, 1))
    w = np.zeros((d,1))

    # while abs((w-wprev).sum()) > delta:
    #     step += 1
    #     if step%10==0:
    #         lembda /= 2
    prevNZ, NZ = 0, 0

    while NZ != prevNZ or NZ == 0:
        step += 1
        lembda /= 2
        rprev = y - X.transpose().dot(w) - b
        bprev = b
        b += rprev.sum()/ n
        r = rprev + bprev - b
        wprev = w.copy()

        prevNZ = NZ
        NZ = np.count_nonzero(w)

        for k in range(d):
            tempW = w.copy()
            c[k] = 2* (X[k].dot(sparse.csr_matrix(r) )).data[0] + 2* X[k].dot(X[k].transpose()).data[0] * w[k]

            ck = c[k]
            ak = a[k]

            if ck < -lembda:
                w[k] = (ck+lembda)/ak
            elif ck > lembda:
                w[k] = (ck - lembda)/ak
            else:
                w[k] = 0

            currR = r.copy()
            objective.append(currR)
            r += X.transpose().dot(w) - X.transpose().dot(tempW)
            newR = r
            if abs(mean_squared_error(currR, newR)) > 0.063:
                break

    print(len(c),  len(a))
    print("lembda: ",lembda)
    print("features: ", np.count_nonzero(w))

    y_predicted = b + X.transpose().dot(w)
    rms = sqrt(mean_squared_error(y, y_predicted))
    print("training RMSD:",rms)

    return wprev,b, lembda


def calLambdaMaxS(X,y):
    newy =  y-np.sum(y)/y.shape[0]
    return 2*max(X.dot(newy))[0]


def calLambdaMax(X,y):
    newy =  y-np.sum(y)/y.shape[0]

    return 2*max(X.dot(newy))[0]


def generateData(d,n,k,sigma):
    b = 0
    w = np.zeros((d, 1))

    X = np.random.normal(0, sigma, (d,n))
    epsilon = np.random.normal(0, sigma, (n,1))

    for i in range(k):
        w[i]=10 if bool(random.getrandbits(1)) else -10
        #w[i]=10

    y = np.matmul(np.transpose(X), w) + b + epsilon

    return X,y,w


def loadData(data,lbl):
    train = np.transpose(np.loadtxt(data))
    X = sparse.csr_matrix((train[2]-1,(train[1]-1,train[0]-1)))
    y = np.asmatrix(np.loadtxt(lbl))
    y = np.transpose(y)

    return X,y


def loadForTest(data):
    train = np.loadtxt(data)
    train = np.transpose(train)
    X = sparse.csr_matrix((train[2] - 1, (train[1] - 1, train[0] - 1)))
    return X


def pred(X,w,b):
    return (b + X.transpose().dot(w))

def rmsd(y,ypred):
    rms = sqrt(mean_squared_error(y, ypred))
    return rms


k=10
#X,y,w = generateData(80,250,k,1)
#Xs = sparse.csr_matrix(X)
#ys = sparse.csr_matrix(y)
data="data/trainData.txt"
lbl = "data/trainLabels.txt"
Xs,y = loadData(data,lbl)

w,b,_ = coordinateDescentWine(Xs, y)


data="data/valData.txt"
lbl = "data/valLabels.txt"
Xs,y = loadData(data,lbl)
yPred = pred(Xs,w,b)
print(yPred)
print("validation RMSD: ", rmsd(y, yPred))


Xs = loadForTest("data/testData.txt")
yPred=pred(Xs,w,b)


with open('predTestLabels.csv','w') as file:

    file.write("ID,Points")
    file.write('\n')
    for i,y in enumerate(yPred):
        dt = str(i+1)+","+str(y[0])
        file.write(dt)
        file.write('\n')

#wPred = coordinateDescent(X, y,w)
#print(calculatePR(w,wPred))