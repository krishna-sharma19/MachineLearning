import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import scipy.sparse as sparse
from sklearn.metrics import mean_squared_error
from math import sqrt



def coordinateDescentSparse(X,y, origW, ld):

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

    lembda = ld
    if ld == None:
        lembda = calLambdaMaxS(X,y)

    print("max lembda",lembda)
    r,rprev,b,bprev = 0,0,0,0
    wprev = np.zeros((d, 1))
    Xnew = X.copy()
    Xnew.data **= 2
    a = 2 * Xnew.sum(axis=1)
    c = np.zeros((d, 1))
    w = np.zeros((d,1))
    precision, recall,lembdas = [],[],[]
    lembdaChange = False

    while step < 10 and not lembdaChange:
        if ld != None:
            lembdaChange = True

        step += 1
        if ld == None:
            lembda /= 2

        rprev = y - X.transpose().dot(w) - b
        bprev = b
        b += rprev.sum()/ n
        r = rprev + bprev - b
        wprev = w.copy()

        for k in range(d):
            tempW = w.copy()
            c[k] = 2* (X[k].dot(sparse.csr_matrix(r))).data[0] + 2* X[k].dot(X[k].transpose()).data[0] * w[k]

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

        origW = [True if w > 0 else  False for w in origW]
        predW1 = [True if w > 0 else  False for w in w]
        prec, rec, _ = precision_recall_curve(origW, predW1)
        prec,rec = calculatePR(origW,w)
        precision.append(prec)
        recall.append(rec)
        lembdas.append(lembda)

    print(precision, recall, lembdas)
    if ld == None:
        plotPR(precision, recall, lembdas)

#    precision, recall, _ = precision_recall_curve(origW, predW)
    return w, lembda


def coordinateDescentWine(X,y,valX,valy):

    def plotSingle(l,nz):


        plt.plot(l, nz)

        plt.xlabel('Lamnda')
        plt.ylabel('nz')
        plt.title('non zero graph')
        plt.show()

    def plot(t,v,l):
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()

        ax1.plot(l, t, 'g-')
        ax2.plot(l, v, 'b-')

        ax1.set_xlabel('Lambda')
        ax1.set_ylabel('training', color='g')
        ax2.set_ylabel('validation', color='b')

        plt.show()

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

    lmd =[]
    trmsds, vrmsds,nzs = [], [],[]

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

        rmsd = None
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
            rmsd = abs(mean_squared_error(currR, newR))
            if  rmsd > 0.063:
                break

        y_predicted = b + X.transpose().dot(w)
        rms = sqrt(mean_squared_error(y, y_predicted))
        trmsds.append(rms)

        y_predicted = b + valX.transpose().dot(w)
        rms = sqrt(mean_squared_error(valy, y_predicted))
        vrmsds.append(rms)

        lmd.append(lembda)
        nzs.append(NZ)
    plot(trmsds,vrmsds,lmd)
    plotSingle(lmd,nzs[::-1])
    print("lembda: ",lembda)
    print("features: ", np.count_nonzero(w))

    y_predicted = b + X.transpose().dot(w)
    rms = sqrt(mean_squared_error(y, y_predicted))
    print("training RMSD:",rms)

    return wprev,b, lembda


def calLambdaMaxS(X,y):
    newy =  y-np.sum(y)/y.shape[0]
    return 2*max(X.dot(newy))[0,0]


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


def q331():
    k = 10
    X,y,w = generateData(80,250,k,1)
    Xs = sparse.csr_matrix(X)
    ws, lembda = coordinateDescentSparse(Xs,y, w, None)
    print("final lembda",lembda)
    print("w ",ws)

    X,y,w = generateData(80,250,k,10)
    Xs = sparse.csr_matrix(X)
    ws, lembda =coordinateDescentSparse(Xs,y,w, lembda)

#q331()
def q4():
    data="data/trainData.txt"
    lbl = "data/trainLabels.txt"
    Xs,y = loadData(data,lbl)

    data="data/valData.txt"
    lbl = "data/valLabels.txt"
    valX,valy = loadData(data,lbl)

    w,b,lembda = coordinateDescentWine(Xs, y, valX, valy)
    sw = [i[0] for i in sorted(enumerate(w), key=lambda x:x[1])]
    print("heighted w",sw[:10])
    print("heighted w", sw[:-10])
#    data="data/valData.txt"
#    lbl = "data/valLabels.txt"
#    Xs,y = loadData(data,lbl)
#    yPred = pred(Xs,w,b)
#    print("validation RMSD: ", rmsd(y, yPred))


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
q4()