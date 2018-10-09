import numpy as np
from pprint import pprint


def generateData(mean, deviation, d, N):
    X = np.random.normal(loc=mean, scale=deviation, size=(d, N))
    # print(np.transpose(X).shape)
    w = np.zeros((d, 1))
    w[:10] = 10
    #print(w)
    ep = np.random.normal(loc=mean, scale=deviation, size=(N, 1))
    b = 0
    y = np.matmul(np.transpose(X), w) + b + ep
    # print(y)
    return (X, w, ep, b, y)


def computeLambda(X, y):
    r, c = y.shape
    lmb = max(np.matmul(X, y - np.sum(y) / r))
    print(2 * lmb)
    return 2 * lmb


# (X,w,ep,b,y) = generateData(0,1,80,250)
def linearRigression():
    mean, deviation, d, N = 0, 1, 80, 250
    (X, w, ep, b, y) = generateData(mean, deviation, d, N)
    #print(y)
    objective = []

    a = np.sum(X ** 2, axis=1)
    a = 2 * a


    b_new = b
    c = np.zeros((d, 1))
    lambd = computeLambda(X, y)
    i = 0
    w_old = np.zeros((d,1))

    while not abs((w_old-w).sum())<1:
        i += 1
        if i % 10:
            lambd = lambd / 2
        r = y - np.matmul(np.transpose(X), w) - b_new

        # update b
        b_old = b_new
        b_new = np.sum(r + b_old)/N

        # update r

        r = r + (b_old - b_new)
        w_old = w.copy()
        for k in range(d):
            #print(c)
            old_w = w.copy()
            c[k] = 2 * (np.dot(X[k], r) + np.dot(X[k], X[k]) * w[k])

            if c[k] < -lambd:
                w[k] = (c[k] + lambd) / a[k]
            elif c[k] > lambd:
                w[k] = (c[k] - lambd) / a[k]
            else:
                w[k] = 0
            old_e = r.sum()
            objective.append(old_e)
            r = r + np.matmul(np.transpose(X),w) - np.matmul(np.transpose(X),old_w)
            e = r.sum()

            if abs(old_e - e) < 0.01:
                break

    print(lambd)
    print(i)
    print(w)
    import matplotlib.pyplot as plt
    plt.plot(objective)
    plt.ylabel('some numbers')
    plt.show()


linearRigression()