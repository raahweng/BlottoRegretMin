import numpy as np
from itertools import permutations

S = 10
N = 3


def partition(n,m):

    a = [0] * (m+1)
    a[0] = n
    a[m] = -1
    actions = []

    while True:
        while True:
            actions.append(a[:-1])
            if a[1] >= a[0] - 1:
                break
            a[0] -= 1
            a[1] += 1
        
        j = 3
        s = a[0] + a[1] - 1
        while a[j-1] >= a[0] - 1:
            s += a[j-1]
            j += 1

        if j > m:
            return actions
        x = a[j-1] + 1
        a[j-1] = x
        j -= 1

        while j > 1:
            a[j-1] = x
            s -= x
            j -= 1
        a[0] = s

def getActions(S, N):
    p = partition(S,N)
    actions = []
    for i in p:
        for j in set(permutations(i)):
            actions.append(list(j))
    return np.array(actions)

actions = getActions(S, N)
N = len(actions)
actionDict = {".".join([str(j) for j in act]): i for i, act in enumerate(actions.tolist())}

def getUtility(A,B):
    return np.sum(A > B) - np.sum(A < B)

def payoffmatrix():
    global N, actions
    
    A = np.tile(actions, (N, 1)).reshape((N,N,3))
    B = A.transpose((1,0,2))
    P = np.sum(B > A, axis = 2) - np.sum(B < A, axis = 2)
    return P


def getAction(p):
    global N, actions
    ind = np.random.choice(np.arange(N), p=p)
    return actions[ind,:]

def getStrategy(regretSum, strategySum):
    global N

    strategy = np.copy(regretSum)
    strategy[strategy<0] = 0
    normalisingSum = np.sum(strategy)

    if normalisingSum > 0:
        strategy /= normalisingSum
    else:
        strategy = 1/N * np.zeros(N)
    strategySum += strategy

    return strategy, strategySum

def train(iterations, oppStrategy):
    global N, actions

    strategySum = np.zeros(N)
    regretSum = 1/N * np.ones(N)
    
    P = payoffmatrix()

    for i in range(iterations):
        strategy, strategySum = getStrategy(regretSum,strategySum)
        myAction = getAction(strategy)
        oppAction = getAction(oppStrategy())

        oppInd = actionDict[".".join([str(j) for j in oppAction])]
        actionUtility = P[:,oppInd]
        regretSum += actionUtility - getUtility(myAction,oppAction)

    return strategySum

def getAverageStrategy(strategySum):
    global N

    avgStrategy = np.zeros(N)

    normalisingSum = np.sum(strategySum)
    if normalisingSum > 0:
        avgStrategy = strategySum / normalisingSum
    else:
        avgStrategy = 1/N * np.ones(N)
    
    return avgStrategy

def opp():
    global N
    return 1/N * np.ones(N)

def doubletrain(iterations):
    global N, actions

    strategySum1 = np.zeros(N)
    strategySum2 = np.zeros(N)
    rlist = []
    regretSum1 = 1/N * np.ones(N)
    regretSum2 = 1/N * np.ones(N)
    P = payoffmatrix()

    for i in range(iterations):
        strategy1, strategySum1 = getStrategy(regretSum1,strategySum1)
        strategy2, strategySum2 = getStrategy(regretSum2,strategySum2)
        action1 = getAction(strategy1)
        action2 = getAction(strategy2)

        oppInd = actionDict[".".join([str(j) for j in action2])]
        actionUtility = P[:,oppInd]
        r1 = actionUtility - getUtility(action1,action2)
        regretSum1 += r1
        rlist.append(strategySum1)

        oppInd = actionDict[".".join([str(j) for j in action1])]
        actionUtility = P[:,oppInd]
        r2 = actionUtility - getUtility(action2,action1)
        regretSum2 += r2


    return strategySum1, strategySum2, rlist
