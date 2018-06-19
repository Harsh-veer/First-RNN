import numpy as np

data=open("text.txt","r").read()
chars=list(set(data))
vocab_size=len(chars)

features=dict()
X=[]
Y=[]

for i in chars:
    t=[]
    for j in chars:
        if i==j:
            t.append(1)
        else:
            t.append(0)
    features[i]=t

for i in data:
    X.append(features[i])


for i in X[1:]:
    Y.append(i)

X=np.array(X[0:-1])
Y=np.array(Y)

def softmax(y):
    return np.exp(y)/np.sum(np.exp(y))

def getkey(arr,dic):
    for k in dic.keys():
        if list(dic[k])==list(arr):
            return k

hidden_size=40

Wxh=np.random.randn(hidden_size,vocab_size)*0.01
Whh=np.random.randn(hidden_size,hidden_size)*0.01
Why=np.random.randn(vocab_size,hidden_size)*0.01
bh=np.zeros([hidden_size,1])
by=np.zeros([vocab_size,1])
learning_rate=0.1
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)

def train():
    loss=0
    for _ in range(700):
        hs=np.zeros([hidden_size,1])
        pre_hs=np.zeros([hidden_size,1])
        dhsprev=np.zeros([hidden_size,1])

        # genstr=""

        for k in range(len(X)):
            x=np.array(np.matrix(X[k]).T)
            pre_hs=np.copy(hs)
            hs=np.tanh(np.dot(Wxh,x)+np.dot(Whh,pre_hs)+bh)
            ys=np.dot(Why,hs)+by
            ps=softmax(ys)

            loss=-np.log(ps[list(Y[k]).index(1)])
            print (loss)
            # predicted=np.zeros([vocab_size,1])
            # predicted[list(ps).index(max(ps))]=1
            # genstr+=getkey(predicted,features)

            dYs=np.copy(ps)
            dYs[list(Y[k]).index(1)]-=1
            dWhy=np.dot(dYs,hs.T)
            dby=np.copy(dYs)
            dhs=np.dot(Why.T,dYs)+dhsprev
            dhsraw=(1-hs*hs)*dhs
            dWxh=np.dot(dhsraw,x.T)
            dbh=np.copy(dhsraw)
            dWhh=np.dot(dhsraw,pre_hs.T)
            dhsprev=np.dot(Whh.T,dhsraw)

            for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
                np.clip(dparam, -5, 5, out=dparam)

            for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],[dWxh, dWhh, dWhy, dbh, dby],[mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8)


def predict(beg,n): #starting char and number of chars to generategenstr=""
    px=np.array(np.matrix(features[beg]).T)
    ppre_hs=np.zeros([hidden_size,1])
    genstr=""
    for _ in range(n):
        phs=np.tanh(np.dot(Wxh,px)+np.dot(Whh,ppre_hs)+bh)
        pys=np.dot(Why,phs)+by
        pps=softmax(pys)
        ppredicted=np.zeros([vocab_size,1])
        ppredicted[list(pps).index(max(pps))]=1
        genstr+=getkey(ppredicted,features)
        px=np.copy(ppredicted)
        ppre_hs=np.copy(phs)

    return genstr
