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

X=np.array(X[0:-1]) # batch x vocab
Y=np.array(Y)
batch_size=25
beg=0
end=batch_size

def softmax(y):
    return np.exp(y)/np.sum(np.exp(y))

def getkey(arr,dic):
    for k in dic.keys():
        if list(dic[k])==list(arr):
            return k

def getBatch():
    global beg,end
    if end>=len(X):
        beg=0
        end=batch_size
    x_batch=[]
    y_batch=[]
    for i in range(beg,end):
        x_batch.append(X[i])
        y_batch.append(Y[i])

    beg=end
    end+=batch_size

    return np.array(x_batch),np.array(y_batch)

hidden_size=40

Wxh=np.random.randn(hidden_size,vocab_size)*0.01
Whh=np.random.randn(hidden_size,hidden_size)*0.01
Why=np.random.randn(vocab_size,hidden_size)*0.01
bh=np.zeros([hidden_size,1])
by=np.zeros([vocab_size,1])
learning_rate=0.1
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory states for adagrad optimizer

def model(x,y,prehs):
    loss=0

    hs,ys,ps={},{},{} # maintaining timestamped record to apply BPTT
    hs[-1]=np.copy(prehs) # for when t=0 and hs[t-1] is hs[-1], prehs is last hs[t] of previous batch
    for t in range(len(x)): # forward pass
        inputs=np.array(np.matrix(x[t]).T)
        hs[t]=np.tanh(np.dot(Wxh,inputs)+np.dot(Whh,hs[t-1])+bh)
        ys[t]=np.dot(Why,hs[t])+by
        ps[t]=softmax(ys[t])
        loss+=-np.log(ps[t][list(y[t]).index(1)])

    # backward pass
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros([hidden_size,1])
    """for BPTT, unroll the rnn, apply and accumulate gradients and then apply, here it is TBPTT
        1.Forward pass: Step through the next k1 time steps, computing the input, hidden, and output states.
        2.Compute the loss, summed over the previous time steps (see below).
        3.Backward pass: Compute the gradient of the loss w.r.t. all parameters, accumulating over the previous k2 time steps (this requires having stored all activations for these time steps). Clip gradients to avoid the exploding gradient problem (happens rarely).
        4.Update parameters (this occurs once per chunk, not incrementally at each time step).
        5.If processing multiple chunks of a longer sequence, store the hidden state at the last time step (will be used to initialize hidden state for beginning of next chunk). If we've reached the end of the sequence, reset the memory/hidden state and move to the beginning of the next sequence (or beginning of the same sequence, if there's only one).
        Repeat from step 1.
        """
    for t in reversed(range(len(x))): # this is BPTT
        inputs=np.array(np.matrix(x[t]).T)
        dy=np.copy(ps[t])
        dy[list(y[t]).index(1)]-=1
        dWhy+=np.dot(dy,hs[t].T)
        dby+=dy
        dh=np.dot(Why.T,dy)+dhnext
        dhraw=(1-hs[t]*hs[t])*dh
        dWxh+=np.dot(dhraw,inputs.T)
        dWhh+=np.dot(dhraw,hs[t-1].T)
        dbh+=dhraw
        dhnext=np.dot(Whh.T,dhraw)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]: # avoiding exploing gradients
        np.clip(dparam, -5, 5, out=dparam)

    return np.float32(loss),dWxh,dWhh,dWhy,dbh,dby,hs[len(x)-1]


def train():
    n=0 #  num of iterations
    prehs=np.zeros([hidden_size,1])
    while True:
        x_batch, y_batch=getBatch()
        loss,dWxh,dWhh,dWhy,dbh,dby,prehs = model(x=x_batch, y=y_batch,prehs=prehs)

        for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],[dWxh, dWhh, dWhy, dbh, dby],[mWxh, mWhh, mWhy, mbh, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

        if n%100==0: # display every 100th iteration
            print ("iter: ",n,"loss: ",loss)
        n+=1

def predict(begchar, n_chars):
    tx=np.array(np.matrix(features[begchar]).T)
    tprehs=np.zeros([hidden_size,1])
    genstr=""
    for _ in range(n_chars):
        ths=np.tanh(np.dot(Wxh,tx)+np.dot(Whh,tprehs)+bh)
        tys=np.dot(Why,ths)+by
        tps=softmax(tys)

        pred=np.zeros([vocab_size,1])
        pred[list(tps).index(max(tps))]=1
        k=getkey(pred,features)
        genstr+=k

        tx=np.copy(pred)
        tprehs=np.copy(ths)

    print (genstr)
