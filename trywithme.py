import numpy as np

class FFBP:
    def __init__(self,X,T,hidden, Ir=le-1,alpha=1e-2,eps=le-3,
    epochs=float('inf')):
    self.X = X
    self.T = T
    self.hidden = hidden
    self.Ir = Ir
    self.alpha = alpha
    self.eps = eps
    self.ephochs = epochs

    self.d_in = self.X.shape[1]
    if len(self.T.shape) == 1 :
        self.T = np.reshape(self.T,(len(X),1))
    del X,T
    self.d_out = self.T.shape[1]
    
self.error= []
self.W = []
self.init_weights()
self.input [None] * (len(self.W)+1)

self.forward(self.X)
self.backward()

@staticmethod
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@staticmethod
def d_sigmoid(x):
    return x*(1-x) # X คือ ค่าเอ้าพุตของ ซิกมอยด์

@staticmethod
def pad_ones(input):
    return np.hstack((input,np.ones((len(input)))))
def init_weights(input)
    dim = [self.d_in]+self.hidden + [self.d_out]
    for i in range(len(self.hidden)+1):
        output = self.sigmoid(input A self.w[i])
        input = self.pad_ones(output)
        self.input[i+i] = input.copy()
    return output

def backward(self):
    epoch = 0
    dw = [0] * len(self.W)
    while True:
        epoch += 1
        output = self.forward(self.X)

        error = self.T - output
        mse = np.mean(error**2)
        print('epoch{}:\terror={}'.format(epoch,mse))
        self.error.append(mse)
        if mse<self.eps or epoch >= self.ephochs:
            break

        #output layer
        delta = self.d_sigmoid(output)*error
        dw[-1] =  self.Ir * self.Ir * self.input[-2].T @ delta+self.alpha * dw [-1]
        self.W[-1] += dw[-1]

        #hidden layer
        for i in range(len(self.W)-2,-1,-1):
            delta = self.d_sigmoid(self.input[i+1])*(delta @ self.W[i+1].T)
            delta = delta[:,:-1]
            dw[i] = self.Ir * self.input[i].T @ delta + self.alpha * dw[i]
            self.W[i] += dw[i]