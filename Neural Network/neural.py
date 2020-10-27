import numpy as np
import random

class nn:
    def __init__(self, hiddenlayers = 1, neurons=5):
        self.hiddenlayers = hiddenlayers
        self.neurons = neurons
    def activation(self, x):
        result = 1/ (1 + np.exp(-x))
        return result
    def deactivate(self, x):
        return self.activation(x) * (1-self.activation(x))
    def cost(self, y, response):
        return np.linalg.norm(y-response)**2
    def train(self, x, y, epochs = 5, alpha = .01):
        x = np.asarray(x)
        y = np.asarray(y)
        w = []
        
        if self.hiddenlayers == 0:
            w.append(np.random.rand(x.shape[1], y.shape[1]))
        else:
            w.append(np.random.rand(x.shape[1], self.neurons))
            for i in range(self.hiddenlayers-1):
                w.append(np.random.rand(self.neurons, self.neurons))
            w.append(np.random.rand(self.neurons, y.shape[1]))
        print('Neural Network structure')
        w = np.asarray(w)
        for i in range(len(w)):
            print("Layer",i,": ", w[i].shape)
        
        #Now its time to train
        for epoch in range(epochs):
            loss = 0
            for (datapoint, target) in zip(x, y):
                xs = []
                ys = []
                xs.append(datapoint)
                ys.append(datapoint)
                
                #Feed forward
                for i in range(len(w)):
                    weights = w[i]
                    z = weights.T @ ys[-1]
                    xs.append(z)
                    ys.append(self.activation(z))
                residvec = ys[-1] - target
                residual = self.cost(ys[-1], target)
                ys.pop(0) #don't need this
                loss += residual
                ##Now its time to back propogate
                gradients = []
#                 print("vars", self.deactivate(xs[-1]).shape, residvec.shape )
                lastgrad = self.deactivate(xs[-1]) * residvec
#                 print("resulting", lastgrad.shape)
                gradients.append(lastgrad)
                #Get all the gradients
                for i in range(len(w)-1):
#                     print("vars", self.deactivate(xs[len(xs)-i-2]).shape, w[len(w)-i-1].shape, gradients[i].shape )
                    nextgrad = self.deactivate(xs[len(xs)-i-2]).reshape((self.deactivate(xs[len(xs)-i-2]).shape[0],1)) *  w[len(w)-i-1] @ gradients[i].reshape((gradients[i].shape[0],1)) 
#                     print("resulting", nextgrad.shape)
                    gradients.append(nextgrad)
#                 print("Gradient layers = ", len(gradients))
#                 for gradient in gradients:
#                     print("_ ", gradient.shape)
                #Update weights
#                 print("updating weights___________")
                for i in range(len(w)):
                    weights = w[len(w)-1-i]
#                     print("vars - need", weights.shape, "=", ys[len(ys)-2 - i].shape, gradients[i].shape )
                    weights -= alpha * ys[len(ys)-2 - i].reshape((ys[len(ys)-2 - i].shape[0],1)) @ gradients[i].reshape((gradients[i].shape[0],1)).T
                    w[len(w)-1-i] = weights
#                 print("___________________________")
            print("loss = " , loss)
        self.w = w
    def predict(self, data):
        results = []
        for point in data:
            xs = []
            ys = []
            xs.append(point)
            ys.append(point)

            #Feed forward
            for i in range(len(self.w)):
                weights = self.w[i]
                z = weights.T @ ys[-1]
                xs.append(z)
                ys.append(self.activation(z))
            results.append(ys[-1])
        return results

