# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 09:55:13 2023

@author: marzova

y = [y1, y2, y3].T = [w1 * X1, w2 * X2, w3 * X3].T
colX = [y1^2, y1y2, y1y3, y2^2, y2y3, y3^3].T
w = [w1, w2, w3].T

d(colX)/dw = | 2*y1*X1, 0,       0       |
             | y2*X1,   y1*X2,    0       |
             | y3*X1,   0,       y1*X3    |
             | 0,       2*y2*X2,  0       |
             | 0,       y3*X2,    y2*X3   |
             | 0,       0,       2*y3*X3  |
---------------------------             
 AD SOME SORT OF SOFTMAX
 ---------------------------
"""

import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
import pickle

from nonlinear_miso_ode import Nonlinear_MISO_ODE
from sklearn.preprocessing import MinMaxScaler

class RandomLayer_LNU_HONU_NN:
    def __init__(self, params = None):
        self.random_output = np.array([])
        self.random_weights = np.array([])
        
        self.LNU_output = np.array([])
        self.LNU_weights = np.array([])
        self.LNU_input = np.array([])
        
        self.HONU_output = np.array([])
        self.HONU_weights = np.array([])
        self.HONU_input = np.array([])
        
        self.error = np.array([])
        
        self.delta_LNU = []
        self.delta_HONU = []
        
        if params == None:
            self.N_random = 20
            self.N_LNU = 5
            self.N_HONU = 1
            self.HONU_order = 2
            self.learning_rate = 0.001
            
        else:
            self.N_random = params['N_random']
            self.N_LNU = params['N_LNU']
            self.N_HONU = params['N_HONU']
            self.HONU_order = params['HONU_order']
            self.learning_rate = params['learning_rate']
            
    def create_state(self, X, y, nu, ny, hp):
        x = np.ones(nu * X[:, None].shape[-1] + ny + 1)
        data = np.ones(nu * X[:, None].shape[-1] + ny + 2)
        
        ### - reshape input array if (N,) to (N, 1)
        if X.shape[-1] ==len(X) :
            X = X.reshape(len(X), 1)
            
        ### - append inputs and outputs into prepared array according to 
        ### - defined delays (nu, ny)
        for k in range(max(nu, ny), len(X) - hp): 
            for l in range(X[:, None].shape[-1]):
                x[l * nu + 1: (l+1) * nu + 1] = X[k-nu:k, l]

            x[(l+1) * nu + 1:] = y[k-ny:k]
            
            x_all = np.append(x, y[k+ hp])
            data = np.vstack((data, x_all))
             
        return data
            
    def initiate_weights(self, X):
        self.random_weights = np.random.uniform(-1, 1, (self.N_random, len(X)))#/self.N_random # Initiate random weights with uniform distribution
        self.LNU_weights = np.random.randn(self.N_LNU, int(self.N_random / self.N_LNU))#/self.N_LNU
        # self.HONU_weights = np.random.randn(math.comb(int(self.N_random / self.N_LNU) +self.HONU_order-1, self.HONU_order))/100
        self.HONU_weights = np.random.randn(math.comb(self.N_LNU +self.HONU_order-1, self.HONU_order))     
        
    def RandomLayer_ff(self, X):
        self.random_output = np.tanh(np.dot(self.random_weights, X))
            
    def create_HONU(self, X):
        X_HONU = np.prod(np.asarray(list(itertools.combinations_with_replacement(X, self.HONU_order))), axis = 1)
        return X_HONU
            
    def LNULayer_ff(self, X):
        
        M = int(self.N_random / self.N_LNU)
        self.LNU_input = np.asarray([X[i-M:i] for i in range(M, len(X)+1, M)])
        
        self.LNU_output = np.tanh(np.sum(np.dot(self.LNU_weights, self.LNU_input.T), axis = 1))
            
    def LNULayer_bp(self, X):
        
        # - dercolX creation
        combo = len(self.HONU_input)
        nlnu = self.N_LNU
        M = int(self.N_random / self.N_LNU)

        dercolX = np.zeros((combo, nlnu, M))
        group = 0
        for i in range(nlnu):
            x = self.LNU_input[i:, :]
            y = self.LNU_output[i:]
            # print(x.shape);print(len(y))
            
            sb = np.zeros((len(x), len(x), M))
            
            mp1 = np.multiply(np.multiply(np.ones(x.shape) * x[0, :], y[:, np.newaxis]), (1-np.tanh(y*y)**2)[:, np.newaxis]) # y1X1, y2X1, y3X1, ...
            # mp1 = (np.ones(x.shape) * x[0, :]) * y
            
            sb[:, 0, :] = mp1 # add to first column
            
            mp2 = np.multiply(np.multiply(x, y[0, np.newaxis]), (1-np.tanh(y*y)**2)[:, np.newaxis])
            for j in range(len(x)):
                sb[j, j, :] += mp2[j, :]
            
            dercolX[group : group+len(x), i:, :] = sb
            group += len(x) 
        
        # self.LNU_weights += (self.learning_rate / self.error**2) * self.error * np.tensordot(self.HONU_weights, dercolX, axes=1)
        self.LNU_weights += (self.learning_rate ) * self.error * np.tensordot(self.HONU_weights, dercolX, axes=1)
        self.delta_LNU.append(self.LNU_weights)
        
    def HONULayer_ff(self, X):
        self.HONU_input = self.create_HONU(X)
        self.HONU_output = np.dot(self.HONU_weights, self.HONU_input)
            
    def HONULayer_bp(self, X):

        # self.HONU_weights += (self.learning_rate/self.error**2) * self.error * self.HONU_input
        self.HONU_weights += (self.learning_rate) * self.error * self.HONU_input
        self.delta_HONU.append(self.HONU_weights)
        # self.HONU_weights += (self.learning_rate/self.error**2) * self.error * self.create_HONU(self.LNU_output)
            
    def fit(self, X, y):
        self.initiate_weights(X[0, :])
        error = []
        for i in range(50):
            for i in range(len(X)):
                self.RandomLayer_ff(X[i, :])
                self.LNULayer_ff(self.random_output)
                self.HONULayer_ff(self.LNU_output)
                
                self.error = y[i] - self.HONU_output
                error.append(self.error)
                
                self.HONULayer_bp(X[i, :])
                self.LNULayer_bp(X[i, :])
        return error
            
    def predict(self, X, y):
        predicted = np.zeros(len(X))
        err = np.zeros(len(X))
        for i in range(len(X)):
            self.RandomLayer_ff(X[i, :])
            self.LNULayer_ff(self.random_output)
            self.HONULayer_ff(self.LNU_output)
            
            self.error = y[i] - self.HONU_output
            
            predicted[i] = self.HONU_output
            err[i] = self.error
        return predicted, err
        
# X = np.random.randn(100, 5)
# y = np.random.randn(100)

# model = RandomLayer_LNU_HONU_NN()
# model.fit(X, y)
# model.predict()
with open('MISO_dict.pkl', 'rb') as f:
    data = pickle.load(f)

model = RandomLayer_LNU_HONU_NN()
data1 = model.create_state(np.c_[data['u1'], data['u2']], data['output'][0][:, 1], nu = 10, ny = 20, hp = 5)
scaler = MinMaxScaler((0,1))
data = scaler.fit_transform(data1)

X_train = data[:5000, :-1]
y_train = data[:5000, -1]

X_test = data[5000:, :-1]
y_test = data[5000:, -1]

error = model.fit(X_train, y_train)
predicted, err = model.predict(X_test, y_test)
plt.figure()
plt.plot(predicted, label='pred')
plt.plot(y_test, label = 'true')
plt.legend()

plt.figure()
plt.plot(error)

fig, ax = plt.subplots(2)
ax[0].plot(np.array(model.delta_LNU).reshape(-1))
ax[1].plot(np.asarray(model.delta_HONU).reshape(-1))