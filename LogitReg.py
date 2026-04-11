#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:26:21 2021

@author: Midas
"""

import numpy as np
import pandas as pd

class LogitReg:

    def __init__(self, lr=0.01, thresh=1e-7, maxiter=10000, eps=1e-9, verbose=False):
        self.X = None # p-by-n
        self.y = None # 1-by-n
        self.w = None # (p+1)-by-1 i.e. [w; b]
        self.lr = 0.01 #学习步长 默认0.01
        self.maxiter = maxiter  #最大次数:10000
        self.thresh = thresh #收敛阈值
        self.eps = eps #数值平滑项，防止0
        self.verbose = True

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.w = self.opt_alg(self.padding(X), y, self.lr)
        return self.w

    def predict(self, X):
        prob = self.compute_prob(self.w, self.padding(X))
        prob[prob >= 0.5] = 1
        prob[prob < 0.5] = 0
        return prob
    def padding(self, X):
        n = X.shape[1]
        return np.vstack((X, np.ones([1, n])))

    def init_w(self, p, d=0.001):
        return np.random.random([p, 1]) * 2 * d - d  # range: [-d, d]

    def sigmoid(self, z):
        #print(z)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_prob(self, w, X):
        return self.sigmoid(np.dot(w.T, X))

    def cost_fn(self, X, y, w):
        n = X.shape[1]
        prob = self.compute_prob(w, X)#求出一组概率向量
        #print(prob)
        J = -1.0 / n * np.sum(y * np.log(prob) + (1 - y) * np.log(1-prob))  #只关心对应概率，接近0则损失大，接近1则几乎没损失
        return J

    def gradient(self, X, y, w):
        p, n = X.shape
        prob = self.compute_prob(w, X)
        grad = -1.0 / n * np.sum((y - prob) * X, axis=1)
        return np.reshape(grad, [p, 1])

    def opt_alg(self, X, y, lr):
        # Optimization: Gradient Descent Algorithm
        p, n = X.shape
        w = self.init_w(p)  #初始化随记权重向量
        loss = [self.cost_fn(X, y, w)] #记录初始损失值
        #print(loss)
        #return
        for i in range(self.maxiter):
            grad = self.gradient(X, y, w)
            w = w - lr * grad
            loss.append(self.cost_fn(X, y, w))
            if self.verbose:
                print('%d-th iteration: loss=%.10f' % (i+1, loss[-1]))
            if i > 1 and abs(loss[-1] - loss[-2]) / abs(loss[-1]) < self.thresh:
                break
        return w


def call_baseline(X, y):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression().fit(X, y)
    y_pred = clf.predict(X)
    total = len(y)
    correct = np.sum(y_pred == y)
    acc = correct / total
    print('Scikit-Learn Baseline: accuracy is %d/%d=%.4f' % (correct, total, acc))


def load_dataset(filename, sep=','):
    # Assume file is n-by-(p+1) table, and the last column is class label.

    def reorder_labels(x): # x is pd.Series
        s = set([val for val in x])
        y = np.zeros([1, len(x)], dtype=int)
        for i, v in enumerate(s):
            y[0, x == v] = i
        return y

    df = pd.read_table(filename, sep=sep)
    X = df.iloc[:, :-1].values.T # p-by-n
    y = reorder_labels(df.iloc[:, -1]) # 1-by-n
    return X, y


def main():
    # X, y = load_dataset('./data/iris.csv', ',')
    X, y = load_dataset('C:\\Users\\17944\\Downloads\\ATNTFaceImages400 (1).txt', ',')

    call_baseline(X.T, np.squeeze(y))

    clf = LogitReg(verbose=False)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    total = y.shape[1]
    correct = np.sum(y_pred == y)
    acc = correct / total
    print('The Implemented LogitReg Model: accuracy is %d/%d=%.4f' % (correct, total, acc))


if __name__ == '__main__':
    main()
