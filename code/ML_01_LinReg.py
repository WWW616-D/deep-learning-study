import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class LinReg:

    def __init__(self, ):
        self.X = None # (p+1)-by-n
        self.Y = None # k-by-n
        self.W = None # (p+1)-by-k [W;b]

    def fit(self, X, y, opt_alg):
        self.X = self.padding(X)
        self.Y = self.oneHotEncoding(y)
        if opt_alg == 'GD': # Gradient Descent
            self.W = self.opt_alg(self.X, self.Y)
        else: # default: Analytical Solution
            self.W = self.opt_alg(self.X, self.Y)
        return self.W

    def predict(self, result,X, y, clf_name='Linear Regression'):
        prob = np.dot(self.W.T, self.padding(X))
        y_pred = np.argmax(prob, axis=0) + 1 # 0-based => 1-based
        total = len(y)
        correct = np.sum(y_pred == y)
        acc = correct / total
        #print('%s:\n\t Accuracy is %d/%d=%.4f' % (clf_name, correct, total, acc))
        result.append(acc)
        return y_pred, acc

    def opt_alg(self, X, Y, option=True):
        # min J(W) = ||W'X-Y||_F^2
        #   W = (XX')^{-1} (XY')
        if option: # pseudo inverse
            W = np.dot(np.linalg.pinv(X.T), Y.T)  #计算伪逆
        else: # works only if XX' is invertible
            XX = np.dot(X, X.T)
            XY = np.dot(X, Y.T)
            W = np.dot(np.linalg.inv(XX), XY)
        return W

    def opt_alg_gd(self, X, Y, maxIter=100000, alpha=1e-5):
        # min J(W) = ||W'X-Y||_F^2
        #   W_{t+1} = W_{t} - alpha * (-2XY' + 2XX'W_{t})
        p, k = X.shape[0], Y.shape[0]
        W = np.random.rand(p, k)
        for i in range(maxIter):
            dW = - 2 * np.dot(X, Y.T) + 2 * np.dot(np.dot(X, X.T), W)
            W = W - alpha * dW
            J = np.linalg.norm(np.dot(W.T, X) - Y, 'fro') ** 2
            print('%d-th iteration: J=%.4f' % (i, J))
        return W

    def padding(self, X):
        n = X.shape[1]
        return np.vstack((X, np.ones([1, n])))

    def oneHotEncoding(self, y):
        n = y.shape[1]
        def reorder_labels(x):
            x = np.squeeze(x) #降低维度
            s = set(x.tolist()) #创立集合
            y = np.zeros([n, ], dtype=int)
            for idx, val in enumerate(s):
                y[x == val] = idx # 0-based 返回一个布尔数组哦
            return y
        y = reorder_labels(y)
        k = np.max(y) + 1
        I = np.identity(k)  #一个单位矩阵
        Y = np.zeros([k, n])
        Y[:, 0:n] = I[:, y[0:n]] #超级拷贝
        return Y


def test():
    df = pd.read_table('C:\\Users\\17944\\Downloads\\HandWrittenLetters.txt', sep=',', header=None)
    df = df.values
    X = df[1:, :] / 255 # scale to [0, 1]
    y = df[0, :]
    index=0.001
    result_array=[]
    for i in range(99):
        index=(i+1)*0.01
        X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=index, random_state=42)
        # y = df[0:1, :]
        #X_train, X_test, y_train, y_test = train_test_split(X.T, np.squeeze(y), test_size=0.5, random_state=42)
        X_train = X_train.T
        X_test = X_test.T
        y_train = np.reshape(y_train, [1, len(y_train)])

        clf = LinReg()
        #print("y_test 前 20 个值:", y_test[:1000])
        clf.fit(X_train, y_train, opt_alg='default')
        y_pred, acc = clf.predict(result_array,X_test, y_test)

    print(result_array)
    maxval=max(result_array)
    maxindex=result_array.index(maxval)
    print(maxval,maxindex)
if __name__ == '__main__':
    test()
