import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


class LogRegPCA:
    def __init__(self, pca=True):
        self.pca = PCA() if pca else None
        self.model = LogisticRegression()

    def model_training(self, x, y):
        x = self.preprocess(x)

        if self.pca is not None:
            x = self.pca.fit_transform(x)
            print(x.shape)

        self.model.fit(x, y)

        acc = self.model.score(x, y)
        print('Accuracy on train:', round(acc, 3))

        return acc

    def model_predict(self, x):
        x = self.preprocess(x)

        if self.pca is not None:
            x = self.pca.transform(x)

        y_pred = self.model.predict(x)
        return y_pred

    def model_testing(self, x, y):
        y_pred = self.model_predict(x)

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        print('Accuracy on test:', round(acc, 3))
        print('F1 score on test:', round(f1, 3))
        cm = confusion_matrix(y, y_pred)

        return cm, acc, f1

    def preprocess(self, x):
        vecs = zscore(x, axis=0)

        for i in vecs:
            np.fill_diagonal(i, 0)

        vecs = vecs.reshape((x.shape[0], -1))

        return vecs
