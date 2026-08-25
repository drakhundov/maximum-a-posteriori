import numpy as np


class MultinomialNaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.P_y_log = None
        self.P_xy = None
        self.P_xy_log = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of rows")
        n_samples, n_features = X_train.shape
        # counts: array with counts of each class in the training set.
        self.classes, counts = np.unique(y_train, return_counts=True)
        nclasses = len(self.classes)
        # Calculate Log Prior: log(P(y))
        # self.P_y_log[class_no] = log(P(class))
        self.P_y_log = np.log(counts / n_samples)
        # Assign P(x | y) for all features.
        self.P_xy = np.zeros((n_features, nclasses))
        for i, c in enumerate(self.classes):
            # Take data points that belong to class 'c'.
            X_c = X_train[y_train == c]
            feat_tot = np.sum(X_c[:, fno])
            tot = np.sum(X_c)
            if feat_tot == 0 or tot == 0:
                self.P_xy[fno][i] = 0.0001
            for fno in range(self.nfeat):
                # Add 0.0001 to avoid log(0.0)
                self.P_xy[fno][i] = feat_tot / tot
        self.P_xy_log = np.log(self.P_xy)

    def predict(self, X_new: np.ndarray):
        log_prob = np.dot(X_new, self.P_xy_log) + self.P_y_log
        return self.classes[np.argmax(log_prob, axis=1)]
