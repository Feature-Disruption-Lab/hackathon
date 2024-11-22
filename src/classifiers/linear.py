from typing import Literal
from sklearn.linear_model import LogisticRegression
import pickle


def logistic_regression(
    X_train,
    y_train,
    X_test,
    y_test,
    penalty: Literal["l2", "l1", "elasticnet"] = "l2",
    C=1.0,
):
    clf = LogisticRegression(
        class_weight="balanced",  # to handle imbalanced classes
        solver="saga",
        n_jobs=-1,  # use all available
        penalty=penalty,
        C=C,
    )
    clf.fit(X_train, y_train)
    return clf, clf.score(X_test, y_test)


def hyperparam_optimize(X_train, y_train, X_test, y_test):
    # hyperparameter optimization
    best_score = 0
    best_params = {}
    for penalty in ["l1", "l2", "elasticnet"]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            clf, score = logistic_regression(
                X_train, y_train, X_test, y_test, penalty=penalty, C=C
            )
            if score > best_score:
                best_score = score
                best_params = {"penalty": penalty, "C": C}
    return best_params, best_score


def load_data(matrix_path, split_pct=0.8):
    # assume data will be in the format of a list of ('prompt', 'activation', 'label') tuples
    with open(matrix_path, "rb") as f:
        data = pickle.load(f)

    X = [d[1] for d in data]
    y = [d[2] for d in data]

    # then split the first split_pct of data into train,
    # and the rest equally into test and validation

    split_idx = int(len(X) * split_pct)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    split_idx = len(X_test) // 2
    X_test, X_val = X_test[:split_idx], X_test[split_idx:]
    y_test, y_val = y_test[:split_idx], y_test[split_idx:]

    return X_train, y_train, X_test, y_test, X_val, y_val


def main(matrix_path):
    X_train, y_train, X_test, y_test, X_val, y_val = load_data(matrix_path)
    best_params, best_score = hyperparam_optimize(X_train, y_train, X_val, y_val)
    clf, score = logistic_regression(X_train, y_train, X_test, y_test, **best_params)
    return clf, score
