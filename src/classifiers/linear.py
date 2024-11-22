from typing import Literal
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


def whiten(X: np.ndarray) -> np.ndarray:
    # X is a array of matrices of size (n_feats)
    # make sure we are zero centered and unit variance
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return X


def split_data(X, y, split_pct=0.8):
    split_idx = int(len(X) * split_pct)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    split_idx = len(X_test) // 2
    X_test, X_val = X_test[:split_idx], X_test[split_idx:]
    y_test, y_val = y_test[:split_idx], y_test[split_idx:]

    return X_train, y_train, X_test, y_test, X_val, y_val


def mean_aggregation(X: list[np.ndarray]) -> np.ndarray:
    return np.asarray([x.mean(axis=0) for x in X])


def last_aggregation(X: list[np.ndarray]) -> np.ndarray:
    return np.asarray([x[-1] for x in X])


def aggregate(X: list[np.ndarray], method: str) -> np.ndarray:
    # X is a list of matrices of size (seq_len, n_feats)
    # we need to contract it to vectors of size (n_feats,)
    if method == "mean":
        return mean_aggregation(X)
    elif method == "last":
        return last_aggregation(X)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def preprocess_data(X, y, aggregation_method="mean"):
    X = aggregate(X, aggregation_method)
    X = whiten(X)
    X_train, y_train, X_test, y_test, X_val, y_val = split_data(X, y)
    return X_train, y_train, X_test, y_test, X_val, y_val


def logistic_regression(
    X,
    y,
    penalty: Literal["l2", "l1", "elasticnet"] = "l2",
    C=1.0,
    aggregation_method="mean",
):
    X_train, y_train, X_test, y_test, X_val, y_val = preprocess_data(
        X, y, aggregation_method
    )

    clf = LogisticRegression(
        class_weight="balanced",  # to handle imbalanced classes
        solver="saga",
        n_jobs=-1,  # use all available
        penalty=penalty,
        C=C,
    )
    clf.fit(X_train, y_train)
    return clf, clf.score(X_val, y_val)


def hyperparam_optimize(X, y):
    # hyperparameter optimization
    best_score = 0
    best_params = {}
    hyper_results = {}
    for aggregation_method in ["mean", "last"]:
        for penalty in ["l1", "l2", "elasticnet"]:
            for C in [0.001, 0.01, 0.1, 1, 10, 100]:
                clf, score = logistic_regression(
                    X,
                    y,
                    penalty=penalty,
                    C=C,
                    aggregation_method=aggregation_method,
                )
                if score > best_score:
                    best_score = score
                    best_params = {"penalty": penalty, "C": C}
                hyper_results[(aggregation_method, penalty, C)] = score
    return best_params, best_score, hyper_results


def load_data(matrix_path, seed=42):
    # assume data will be in the format of a list of ('prompt', 'activation', 'label') tuples
    with open(matrix_path, "rb") as f:
        data = pickle.load(f)

    X = [d[1] for d in data]
    y = [d[2] for d in data]

    # random.seed(seed)
    # order = list(range(len(X)))
    # random.shuffle(order)
    # X = [X[i] for i in order]
    # y = [y[i] for i in order]

    return X, y


def main(matrix_path):
    X, y = load_data(matrix_path)
    best_params, best_score, hyper_results = hyperparam_optimize(X, y)
    clf, score = logistic_regression(X, y, **best_params)
    _, _, _, _, X_test, y_test = preprocess_data(
        X, y, best_params["aggregation_method"]
    )
    test_score = clf.score(X_test, y_test)
    return clf, test_score, hyper_results
