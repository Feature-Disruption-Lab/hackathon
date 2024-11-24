import random
from typing import Literal
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from tqdm import tqdm


def whiten(X: np.ndarray) -> np.ndarray:
    # X is a array of matrices of size (n_feats)
    # make sure we are zero centered and unit variance
    X -= X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X /= std
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
    return np.stack([x.mean(axis=0) for x in X])


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
    # X = aggregate(X, aggregation_method)
    # X = whiten(X)
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
    kwargs = {}
    if penalty == "elasticnet":
        kwargs = {"l1_ratio": 0.5}

    clf = LogisticRegression(
        class_weight="balanced",  # to handle imbalanced classes
        solver="saga",
        n_jobs=-1,  # use all available
        penalty=penalty,
        C=C,
        **kwargs,
    )
    clf = clf.fit(X_train, y_train)
    return clf, clf.score(X_val, y_val)


def hyperparam_optimize(X, y):
    # hyperparameter optimization
    best_score = 0
    best_nnz = 99999999
    best_params = {}
    hyper_results = {}
    hypers = []
    # for aggregation_method in ["mean", "last"]:
    for penalty in ["l1", "l2", "elasticnet"]:
        for C in [0.001, 0.002, 0.003, 0.004, 0.008, 0.01]:
            hypers.append((penalty, C))
    for penalty, C in tqdm(hypers):
        clf, score = logistic_regression(
            X,
            y,
            penalty=penalty,
            C=C,
        )

        nnz = np.count_nonzero(clf.coef_)
        if score > best_score or (score == best_score and nnz < best_nnz):
            print("new best: ", score, penalty, C, nnz)
            best_score = score
            best_params = {"penalty": penalty, "C": C}
            best_nnz = nnz
        hyper_results[(penalty, C)] = score
    return best_params, best_score, hyper_results


def load_data(matrix_path, seed=42):
    # assume data will be in the format of a list of ('prompt', 'response', 'activation', 'label') tuples
    with open(matrix_path, "rb") as f:
        data = pickle.load(f)

    X = [d[2] for d in data]
    y = [d[3] for d in data]

    return X, y


def load_as_npz(filename: str):
    loaded = np.load(filename, allow_pickle=True)
    data = []
    i = 0
    while f"response_{i}" in loaded:
        data.append(
            {
                "prompt": str(),
                "response": str(loaded[f"response_{i}"]),
                "features": loaded[f"dense_features_{i}"][None][0].toarray(),
                "label": str(loaded[f"label_{i}"]),
            }
        )
        i += 1

    random.shuffle(data)
    return data


def extract_data(data):
    X = [d["features"] for d in data]
    y = [d["label"] for d in data]
    X = aggregate(X, "mean")
    X = whiten(X)
    return X, y


def main(matrix_path):
    X, y = extract_data(load_as_npz(matrix_path))
    print("Data loaded:", len(X))
    print("Hyperparam optimization...")
    best_params, best_score, hyper_results = hyperparam_optimize(X, y)
    clf, score = logistic_regression(X, y, **best_params)
    _, _, _, _, X_test, y_test = preprocess_data(X, y)
    test_score = clf.score(X_test, y_test)
    return clf, test_score, hyper_results


if __name__ == "__main__":
    main("./output_sparse.npz")
