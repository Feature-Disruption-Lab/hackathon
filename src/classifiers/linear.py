import random
from typing import Literal
from anyio import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import multiprocessing as mp
from functools import partial
import os


def whiten(X: np.ndarray) -> np.ndarray:
    # X is a array of matrices of size (n_feats)
    # make sure we are zero centered and unit variance
    X -= X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X /= std
    return X


def split_data(X, y, split_pct=0.8):
    # First split into train and temp (test+val)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=split_pct, stratify=y, random_state=42
    )

    # Split temp into test and validation with equal proportions
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


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


def logistic_regression(
    X,
    y,
    penalty: Literal["l2", "l1", "elasticnet"] = "l2",
    C=1.0,
):
    X_train, y_train, X_val, y_val, _, _ = split_data(X, y)
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


def _train_with_params(args, X, y):
    # Set a different but deterministic seed for each process
    np.random.seed(os.getpid())  # Use process ID as seed
    penalty, C = args
    clf, score = logistic_regression(
        X,
        y,
        penalty=penalty,
        C=C,
    )
    clf.set_params(n_jobs=1)
    nnz = np.count_nonzero(clf.coef_)
    return penalty, C, score, nnz


def hyperparam_optimize(X, y):
    # hyperparameter optimization
    best_score = 0
    best_nnz = 99999999
    best_params = {}
    hyper_results = {}
    
    hypers = []
    for penalty in ["l1", "l2", "elasticnet"]:
        for C in [0.001, 0.002, 0.003, 0.004, 0.008, 0.01]:
            hypers.append((penalty, C))
    
    # Create a partial function with X, y fixed
    train_fn = partial(_train_with_params, X=X, y=y)
    
    # Use all available CPUs except one
    n_processes = max(1, mp.cpu_count() - 1)
    
    # Run parallel hyperparameter optimization
    with mp.Pool(n_processes) as pool:
        # Rule of thumb: aim for each process to get 4-8 chunks
        chunksize = max(1, len(hypers) // (n_processes * 4))
        results = list(tqdm(
            pool.map(train_fn, hypers, chunksize=chunksize),
            total=len(hypers),
            desc="Hyperparameter optimization"
        ))
    
    # Process results
    for penalty, C, score, nnz in results:
        print(f"hypers {penalty=} and {C=} got {score=}")
        
        if score > best_score or (score == best_score and nnz < best_nnz):
            print("new best: ", score, penalty, C, nnz)
            best_score = score
            best_params = {"penalty": penalty, "C": C}
            best_nnz = nnz
        hyper_results[(penalty, C)] = score
        
    return best_params, best_score, hyper_results


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

    return data


def extract_data(data):
    X = [d["features"] for d in data]
    y = [d["label"] for d in data]
    labels = set(y)
    print(f"found {len(labels)} classes: {labels}")
    X = aggregate(X, "mean")
    X = whiten(X)
    return X, y


def train(X, y):
    print("Data loaded:", len(X))
    print("Hyperparam optimization...")
    best_params, best_score, hyper_results = hyperparam_optimize(X, y)
    clf, score = logistic_regression(X, y, **best_params)
    _, _, _, _, X_test, y_test = split_data(X, y)
    test_score = clf.score(X_test, y_test)
    return clf, test_score, hyper_results


def main(path: str):
    X, y = extract_data(load_as_npz(path))
    print("Data loaded:", len(X))
    return train(X, y)


# IGNORE: oli's working:
if __name__ == "__main__":
    # main("./data/h4rm3l-output-eval_prompt.npz")
    DATA_DIR = Path("./data")

    datasets = dict(
        harmful=load_as_npz(DATA_DIR / "h4rm3l-output-harmful_prompt.npz"),
        eval_granted=load_as_npz(DATA_DIR / "h4rm3l-output-eval_prompt.npz"),
        eval_refused=load_as_npz(DATA_DIR / "h4rm3l-output-eval_prompt-refused.npz"),
    )
    for name, data in datasets.items():

    results = {}
    for class_a, class_b in [
        ("harmful", "eval_granted"),
        ("eval_granted", "eval_refused"),
        ("eval_refused", "harmful"),
    ]:
        print(f"Training on {class_a} and {class_b}")
        data = datasets[class_a] + datasets[class_b]
        random.shuffle(data)
        X, y = extract_data(data)
        clf, test_score, hyper_results = train(X, y)
        print(f"test score: {test_score}")
        print(f"hyper results: {hyper_results}")
        results[(class_a, class_b)] = {
            "test_score": test_score,
            "hyper_results": hyper_results,
        }

    print(results)
