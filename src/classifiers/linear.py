from dataclasses import dataclass
import random
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from functools import partial
from sklearn.model_selection import StratifiedKFold


def whiten(X: np.ndarray) -> np.ndarray:
    # X is a array of matrices of size (n_feats)
    # make sure we are zero centered and unit variance
    X = X.copy()
    X -= X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X /= std
    return X


def mean_aggregation(X: list[np.ndarray]) -> np.ndarray:
    return np.stack([x.mean(axis=0) for x in X])


def last_aggregation(X: list[np.ndarray]) -> np.ndarray:
    return np.asarray([x[-1] for x in X])


def aggregate(X_list: list[np.ndarray], method: str) -> np.ndarray:
    # X is a list of matrices of size (seq_len, n_feats)
    # we need to contract it to vectors of size (n_feats,)
    if method == "mean":
        return mean_aggregation(X_list)
    elif method == "last":
        return last_aggregation(X_list)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def k_fold_cross_validate(
    X: np.ndarray, y: list[str], k: int = 5, **model_params
) -> tuple[float, float]:
    """
    Perform k-fold cross validation.
    Returns mean score and standard deviation.
    """
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    scores = []
    for train_idx, val_idx in tqdm(kf.split(X, y), total=k, desc="K-fold CV"):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = np.array(y)[train_idx], np.array(y)[val_idx]

        clf = make_log_regression(model_params)
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_val, y_val))

    return np.mean(scores), np.std(scores)


def hyperparam_optimize(X: np.ndarray, y: list[str]) -> tuple[dict, float, float]:
    best_score_mean = float("-inf")
    best_score_std = None
    best_params = {}

    # hypers = []
    # for penalty in ["l1", "l2", "elasticnet"]:
    #     for C in [0.001, 0.002, 0.003, 0.004, 0.008, 0.01]:
    #         hypers.append((penalty, C))
    # 

    hypers = []
    for C in [0.001, 0.003, 0.008, 0.01]:
        hypers.append(("l2", C))

    train_fn = partial(k_fold_cross_validate, X=X, y=y)

    for penalty, C in hypers:
        score_mean, score_std = train_fn(penalty=penalty, C=C)
        print(f"hypers {penalty=} and {C=} got {score_mean=:.3f} (±{score_std:.3f})")
        if score_mean > best_score_mean:
            print(f"new best: {score_mean:.3f} (±{score_std:.3f}), {penalty}, {C}")
            best_score_mean = score_mean
            best_score_std = score_std
            best_params = {"penalty": penalty, "C": C}

    return best_params, best_score_mean, best_score_std


def load_as_npz(filename: str) -> list[dict]:
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


def extract_data(data: list[dict]) -> tuple[np.ndarray, list[str]]:
    X_list = [d["features"] for d in data]
    y = [d["label"] for d in data]
    labels = set(y)
    print(f"found {len(labels)} classes: {labels}")
    X = aggregate(X_list, "mean")
    X = whiten(X)
    return X, y


def split(X: np.ndarray, y: list[str], test_size=0.2):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)


@dataclass
class TrainResult:
    clf: LogisticRegression
    cv_score_mean: float
    cv_score_std: float
    test_score: float
    precision: float
    recall: float
    f1: float


def train(X: np.ndarray, y: list[str]) -> TrainResult:
    # First split into train and test
    X_train, X_test, y_train, y_test = split(X, y)

    print(f"Data loaded: {len(X_train)} training examples, {len(X_test)} test examples")
    print("Hyperparam optimization...")

    # Do hyperparameter optimization using k-fold CV on training data only
    best_params, best_mean, best_std = hyperparam_optimize(X_train, y_train)

    # Train final model on full training data
    clf = make_log_regression(best_params)

    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    # get pr and f1 on test set
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, clf.predict(X_test), average="macro"
    )

    return TrainResult(
        clf=clf,
        cv_score_mean=best_mean,
        cv_score_std=best_std,
        test_score=test_score,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def make_log_regression(model_kwargs=None):
    model_kwargs = model_kwargs or {}
    if model_kwargs["penalty"] == "elasticnet":
        model_kwargs["l1_ratio"] = 0.5

    return LogisticRegression(
        class_weight="balanced", solver="saga", n_jobs=-1, **model_kwargs
    )


def main(path: str):
    X, y = extract_data(load_as_npz(path))
    print("Data loaded:", len(X))
    return train(X, y)


DATA_DIR = Path("./data")


def test_h4rm3l():
    datasets = dict(
        harmful=load_as_npz(DATA_DIR / "h4rm3l-output-harmful_prompt.npz"),
        eval_granted=load_as_npz(DATA_DIR / "h4rm3l-output-eval_prompt.npz"),
        eval_refused=load_as_npz(DATA_DIR / "h4rm3l-output-eval_prompt-refused.npz"),
    )

    for name, data in datasets.items():
        print(f"class [{name}] has {len(data)} examples")

    results = {}
    for class_a, class_b in [
        ("harmful", "eval_granted"),
        # ("eval_granted", "eval_refused"),
        ("eval_refused", "harmful"),
    ]:
        print(f"Training on {class_a} and {class_b}")
        data = datasets[class_a] + datasets[class_b]
        random.shuffle(data)
        X, y = extract_data(data)

        train_result = train(X, y)

        print(train_result)

        results[(class_a, class_b)] = train_result

    print(results)


if __name__ == "__main__":
    test_h4rm3l()
