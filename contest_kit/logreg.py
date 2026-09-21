"""Logistic regression viet tay bang numpy.

Quy tac cua kit (vi phai go lai trong phong thi):
  - chi numpy, khong import cheo trong contest_kit
  - moi ham tu dung mot minh, tham so tuong minh
  - cat pandas ngay o bien: moi ham asarray dau vao, ben trong khong con Series
"""

import numpy as np


def sigmoid(z):
    """1 / (1 + e^-z), khong bao gio tran so.

    Meo: dung e = exp(-|z|) cho ca hai nhanh, nen exp luon nhan so <= 0.
      z >= 0 -> 1 / (1 + e)
      z <  0 -> e / (1 + e)
    Viet truc tiep 1/(1+np.exp(-z)) thi z = -1000 lam exp(1000) = inf.
    """
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def predict_proba(X, W, b):
    """Xac suat thuoc lop 1. Tra ve mang (n,)."""
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    if X.shape[1] != W.shape[0]:
        raise ValueError(f"Shape lech: X co {X.shape[1]} cot, W co {W.shape[0]}")
    return sigmoid(X @ W + float(b))


def predict(X, W, b, nguong=0.5):
    """Nhan 0/1 sau khi cat o nguong."""
    return (predict_proba(X, W, b) >= nguong).astype(int)


def bce_loss(p, y, eps=1e-7):
    """Binary cross-entropy: -mean(y log p + (1-y) log(1-p)).

    Phai clip: p dung bang 0 hoac 1 thi log ra -inf.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"Shape lech: {p.shape} vs {y.shape}")
    if p.size == 0:
        raise ValueError("Khong co gi de tinh loss")

    p = np.clip(p, eps, 1.0 - eps)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def gradient(X, p, y):
    """Gradient cua BCE: (X^T (p - y) / n, mean(p - y)).

    Dung y nguyen hinh gradient cua MSE o linear regression — dao ham cua
    sigmoid va dao ham cua log triet tieu nhau. Chi ham lien ket doi.
    """
    X = np.asarray(X, dtype=float)
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"Shape lech: {p.shape} vs {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"So dong lech: X co {X.shape[0]}, y co {y.shape[0]}")

    r = p - y
    n = y.shape[0]
    return X.T @ r / n, float(r.mean())


def fit(X, y, lr=0.1, epoch=100, X_val=None, y_val=None):
    """Gradient descent tron. Tra ve (W, b, history).

    history["train"] va history["val"] la loss tung epoch (do TRUOC khi cap
    nhat, nen history["train"][0] la loss cua W = 0). Giu lai de xem da hoi tu
    chua — loss cuoi con dang giam manh nghia la thieu epoch, khong phai het cach.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"So dong lech: X co {X.shape[0]}, y co {y.shape[0]}")

    co_val = X_val is not None and y_val is not None
    if co_val:
        X_val = np.asarray(X_val, dtype=float)
        y_val = np.asarray(y_val, dtype=float)

    W = np.zeros(X.shape[1])
    b = 0.0
    history = {"train": [], "val": []}

    for _ in range(epoch):
        p = predict_proba(X, W, b)
        history["train"].append(bce_loss(p, y))
        if co_val:
            history["val"].append(bce_loss(predict_proba(X_val, W, b), y_val))

        gW, gb = gradient(X, p, y)
        W = W - lr * gW
        b = b - lr * gb

    return W, b, history
