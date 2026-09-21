"""Cac ham cham diem tu viet cho contest.

Quy tac chung (vi phai go lai trong phong thi):
  - chi dung numpy, khong import cheo trong contest_kit
  - moi ham tu dung mot minh, tham so tuong minh

Trang thai kiem chung:
  [DA KHOP] macro_f1      -> khop sklearn.f1_score(average="macro")
  [DA KHOP] count_score   -> khop exp(-MAPE)
  [CHUA KHOP] dice, weather_score, chon_nguong, radar_score
             -> cong thuc dang chung, PHAI doi chieu notebook de goc truoc khi tin.
"""

import numpy as np


def macro_f1(y_true, y_pred, labels=None) -> float:
    """F1 trung binh khong trong so tren cac nhan.

    labels=None -> lay hop cac nhan xuat hien o y_true HOAC y_pred
    (giong sklearn). Nhan nao ca hai deu khong nhac den thi bo qua.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape lech: {y_true.shape} vs {y_pred.shape}")

    if labels is None:
        labels = np.union1d(np.unique(y_true), np.unique(y_pred))

    f1s = []
    for c in labels:
        true_c = (y_true == c)
        pred_c = (y_pred == c)

        tp = int((true_c & pred_c).sum())
        fp = int((~true_c & pred_c).sum())   # doan la c nhung that ra khong phai
        fn = int((true_c & ~pred_c).sum())   # that ra la c nhung doan truot

        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)

    if not f1s:
        return 0.0
    return float(np.mean(f1s))


def count_score(y_true, y_pred) -> float:
    """Diem de Chicken Counting: exp(-MRE), MRE = mean(|pred-true| / true)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape lech: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Khong co gi de cham")
    if (y_true <= 0).any():
        raise ValueError("y_true co phan tu <= 0, khong chia duoc")
    if (y_pred < 0).any():
        raise ValueError("y_pred co phan tu am")

    mre = float((np.abs(y_pred - y_true) / y_true).mean())
    score = float(np.exp(-mre))

    assert 0.0 <= score <= 1.0, f"score ngoai khoang: {score}"
    return score


def dice(y_true, y_pred) -> float:
    """Dice cho mat na nhi phan: 2|A giao B| / (|A| + |B|).

    Hai mat na cung rong -> 1.0 (doan dung la "khong co gi").
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape lech: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Mat na rong, khong co gi de cham")

    a = y_true.astype(bool).ravel()
    b = y_pred.astype(bool).ravel()

    inter = int((a & b).sum())
    total = int(a.sum()) + int(b.sum())

    if total == 0:
        return 1.0
    return float(2 * inter / total)


def weather_score(y_true, y_pred, class_axis=0) -> float:
    """De Satellite Weather: Dice trung binh tren tung lop.

    y_true, y_pred: mang nhi phan, mot truc la truc lop
    (vi du (C, H, W) voi class_axis=0). Mang 1 chieu -> coi nhu 1 lop.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape lech: {y_true.shape} vs {y_pred.shape}")
    if y_true.ndim == 1:
        return dice(y_true, y_pred)

    t = np.moveaxis(y_true, class_axis, 0)
    p = np.moveaxis(y_pred, class_axis, 0)

    return float(np.mean([dice(t[c], p[c]) for c in range(t.shape[0])]))


def chon_nguong(y_true, y_prob, nguongs=None):
    """Quet nguong, tra ve (nguong tot nhat, dice tot nhat) theo Dice.

    y_prob: xac suat/diem thuc. Hoa thi lay nguong nho hon.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape lech: {y_true.shape} vs {y_prob.shape}")

    if nguongs is None:
        nguongs = np.arange(0.05, 1.0, 0.05)

    best_t, best_d = None, -1.0
    for t in nguongs:
        d = dice(y_true, y_prob >= t)
        if d > best_d:
            best_t, best_d = float(t), d

    return best_t, best_d


def radar_score(y_true, y_pred, background=0, weight=50.0, ignore_label=None) -> float:
    """De Radar: do chinh xac co trong so, pixel khong nen nang `weight` lan.

    y_true co nhan tu -1 den 3; -1 truyen qua ignore_label neu de noi la
    vung bo qua. Diem = sum(w_i * dung_i) / sum(w_i).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape lech: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Khong co gi de cham")

    t = y_true.ravel()
    p = y_pred.ravel()

    if ignore_label is not None:
        keep = (t != ignore_label)
        t, p = t[keep], p[keep]
        if t.size == 0:
            raise ValueError("Bo het pixel sau khi loai ignore_label")

    w = np.where(t == background, 1.0, float(weight))
    dung = (t == p).astype(float)

    return float((w * dung).sum() / w.sum())
