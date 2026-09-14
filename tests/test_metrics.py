import numpy as np
import pytest
from sklearn.metrics import f1_score, mean_absolute_percentage_error

from contest_kit.metrics import macro_f1, count_score


# ---------- macro_f1 ----------

def test_macro_f1_vi_du_tay():
    y = np.array([0] * 90 + [1] * 7 + [2] * 3)
    p = y.copy()
    p[85:90] = 1      # 5 con A bị đoán thành B
    p[90:92] = 0      # 2 con B bị đoán thành A
    p[97] = 0         # 1 con C bị đoán thành A
    assert abs(macro_f1(y, p, labels=[0, 1, 2]) - 0.7811) < 1e-4


def test_macro_f1_ca_bien():
    y = np.array([0, 0, 1, 2])
    assert macro_f1(y, y) == 1.0                                    # đoán đúng hết
    assert macro_f1(y, np.zeros(4, dtype=int),
                    labels=[0, 1, 2]) == pytest.approx((2 / 3) / 3)  # đoán tất cả là 0
    assert macro_f1(np.zeros(4, dtype=int),
                    np.zeros(4, dtype=int)) == 1.0                   # chỉ có một lớp


def test_macro_f1_khop_sklearn():
    rng = np.random.default_rng(0)
    for i in range(2000):
        k = rng.integers(2, 6)                  # số lớp
        n = rng.integers(5, 200)                # số mẫu
        w = rng.random(k) ** 3                  # phân bố lệch nặng
        y = rng.choice(k, size=n, p=w / w.sum())
        p = rng.choice(k, size=n, p=w / w.sum())
        labels = list(range(k))

        mine = macro_f1(y, p, labels=labels)
        ref = f1_score(y, p, labels=labels, average="macro", zero_division=0)

        assert np.isfinite(mine), f"case {i}: hàm trả về {mine}"
        assert mine == pytest.approx(ref, abs=1e-9), f"case {i}: mine={mine} ref={ref}"


def test_macro_f1_shape_lech():
    with pytest.raises(ValueError):
        macro_f1([0, 1, 2], [0, 1])


# ---------- count_score ----------

def test_count_score_vi_du_tay():
    y = np.array([3, 5, 12, 40, 150, 200], dtype=float)

    assert count_score(y, y * 1.10) == pytest.approx(0.9048, abs=1e-4)

    p = y.copy()
    p[:3] += 2
    assert count_score(y, p) == pytest.approx(0.8142, abs=1e-4)

    assert count_score(y, y) == 1.0


def test_count_score_khop_sklearn():
    rng = np.random.default_rng(1)
    for i in range(2000):
        n = rng.integers(1, 60)
        y = rng.integers(1, 250, size=n).astype(float)
        p = np.clip(y * rng.normal(1.0, 0.3, size=n), 0, None)

        mine = count_score(y, p)
        ref = float(np.exp(-mean_absolute_percentage_error(y, p)))

        assert mine == pytest.approx(ref, abs=1e-9), f"case {i}: mine={mine} ref={ref}"


def test_count_score_dau_vao_hong():
    with pytest.raises(ValueError):
        count_score([1, 2], [1, 2, 3])     # shape lệch
    with pytest.raises(ValueError):
        count_score([0, 2], [1, 2])        # y_true có số 0
    with pytest.raises(ValueError):
        count_score([1, 2], [-1, 2])       # y_pred âm
    with pytest.raises(ValueError):
        count_score([], [])                # rỗng