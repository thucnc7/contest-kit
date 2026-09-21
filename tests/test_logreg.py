import warnings

import numpy as np
import pytest

from contest_kit.logreg import (
    sigmoid, predict_proba, predict, bce_loss, gradient, fit,
)


# ---------- sigmoid ----------

def test_sigmoid_moc_quen():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(np.log(3)) == pytest.approx(0.75)      # 3/(1+3)
    assert sigmoid(-np.log(3)) == pytest.approx(0.25)


def test_sigmoid_doi_xung():
    z = np.array([-4.0, -0.7, 0.0, 0.7, 4.0])
    assert sigmoid(z) + sigmoid(-z) == pytest.approx(np.ones_like(z))


def test_sigmoid_khong_tran_so():
    """Ban ngay tho 1/(1+exp(-z)) se bao RuntimeWarning o day."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sigmoid(np.array([-1000.0, -710.0, 710.0, 1000.0]))
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0 and out[-1] == 1.0


# ---------- ca tinh tay ----------

def test_ca_tinh_tay_hai_mau():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([1.0, 0.0])
    W = np.zeros(2)
    b = 0.0

    p = predict_proba(X, W, b)
    assert p == pytest.approx([0.5, 0.5])
    assert bce_loss(p, y) == pytest.approx(np.log(2))

    gW, gb = gradient(X, p, y)
    assert gW == pytest.approx([-0.25, 0.25])
    assert gb == pytest.approx(0.0)        # hai mau lech nguoc nhau, triet tieu


def test_ca_tinh_tay_mot_mau():
    X = np.array([[1.0, 0.0]])
    y = np.array([1.0])
    p = predict_proba(X, np.zeros(2), 0.0)

    assert bce_loss(p, y) == pytest.approx(np.log(2))
    gW, gb = gradient(X, p, y)
    assert gW == pytest.approx([-0.5, 0.0])
    assert gb == pytest.approx(-0.5)       # khac ca tren, de phan biet


def test_bce_loss_khong_ra_inf_khi_p_bang_bien():
    assert np.isfinite(bce_loss(np.array([0.0, 1.0]), np.array([1.0, 0.0])))


# ---------- gradient check bang sai phan huu han ----------

def test_gradient_khop_sai_phan_huu_han():
    """Cach kiem gradient khong can thu vien: so no voi do doc do bang so.

    Neu cong thuc giai tich sai dau hoac sai he so 1/n, test nay bat duoc.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 4))
    y = rng.integers(0, 2, size=7).astype(float)
    W = rng.normal(scale=0.3, size=4)
    b = 0.2
    h = 1e-6

    gW, gb = gradient(X, predict_proba(X, W, b), y)

    for j in range(4):
        Wp, Wm = W.copy(), W.copy()
        Wp[j] += h
        Wm[j] -= h
        so = (bce_loss(predict_proba(X, Wp, b), y)
              - bce_loss(predict_proba(X, Wm, b), y)) / (2 * h)
        assert gW[j] == pytest.approx(so, abs=1e-6)

    so_b = (bce_loss(predict_proba(X, W, b + h), y)
            - bce_loss(predict_proba(X, W, b - h), y)) / (2 * h)
    assert gb == pytest.approx(so_b, abs=1e-6)


# ---------- fit ----------

def test_fit_giam_loss_va_tach_duoc_du_lieu_de():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(-1.5, 0.5, size=(60, 2)),
                   rng.normal(+1.5, 0.5, size=(60, 2))])
    y = np.r_[np.zeros(60), np.ones(60)]

    W, b, h = fit(X, y, lr=0.5, epoch=300)

    assert h["train"][0] == pytest.approx(np.log(2))   # W = 0 -> p = 0.5
    assert h["train"][-1] < h["train"][0]
    assert (predict(X, W, b) == y).mean() > 0.95


def test_fit_ghi_val_khi_duoc_truyen():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    _, _, h = fit(X, y, epoch=5, X_val=X, y_val=y)
    assert len(h["train"]) == 5 and len(h["val"]) == 5


def test_fit_nhan_pandas_va_tra_ve_numpy():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 1.0, 0.0, 0.0]})
    y = pd.Series([0, 0, 1, 1])

    W, b, _ = fit(X, y, epoch=10)
    assert isinstance(W, np.ndarray)       # khong duoc bien thanh Series
    assert isinstance(b, float)
    assert isinstance(predict_proba(X, W, b), np.ndarray)


def test_shape_lech_thi_bao_loi():
    with pytest.raises(ValueError):
        predict_proba(np.zeros((3, 4)), np.zeros(5), 0.0)
    with pytest.raises(ValueError):
        gradient(np.zeros((3, 2)), np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError):
        fit(np.zeros((3, 2)), np.zeros(4))
