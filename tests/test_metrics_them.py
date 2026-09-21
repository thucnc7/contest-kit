import numpy as np
import pytest

from contest_kit.metrics import dice, weather_score, chon_nguong, radar_score


# ---------- dice ----------

def test_dice_vi_du_tay():
    a = np.array([1, 1, 0, 0])
    b = np.array([1, 0, 0, 0])
    assert dice(a, b) == pytest.approx(2 * 1 / (2 + 1))


def test_dice_trung_khop_hoan_toan():
    a = np.array([[1, 0], [0, 1]])
    assert dice(a, a) == 1.0


def test_dice_ca_hai_rong():
    assert dice(np.zeros(5), np.zeros(5)) == 1.0


def test_dice_khong_giao():
    assert dice(np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1])) == 0.0


def test_dice_shape_lech():
    with pytest.raises(ValueError):
        dice(np.zeros(3), np.zeros(4))


# ---------- weather_score ----------

def test_weather_score_la_dice_trung_binh():
    t = np.array([[1, 1, 0, 0], [1, 1, 1, 1]])
    p = np.array([[1, 0, 0, 0], [1, 1, 1, 1]])
    cho = (dice(t[0], p[0]) + dice(t[1], p[1])) / 2
    assert weather_score(t, p) == pytest.approx(cho)


def test_weather_score_mot_chieu():
    t = np.array([1, 1, 0])
    assert weather_score(t, t) == 1.0


# ---------- chon_nguong ----------

def test_chon_nguong_tim_dung_cho_cat():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    t, d = chon_nguong(y, prob, nguongs=[0.15, 0.5, 0.95])
    assert t == 0.5 and d == 1.0


def test_chon_nguong_hoa_lay_nguong_nho():
    y = np.array([0, 1])
    prob = np.array([0.1, 0.9])
    t, d = chon_nguong(y, prob, nguongs=[0.5, 0.6, 0.7])
    assert t == 0.5 and d == 1.0


# ---------- radar_score ----------

def test_radar_score_dung_het():
    y = np.array([0, 1, 2, 3])
    assert radar_score(y, y) == 1.0


def test_radar_score_truot_pixel_nang():
    y = np.array([0, 0, 0, 1])        # w = 1,1,1,50
    p = np.array([0, 0, 0, 0])        # truot dung pixel nang
    assert radar_score(y, p) == pytest.approx(3 / 53)


def test_radar_score_truot_pixel_nen():
    y = np.array([0, 0, 0, 1])
    p = np.array([1, 0, 0, 1])        # truot mot pixel nen
    assert radar_score(y, p) == pytest.approx(52 / 53)


def test_radar_score_bo_qua_nhan_am_mot():
    y = np.array([-1, 0, 1])
    p = np.array([3, 0, 1])           # pixel -1 sai nhung bi bo qua
    assert radar_score(y, p, ignore_label=-1) == 1.0


def test_radar_score_shape_lech():
    with pytest.raises(ValueError):
        radar_score(np.zeros(3), np.zeros(4))
