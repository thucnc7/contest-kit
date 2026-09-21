import numpy as np
import pandas as pd
import pytest

from contest_kit.submission import make_submission, check_submission


def _viet(tmp_path, df, **kw):
    p = tmp_path / "sub.csv"
    df.to_csv(p, **kw)
    return p


def test_file_dung_thi_qua(tmp_path):
    p = _viet(tmp_path, pd.DataFrame({"id": [1, 2, 3], "label": [0, 1, 2]}), index=False)
    out = check_submission(p, n_rows=3, columns=["id", "label"],
                           expected_ids=[1, 2, 3], valid_labels={0, 1, 2})
    assert len(out) == 3


def test_make_submission_roundtrip(tmp_path):
    p = tmp_path / "s.csv"
    make_submission([10, 11], ["a", "b"], p, id_col="id", label_col="label")
    check_submission(p, n_rows=2, columns=["id", "label"], expected_ids=[10, 11])


def test_quen_index_false(tmp_path):
    p = _viet(tmp_path, pd.DataFrame({"id": [1, 2], "label": [0, 1]}))  # co index
    with pytest.raises(ValueError, match="index=False"):
        check_submission(p, n_rows=2, columns=["id", "label"])


def test_thieu_dong(tmp_path):
    p = _viet(tmp_path, pd.DataFrame({"id": [1, 2], "label": [0, 1]}), index=False)
    with pytest.raises(ValueError, match="So dong sai"):
        check_submission(p, n_rows=3, columns=["id", "label"])


def test_sai_thu_tu_id(tmp_path):
    p = _viet(tmp_path, pd.DataFrame({"id": [2, 1, 3], "label": [0, 1, 2]}), index=False)
    with pytest.raises(ValueError, match="Thu tu ID sai tu dong 0"):
        check_submission(p, n_rows=3, columns=["id", "label"], expected_ids=[1, 2, 3])


def test_id_trung(tmp_path):
    p = _viet(tmp_path, pd.DataFrame({"id": [1, 1], "label": [0, 1]}), index=False)
    with pytest.raises(ValueError, match="ID trung"):
        check_submission(p, n_rows=2, columns=["id", "label"])


def test_nan_va_am_va_nhan_la(tmp_path):
    p = _viet(tmp_path, pd.DataFrame({"id": [1, 2, 3], "label": [np.nan, -1, 9]}),
              index=False)
    with pytest.raises(ValueError) as e:
        check_submission(p, n_rows=3, columns=["id", "label"], valid_labels={0, 1, 2})
    msg = str(e.value)
    assert "NaN" in msg and "am" in msg and "nhan la" in msg


def test_sai_ten_cot(tmp_path):
    p = _viet(tmp_path, pd.DataFrame({"ID": [1], "Label": [0]}), index=False)
    with pytest.raises(ValueError, match="Cot sai"):
        check_submission(p, n_rows=1, columns=["id", "label"])
