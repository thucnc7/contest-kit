"""Tao file nop va kiem tra lai file nop.

check_submission la ham phai go lai duoc tu con so khong duoi 5 phut.
Chi numpy/pandas, khong import cheo trong contest_kit.
"""

import numpy as np
import pandas as pd


def make_submission(ids, labels, path, id_col="id", label_col="label"):
    """Ghi file nop. index=False la bat buoc, khong thi thua mot cot."""
    if len(ids) != len(labels):
        raise ValueError(f"Do dai lech: {len(ids)} ids vs {len(labels)} labels")

    df = pd.DataFrame({id_col: ids, label_col: labels})
    df.to_csv(path, index=False)
    return path


def check_submission(
    path,
    n_rows=None,
    columns=None,
    expected_ids=None,
    id_col=None,
    label_cols=None,
    valid_labels=None,
    allow_negative=False,
):
    """Doc LAI file tu dia roi kiem. Loi nao cung gom lai, bao mot lan.

    n_rows        : so dong mong doi (khong tinh header)
    columns       : danh sach ten cot dung thu tu. Bay hay gap nhat:
                    to_csv quen index=False -> them cot "Unnamed: 0"
    expected_ids  : day ID dung THU TU. Sai thu tu la bai chet am tham.
    id_col        : ten cot ID (mac dinh columns[0] neu co columns)
    label_cols    : cac cot bi kiem NaN/am/nhan la. Mac dinh: moi cot tru id_col
    valid_labels  : tap nhan hop le. Gia tri ngoai tap nay -> loi
    allow_negative: False -> gia tri am trong label_cols la loi

    Tra ve DataFrame da doc lai. Sai thi raise ValueError liet ke het.
    """
    loi = []

    df = pd.read_csv(path)

    # --- cot ---
    if columns is not None:
        if list(df.columns) != list(columns):
            loi.append(f"Cot sai: file co {list(df.columns)}, can {list(columns)}")
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        loi.append(f"Co cot rac {unnamed} - chac quen index=False khi to_csv")

    # --- so dong ---
    if n_rows is not None and len(df) != n_rows:
        loi.append(f"So dong sai: file co {len(df)}, can {n_rows}")

    # --- ID ---
    if id_col is None and columns:
        id_col = columns[0]
    if id_col is not None and id_col in df.columns:
        if df[id_col].duplicated().any():
            dup = df.loc[df[id_col].duplicated(), id_col].unique()[:5]
            loi.append(f"ID trung: {list(dup)} ...")
        if expected_ids is not None:
            got = df[id_col].to_numpy()
            want = np.asarray(expected_ids)
            if got.shape != want.shape:
                loi.append(f"So ID sai: {got.shape[0]} vs {want.shape[0]}")
            elif not (got == want).all():
                sai = int(np.argmax(got != want))
                loi.append(
                    f"Thu tu ID sai tu dong {sai}: co {got[sai]!r}, can {want[sai]!r}"
                )
    elif id_col is not None:
        loi.append(f"Khong thay cot ID {id_col!r}")

    # --- gia tri nhan ---
    if label_cols is None:
        label_cols = [c for c in df.columns if c != id_col]
    for c in label_cols:
        if c not in df.columns:
            loi.append(f"Khong thay cot {c!r}")
            continue
        s = df[c]
        n_nan = int(s.isna().sum())
        if n_nan:
            loi.append(f"Cot {c!r} co {n_nan} o NaN")
        if not allow_negative and pd.api.types.is_numeric_dtype(s):
            n_am = int((s.dropna() < 0).sum())
            if n_am:
                loi.append(f"Cot {c!r} co {n_am} gia tri am")
        if valid_labels is not None:
            la = set(pd.unique(s.dropna())) - set(valid_labels)
            if la:
                loi.append(f"Cot {c!r} co nhan la: {sorted(la, key=str)[:5]}")

    if loi:
        raise ValueError("File nop khong hop le:\n- " + "\n- ".join(loi))
    return df
