import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "about_this_challenge" / "coffee_db.parquet"

YEAR_COLS = [
    '1990/91','1991/92','1992/93','1993/94','1994/95','1995/96','1996/97',
    '1997/98','1998/99','1999/00','2000/01','2001/02','2002/03','2003/04',
    '2004/05','2005/06','2006/07','2007/08','2008/09','2009/10','2010/11',
    '2011/12','2012/13','2013/14','2014/15','2015/16','2016/17','2017/18',
    '2018/19','2019/20'
]

YEAR_NUMS = list(range(1990, 2020))  # numeric anchor = start year


def load_raw() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


def load_data() -> pd.DataFrame:
    df = load_raw()
    df["cagr"] = df.apply(_cagr, axis=1)
    df["avg_annual"] = df[YEAR_COLS].mean(axis=1)
    df["growth_pct"] = df.apply(
        lambda r: (r["2019/20"] - r["1990/91"]) / r["1990/91"] * 100
        if r["1990/91"] > 0 else np.nan, axis=1
    )
    return df


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    long = df.melt(
        id_vars=["Country", "Coffee type"],
        value_vars=YEAR_COLS,
        var_name="year_label",
        value_name="consumption"
    )
    year_map = dict(zip(YEAR_COLS, YEAR_NUMS))
    long["year"] = long["year_label"].map(year_map)
    return long.sort_values(["Country", "year"]).reset_index(drop=True)


def global_trend(df: pd.DataFrame) -> pd.DataFrame:
    totals = df[YEAR_COLS].sum()
    return pd.DataFrame({"year_label": YEAR_COLS, "year": YEAR_NUMS, "consumption": totals.values})


def top_countries(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        df.groupby("Country")["Total_domestic_consumption"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )


def coffee_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Coffee type")["Total_domestic_consumption"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


def growth_table(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("Country")[["1990/91", "2019/20", "cagr", "growth_pct"]].sum()
    grp["cagr"] = df.groupby("Country")["cagr"].mean()
    grp["growth_pct"] = ((grp["2019/20"] - grp["1990/91"]) / grp["1990/91"].replace(0, np.nan)) * 100
    return grp.sort_values("growth_pct", ascending=False).reset_index()


def _cagr(row) -> float:
    start = row["1990/91"]
    end = row["2019/20"]
    if start <= 0 or end <= 0:
        return np.nan
    return (end / start) ** (1 / 29) - 1
