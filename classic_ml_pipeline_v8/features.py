import pandas as pd
import numpy as np


def feature_engineering(df: pd.DataFrame, config) -> pd.DataFrame:
    if not bool(config.feature_engineering.enabled):
        return df
    df = df.copy()
    ms_class_map = {
        20: "One",
        30: "One",
        40: "One",
        45: "One",
        50: "One",
        60: "Two",
        70: "Two",
        75: "Two",
        80: "Split",
        85: "Split",
        90: "Duplex",
        120: "One",
        150: "One",
        160: "Two",
        180: "PUD",
        190: "Two",}
    df["MSSubClassCat"] = df["MSSubClass"].astype(str)
    #good df["MedNhbdArea"] = (df.groupby("Neighborhood")["GrLivArea"].transform("median"))
    #bad df["PorchTypes"] = df[["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",]].gt(0).sum(axis=1)
    #bad df["MSClass"] = df["MSSubClass"].map(ms_class_map).fillna("Other")
    #bad df["LivLotRatio"] = (df["GrLivArea"] / df["LotArea"].replace(0, np.nan))
    df["Spaciousness"] = ((df["1stFlrSF"] + df["2ndFlrSF"])/ df["TotRmsAbvGrd"].replace(0, np.nan))
    #bad df["TotalOutsideSF"] = df[["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",]].sum(axis=1)
    #bad bldg_interaction = pd.get_dummies(df["BldgType"], prefix="Bldg").mul(df["GrLivArea"], axis=0)
    #bad df = pd.concat([df, bldg_interaction], axis=1)
    #bad df["Feature1"] = df["GrLivArea"] + df["TotalBsmtSF"]
    #bad df["Feature2"] = df["YearRemodAdd"] * df["TotalBsmtSF"]
    return df