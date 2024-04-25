from datetime import datetime, timedelta
from pathlib import Path

import mlossp as sp
import numpy as np
import pandas as pd


REGION = lambda x: x.split(".")[-2][-1]
NDVI = lambda x: str(x).split("_")[-2]
GPM = lambda x: str(x).split(".")[-4][:8]
GLDAS = lambda x: str(x).split("_")[-1][4:12]
CONFIG_LIST = [
    "Configs/NDVI_config.json",
    "Configs/GPM_config.json",
    "Configs/GLDAS_config.json",
    "Configs/POP_DEN_config.json",
]
KWARGS_LIST = [
    {
        "record_out": NDVI,
        "region_out": REGION,
        "extra_config": "Configs/NDVI_nc.json",
        "suffix_out": lambda: "NDVI",
    },
    {
        "record_out": GPM,
        "region_out": REGION,
        "extra_config": "Configs/GPM_nc.json",
        "suffix_out": lambda: "GPM",
    },
    {
        "record_out": GLDAS,
        "region_out": REGION,
        "extra_config": "Configs/GLDAS_nc.json",
        "suffix_out": lambda: "GLDAS",
    },
    {
        "region_out": REGION,
        "suffix_out": lambda: "POP_DEN"
    },
]


def process_one(config_file, kwargs):
    preprocessor = sp.Preprocessor(
        config_file=config_file,
        verbose=False,
        **kwargs,
    )
    preprocessor.preprocess()


def process_all(config_list, kwargs_list):
    preprocessor = sp.Preprocessor(config_file=config_list[0], verbose=False)
    preprocessor.preprocess_multi(config_list, kwargs_list)


if __name__ == "__main__":
    # Preprocess data
    # process_all(CONFIG_LIST, KWARGS_LIST)
    # process_one("Configs/GPM_config.json", KWARGS_LIST[1])

    # Load and reshape dataframes
    data_series = [n for n in Path("C:/CS/disease_modeling_MLOS2/Data/TEST").glob("*.csv")]
    dse = pd.read_csv("Data/Out/SriLankaDiseaseFull.csv").applymap(lambda x: float(x[2:-2]))
    dse.index.name = "index"
    dse = dse.melt(ignore_index=False)
    dse.rename(columns={"variable": "region", "value":"cases"}, inplace=True)
    dse.sort_values(["index", "region"], inplace=True)

    # Slice datasets to desired timestamps
    columns = list(pd.read_csv(data_series[0]).columns)
    data_series = [pd.read_csv(n) for n in data_series]
    data_series += [dse]
    data_series[0] = data_series[0][315*25:791*25].reset_index()
    data_series[1] = data_series[1][333*25:809*25].reset_index()
    data_series[1]["minTime"] = data_series[1]["minTime"].apply(lambda x : str(datetime(1970, 1, 1) + timedelta(days=x)))
    data_series[2] = data_series[2][333*25:809*25].reset_index()
    data_series[3] = data_series[3].reset_index()

    # Vertical Alignment
    sp.Preprocessor.align_vertical_df(data_series, "Data/Datasets/sri_lanka_2013-2022_vertical.csv")

    # Drop unnecessary cols
    df = pd.read_csv("Data/Datasets/sri_lanka_2013-2022_vertical.csv")
    df.drop(["index", "time", "Unnamed: 0"], axis=1, inplace=True)
    df.to_csv("Data/Datasets/sri_lanka_2013-2022_vertical.csv", index=False)
