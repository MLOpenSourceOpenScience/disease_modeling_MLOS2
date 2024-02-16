from pathlib import Path

import numpy as np

from .converter import get_csv
from .globals import save_np_as_csv


def format_nlp_disease_csv(path: str, out: str, config: str):
    """
    Formats csv data to the shape (timestamp x row x columns)
    Saves the formatted data to out_path
    :param path: path to the data file
    :param out: path to the output file
    :param config: path to the config for the csv file
    """
    gdf = get_csv(path, config, verbose=False)
    gdf_arr = [
        n.set_index("Location Name").T.drop(["geometry"])
        for n in np.array_split(gdf, len(gdf) // gdf["Location Name"].nunique())
    ]
    res_arr = [d.reindex(sorted(d.columns), axis=1).to_numpy() for d in gdf_arr]
    res = np.swapaxes(np.stack(res_arr, axis=0), 1, 2)
    save_np_as_csv(out, res, list(gdf["Location Name"].unique()))


def segment_csv(path: str, out: str, config: str, rows_per_timestamp: int):
    """
    Segments a csv file into length // rows_per_timestamp separate csv files.
    :param path: path to the directory
    :param out: path to the output file
    :param config: path to the config file
    :param rows_per_timestamp: number of rows per timestamp
    """
    gdf = get_csv(path, config, verbose=False)
    gdf_arr = np.array_split(gdf, len(gdf) // rows_per_timestamp)
    for i, df in enumerate(gdf_arr):
        save_np_as_csv(f"{out}_{i}.csv", df)


def segment_dir(path: str, out_dir: str, config: str, rows_per_timestamp: int):
    """
    Segments all csv files in a directory and writes them to out_dir
    :param path: path to the directory
    :param out_dir: path to the output directory
    :param config: path to the config file
    :param rows_per_timestamp: number of rows per timestamp
    """
    for pth in Path(path).glob("*.csv"):
        segment_dir(
            str(pth),
            f"{out_dir}/{str(pth).split('/')[-1]}",
            config,
            rows_per_timestamp,
        )


def format_csv(
    path: str,
    out: str,
    config: str,
    rows_per_timestamp: int,
    label_col: str,
):
    """
    Formats a 3-D csv by splitting by timestamp , transposing, and reassigning feature names
    :param path: path to the csv
    :param out: path to the output file
    :param config: path to config file
    :param rows_per_timestamp: number of rows per time stamp
    :param label_col: the column that will be used as the new feature labels
    """
    gdf = get_csv(path, config, verbose=False)
    gdf_arr = [
        n.set_index(label_col)
        for n in np.array_split(gdf, len(gdf) // rows_per_timestamp)
    ]
    res_arr = [d.reindex(sorted(d.columns), axis=1).to_numpy() for d in gdf_arr]
    res = np.stack(res_arr, axis=0)
    save_np_as_csv(out, res, list(gdf[label_col].unique()))


def format_dir(
    path: str, out_dir: str, config: str, rows_per_timestamp: int, label_col: str
):
    """
    :param path: path to the csv directory
    :param out_dir: path to the output directory
    :param config: path to the config file
    :param rows_per_timestamp: number of rows per timestamp
    :param label_col: the column that will be used as the new feature labels
    """
    for pth in Path(path).glob("*.csv"):
        format_csv(
            str(pth),
            f"{out_dir}/{str(pth).split('/')[-1]}",
            config,
            rows_per_timestamp,
            label_col,
        )
