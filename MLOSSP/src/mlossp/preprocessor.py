# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Any

import numpy.typing
import xarray
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from .dataloader import DataLoader
from .globals import (
    save_np_as_horizontal_csv,
    save_np_as_vertical_csv,
    load_horizontal_csv,
)


class Preprocessor:
    def __init__(self, config_file: str, verbose=True, **kwargs):
        r"""
        :param config_file: The path to the preprocessor configuration.
        :param verbose: print logging enabled
        :key extra_config: Provide this config if you are loading a *.nc* file.
        :key record_out: A function to generate a value from a file name.
        :key region_out: A function to generate a value from a region's file name
        :key suffix_out: A function that returns a label to be appended to the end of an output file name.
        :returns: A preprocessor instance
        """
        super().__init__()

        self.__multi_idx = 0
        self.config_file = config_file
        self.verbose = verbose
        self._load_kwargs(kwargs)
        self._load_config(config_file)

    def _log(self, m: str):
        if self.verbose:
            print(m)

    def _initialize_dirs(self):
        for d in [self.DATA_DIR, self.OUT_DIR, self.COUNTRIES_DIR]:
            if not os.path.exists(d):
                os.mkdir(d)

    def _load_config(self, cfg: str):
        self._log(f"Loading Config from {cfg}")
        with open(cfg) as f:
            config = json.load(f)
            self.COMPRESS_TO = config.get("compress_to", None)
            self.CRS = config.get("crs", "EPSG:4326")
            self.DATA_DIR = config.get("data_dir", "")
            self.COUNTRIES_DIR = config.get("regions_dir", "")
            self.OUT_DIR = config.get("out_dir", "")
            self.REGIONS = config.get("regions_file_map", None)
            self.SELECTED_REGIONS = config.get("selected_regions", [])
            self.JOIN_ON = config.get("join_on", None)
            self.JOINS = config.get("joins", None)
            self.EXTENSION = config.get("file_extension", "*.nc4")
            self.CHUNK = config.get("chunk", 1)
        self._initialize_dirs()

    def _load_kwargs(self, kwargs: dict[str, Any]):
        self.extra_config = kwargs.get("extra_config", None)
        self.record_out = kwargs.get("record_out", lambda x: "")
        self.region_out = kwargs.get("region_out", lambda x: "")
        self.suffix_out = kwargs.get("suffix_out", lambda: "")

    def _load_regions(self) -> dict[str, Any]:
        return {
            r: gpd.read_file(f"{self.COUNTRIES_DIR}/{self.REGIONS[r]}")
            for r in self.SELECTED_REGIONS
        }

    def _get_checkpoint_path(self) -> str:
        return f"{self.DATA_DIR}/mlossp_{self.__multi_idx}_checkpoint.pkl"

    def _layer_and_concat(
        self,
        path_to_dir: str,
        extension: str,
        regions: dict[str, Any],
        from_checkpoint: bool,
    ):
        self._log(f"Joining and Concatenating GeoJsons")

        # Look for a checkpoint
        checkpoint_path = self._get_checkpoint_path()
        if from_checkpoint and os.path.isfile(checkpoint_path):
            with open(checkpoint_path, "rb") as checkpoint:
                file_list, out = pickle.load(checkpoint)
            os.remove(checkpoint_path)
        else:
            file_list = iter(
                DataLoader(
                    path_to_dir,
                    extension,
                    self.CRS,
                    self.extra_config,
                    self.verbose,
                    self.CHUNK,
                )
            )
            out = defaultdict(list)

        # Join satellite data across all data sources and regions
        for f, gdf in tqdm(file_list):
            for r, rdf in regions.items():
                gdf.to_crs(rdf.crs)
                joined = gpd.sjoin(rdf, gdf, how="left", predicate="intersects")
                aggregate = joined.dissolve(by=self.JOIN_ON, aggfunc=self.JOINS)
                aggregate.columns = [
                    "".join([s[1], s[0].title()]) if type(s) == tuple else s
                    for s in aggregate.columns
                ]

                if self.COMPRESS_TO:
                    aggregate.to_pickle(
                        f"{self.COMPRESS_TO}/{r}_{self.region_out(self.REGIONS[r])}_{self.record_out(f)}.pkl"
                    )

                out[r] += [aggregate.drop("geometry", axis=1)]

            # Checkpoint Iterator State
            with open(checkpoint_path, "wb") as checkpoint:
                pickle.dump((file_list, out), checkpoint, protocol=-1)

        # Save all time series as numpy files.
        for r, rdf in out.items():
            series = np.squeeze(xarray.DataArray([rdf]).to_numpy(), axis=0)
            pth = (
                f"{self.OUT_DIR}/{r}_time_series_{self.__multi_idx}_{self.suffix_out()}"
            )
            self._log(f"Saving outputs for {r} to {pth}")
            np.save(f"{pth}.npy", series)
            save_np_as_horizontal_csv(
                f"{pth}.csv", series, list(regions[r][self.JOIN_ON])
            )
            save_np_as_vertical_csv(
                f"{pth}_vertical.csv",
                series,
                list(regions[r][self.JOIN_ON]),
                list(rdf[0].columns),
            )

        os.remove(checkpoint_path)

    def preprocess(self, from_checkpoint: bool = False):
        r"""
        :param from_checkpoint: Set to true to load from a checkpoint
        """
        self._layer_and_concat(
            self.DATA_DIR, self.EXTENSION, self._load_regions(), from_checkpoint
        )

    def preprocess_multi(
        self,
        config_list: List[str],
        kwargs_list: Optional[List[dict]],
        from_checkpoint: bool = False,
    ):
        r"""
        :param config_list: A list of config file names
        :param kwargs_list: A list of dictionaries mapping kwargs for each config
        :param from_checkpoint: Set to true to load from the latest checkpoint
        """
        for i, c_f in enumerate(config_list):
            self.__multi_idx = i
            self._load_config(c_f)
            if kwargs_list:
                self._load_kwargs(kwargs_list[i])
            if from_checkpoint and not os.path.isfile(self._get_checkpoint_path()):
                continue
            self.preprocess(from_checkpoint)

    @staticmethod
    def _align_npy_series(data: List[np.typing.NDArray]) -> np.typing.NDArray:
        max_len = max([n.shape[0] for n in data])
        if not all([x.shape[1] == data[0].shape[1] for x in data]):
            raise ValueError(
                f"All data must have the same number of regions, received {[s.shape for s in data]}"
            )

        for i, n in enumerate(data):
            if n.shape[0] == max_len:
                continue
            data[i] = np.resize(data[i], (max_len, n.shape[1], n.shape[2]))

        aligned_data = np.concatenate(data, axis=2)
        return aligned_data

    @staticmethod
    def align_npy(path: str, out_path: str):
        r"""
        Aligns all .npy files within the provided directory into a npy and csv file.
        All files must be preprocessed to have the same number of timestamps.
        Data of different lengths are repeated to be equal to the longest time series.
        :param path: Path to the directory containing the data files.
        :param out_path: Path to the output file, e.g. Data/output_file_name.
        """
        data_series = [
            np.load(str(n), allow_pickle=True) for n in Path(path).glob("*.npy")
        ]
        aligned_data = Preprocessor._align_npy_series(data_series)
        print(f"Saving time series of shape {aligned_data.shape}.")
        np.save(f"{out_path}.npy", aligned_data)
        save_np_as_horizontal_csv(f"{out_path}.csv", aligned_data)

    @staticmethod
    def align_horizontal_csv(path: str, out_path: str):
        r"""
        Aligns all horizontal csv time series data within a directory. All data must have identical columns.
        If you have vertical csv data, see mlossp.formatter's format_csv and format_dir functions or align_vertical_csv.
        :param path: Path to the directory with the csv data
        :param out_path: Path to the output file
        """
        data_series = [n for n in Path(path).glob("*.csv")]
        columns = pd.read_csv(data_series[0]).columns
        data_series = [load_horizontal_csv(str(n)) for n in data_series]
        aligned_data = Preprocessor._align_npy_series(data_series)
        save_np_as_horizontal_csv(out_path, aligned_data, columns)

    @staticmethod
    def align_vertical_csv(path: str, out_path: str):
        r"""
        Aligns all csv time series data within a directory.
        All data must be vertical and have the same number of rows.
        :param path: Path to the directory with the csv data
        :param out_path: Path to the output file
        """
        data_series = [pd.read_csv(n) for n in Path(path).glob("*.csv")]
        aligned_data = pd.concat(data_series, axis = 1)
        aligned_data.to_csv(out_path)

    @staticmethod
    def align_all(path: str, out_path: str):
        r"""
        Aligns all horizontal csvs and numpy time series data within a directory. All data must have identical columns.
        :param path: Path to the directory
        :param out_path: Path to the output file
        """
        npy_series = [
            np.load(str(n), allow_pickle=True) for n in Path(path).glob("*.npy")
        ]
        csv_series = [n for n in Path(path).glob("*.csv")]
        columns = pd.read_csv(csv_series[0]).columns
        csv_series = [load_horizontal_csv(str(n)) for n in csv_series]
        aligned_data = Preprocessor._align_npy_series(csv_series + npy_series)
        save_np_as_horizontal_csv(out_path, aligned_data, columns)
