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
import csv
import json
import pickle
from collections import defaultdict
from typing import List, Optional, Any

import numpy.typing
import xarray
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

from .converter import run


def load_from_pickle(path: Path, crs: str):
    p_df = pd.read_pickle(path)
    return gpd.GeoDataFrame(p_df, geometry=p_df["geometry"], crs=crs)


def save_np_as_csv(dest: str, arr: np.typing.NDArray):
    arr = arr.tolist()
    with open(dest, "w", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerows(arr)


class Preprocessor:
    def __init__(self, config_file: str, verbose=True, **kwargs):
        r"""
        :param config_file: The path to the preprocessor configuration.
        :param verbose: print logging enabled
        :key nc_config: Provide this config if you are loading a *.nc* file.
        :key record_out: A function to generate a value from a file name.
        :key region_out: A function to generate a value from a region's file name
        :key out_out: A function that returns a label to be appended to the end of an output file name.
        :returns: A preprocessor instance
        """
        super().__init__()

        self.config_file = config_file
        self.verbose = verbose
        self._load_kwargs(kwargs)
        self._load_config(config_file)

    def _log(self, m: str):
        if self.verbose:
            print(m)

    def _initialize_dirs(self):
        """Creates directories that do not exist in the path"""
        for d in ["", self.DATA_DIR, self.OUT_DIR, self.COUNTRIES_DIR]:
            if not os.path.exists(self.PATH_OF(d)):
                os.mkdir(self.PATH_OF(d))

    def _load_config(self, c_f: str):
        self._log(f"Loading Config from {c_f}")
        with open(c_f) as f:
            config = json.load(f)
            self.PATH_OF = lambda x: config.get("data_path", "") + x
            self.COMPRESS = config.get("compress", False)
            self.CRS = config.get("crs", "EPSG:4326")
            self.DATA_DIR = config.get("data_dir", "")
            self.COUNTRIES_DIR = config.get("regions_dir", "")
            self.OUT_DIR = config.get("out_dir", "")
            self.REGIONS = config.get("regions_file_map", None)
            self.SELECTED_REGIONS = config.get("selected_regions", [])
            self.JOIN_ON = config.get("join_on", None)
            self.JOINS = config.get("joins", None)
            self.EXTENSION = config.get("file_extension", "*.nc4")
        self._initialize_dirs()

    def _load_kwargs(self, kwargs: dict[str, Any]):
        self.nc_config = kwargs.get("nc_config", None)
        self.record_out = kwargs.get("record_out", lambda x: "")
        self.region_out = kwargs.get("region_out", lambda x: "")
        self.out_out = kwargs.get("out_out", lambda: "")

    def _load_all_regions(self):
        return {
            r: gpd.read_file(self.PATH_OF(f"{self.COUNTRIES_DIR}/{self.REGIONS[r]}"))
            for r in self.SELECTED_REGIONS
        }

    def _get_all_from_dir(self, path_to_dir: str, extension: str):
        for f in Path(path_to_dir).glob(extension):
            match extension:
                case "*.pkl":
                    yield f, load_from_pickle(f, self.CRS)
                case "*.json":
                    yield f, gpd.read_file(f, engine="pyogrio", use_arrow=True)
                case "*.nc4":
                    yield f, run(
                        f, self.PATH_OF(self.OUT_DIR), self.nc_config, self.verbose
                    )
                case "*.nc":
                    yield f, run(
                        f, self.PATH_OF(self.OUT_DIR), self.nc_config, self.verbose
                    )
                case _:
                    raise ValueError("Unsupported File Extension")

    def _layer_and_concat(
        self, path_to_dir: str, extension: str, regions: dict[str, Any]
    ):
        self._log(f"Joining and Concatenating GeoJsons")

        # Join satellite data across all selected regions
        out = defaultdict(list)
        for f, gdf in self._get_all_from_dir(path_to_dir, extension):
            for r, rdf in regions.items():
                gdf.to_crs(rdf.crs)
                joined = gpd.sjoin(rdf, gdf, how="left", predicate="intersects")
                aggregate = joined.dissolve(by=self.JOIN_ON, aggfunc=self.JOINS)
                aggregate.columns = [
                    "".join([s[1], s[0].title()]) if type(s) == tuple else s
                    for s in aggregate.columns
                ]

                if self.COMPRESS:
                    aggregate.to_pickle(
                        self.PATH_OF(
                            f"{self.OUT_DIR}/{r}_{self.region_out(self.REGIONS[r])}_{self.record_out(f)}.pkl"
                        )
                    )

                out[r] += [aggregate.drop("geometry", axis=1)]

        # Save all time series as numpy files.
        for r, rdf in out.items():
            rdf = np.squeeze(xarray.DataArray([rdf]).to_numpy())
            pth = self.PATH_OF(
                f"{self.OUT_DIR}/{r}_time_series_{self.DATA_DIR}_{self.out_out()}"
            )
            np.save(f"{pth}.npy", rdf)
            save_np_as_csv(f"{pth}.csv", rdf)

    def preprocess(self):
        self._layer_and_concat(
            self.PATH_OF(self.DATA_DIR), self.EXTENSION, self._load_all_regions()
        )

    def preprocess_multi(
        self, config_list: List[str], kwargs_list: Optional[List[dict]]
    ):
        """
        :param config_list: A list of config file names
        :param kwargs_list: A list of dictionaries mapping kwargs for each config
        """
        for i, c_f in enumerate(config_list):
            self._load_config(c_f)
            if kwargs_list:
                self._load_kwargs(kwargs_list[i])
            self.preprocess()
