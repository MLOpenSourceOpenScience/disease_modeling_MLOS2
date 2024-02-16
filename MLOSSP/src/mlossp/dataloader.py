from typing import List, Any

import pandas as pd
import geopandas as gpd
from functools import partial
from pathlib import Path

from .converter import get_nc, get_csv


class DataLoader:
    def __init__(
        self,
        path_to_dir: str,
        extension: str,
        crs: str,
        nc_config: str,
        verbose: bool,
        chunk_size: int = 1,
    ):
        super().__init__()

        self.crs = crs
        self.nc_config = nc_config
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.f_id = 0
        self.files_to_load = [f for f in Path(path_to_dir).glob(extension)]
        self.load = partial(self.__load, extension)

    def __iter__(self):
        self.f_id = 0
        return self

    def __next__(self):
        if self.f_id >= len(self.files_to_load):
            raise StopIteration
        self.f_id += self.chunk_size

        if self.chunk_size != 1:
            return self.__load_chunked(
                [
                    self.load(f)
                    for f in self.files_to_load[
                        self.f_id - self.chunk_size : min(self.f_id, len(self))
                    ]
                ]
            )

        return self.load(self.files_to_load[self.f_id - 1])

    def __len__(self):
        return len(self.files_to_load)

    @staticmethod
    def __load_from_pickle(path: Path, crs: str):
        p_df = pd.read_pickle(path)
        return gpd.GeoDataFrame(p_df, geometry=p_df["geometry"], crs=crs)

    @staticmethod
    def __load_chunked(f: List[Any]):
        gdf = pd.concat([d for _, d in f]).drop("geometry", axis=1)
        gdf = gdf.groupby(gdf.index).mean()
        gdf = gpd.GeoDataFrame(gdf, geometry=f[0][1].geometry)
        return f[0][0], gdf.to_crs(f[0][1].crs)

    def __load(self, extension: str, f: Path):
        match extension:
            case "*.pkl":
                return f, self.__load_from_pickle(f, self.crs)
            case "*.json":
                return f, gpd.read_file(f, engine="pyogrio", use_arrow=True)
            case "*.geojson":
                return f, gpd.read_file(f, engine="pyogrio", use_arrow=True)
            case "*.nc4":
                return f, get_nc(f, self.nc_config, self.crs, self.verbose)
            case "*.nc":
                return f, get_nc(f, self.nc_config, self.crs, self.verbose)
            case "*.csv":
                return f, get_csv(f, self.nc_config, self.crs, self.verbose)
            case _:
                raise ValueError("Unsupported File Extension")
