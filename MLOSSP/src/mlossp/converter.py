# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert CF-compliant NetCDF files with vector attributes to GeoJSON
Modified by qiud1.
"""

import json

import geopandas as gpd
import xarray as xr


def log(m, verbose):
    if verbose:
        print(m)


def read_config(config_file):
    """
    Parses a JSON dataset config file
    Parameters
    ----------
    config_file = path to configuration file
    """
    with open(config_file) as config_f:
        config = json.load(config_f)
        return config


def get_nc(input_file, config_file, crs="EPSG:4326", verbose=True):
    """Converts NetCDF file to GeoJSON based on config"""
    log(f"Reading {input_file}", verbose)
    conf = read_config(config_file)

    local_dataset = xr.open_dataset(input_file, chunks="auto", decode_times=False)
    df = local_dataset.to_dataframe()
    df = df.dropna(how="any", axis=0).reset_index()

    if conf["is360"] is True:
        df[conf["lonVar"]] = (df[conf["lonVar"]] + 180) % 360 - 180

    log(df, verbose)
    data = {}

    # Extract desired features
    if extraVars := conf.get("extraVars"):
        if isinstance(extraVars, str):
            extraVars = [extraVars]
        for var in extraVars:
            log(f"Including {var}", verbose)
            data[var] = df[var]

    # Convert to geopandas dataframe
    log("Converting to GeoJSON", verbose)
    gdf = gpd.GeoDataFrame(
        data, geometry=gpd.points_from_xy(df[conf["lonVar"]], df[conf["latVar"]])
    )
    gdf.crs = crs

    return gdf


def get_csv(input_file, config, crs="EPSG:4326", verbose=True):
    log(f"Reading {input_file}", verbose)
    conf = read_config(config)

    gdf = gpd.read_file(input_file)
    gdf["geometry"] = gpd.points_from_xy(gdf[conf["lonVar"]], gdf[conf["latVar"]])

    if conf["is360"]:
        gdf[conf["lonVar"]] = (gdf[conf["lonVar"]] + 180) % 360 - 180

    # Filter attributes
    if filters := conf.get("filter"):
        for col, feat, kind in filters:
            match kind:
                case "e":
                    gdf = gdf.loc[gdf[col] == feat]
                case "n":
                    gdf = gdf.loc[gdf[col] != feat]
                case _:
                    raise ValueError("Unsupported filter type.")

    # Rename attributes
    if r_dict := conf.get("replace"):
        gdf.replace(r_dict, inplace=True)

    # Extract desired features
    if extraVars := conf.get("extraVars"):
        if isinstance(extraVars, str):
            extraVars = [extraVars]
        for col in gdf.columns:
            if col not in extraVars + ["geometry"]:
                gdf.drop(col, axis=1, inplace=True)

    gdf.crs = crs

    return gdf
