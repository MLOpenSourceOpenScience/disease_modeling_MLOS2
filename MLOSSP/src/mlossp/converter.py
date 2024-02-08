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

import os
import json
from pathlib import Path

import geopandas
import numpy as np
import xarray as xr


def log(m, verbose):
    if verbose:
        print(m)


def netcdf2geojson(config_file, input_file, output_dir, verbose, max_records=None):
    """Converts NetCDF file to GeoJSON based on config"""
    conf = read_config(config_file)

    # Make sure output path exists
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
    output_file = Path(output_dir).joinpath(Path(input_file).stem + '.json')
    log(f'Reading {input_file}', verbose)

    # Read dataset and remove nan values
    local_dataset = xr.open_dataset(input_file, chunks='auto', decode_times=False)
    df = local_dataset.to_dataframe()
    df = df.dropna(how='any', axis=0).reset_index()

    # Limit records (generally for testing)
    if max_records is not None:
        df = df[0: int(max_records)]

    # Convert extents from 0 - 360 to -180 - 180
    if conf['is360'] is True:
        df[conf['lonVar']] = (df[conf['lonVar']] + 180) % 360 - 180

    log(df, verbose)
    data = {}

    # Set output values for magnitude and direction
    if conf.get('magnitudeVar') and conf.get('directionVar'):
        if conf['convertMagDir'] is True:
            log('Calculating u and v from magnitudeVar and directionVar', verbose)
            data['u'] = df.apply(lambda x: magdir2u(x[conf['magnitudeVar']], x[conf['directionVar']]), axis=1)
            data['v'] = df.apply(lambda x: magdir2v(x[conf['magnitudeVar']], x[conf['directionVar']]), axis=1)
        log('Using magnitudeVar and directionVar', verbose)
        data['magnitude'] = df[conf['magnitudeVar']]
        data['direction'] = df[conf['directionVar']]

    # Convert u and v components to magnitude and direction if desired
    if conf.get('uVar') and conf.get('vVar'):
        if conf['convertUV'] is True:
            log('Calculating magnitude and direction from uVar and vVar', verbose)
            data['magnitude'] = df.apply(lambda x: uv2magnitude(x[conf['uVar']], x[conf['vVar']]), axis=1)
            data['direction'] = df.apply(lambda x: uv2direction(x[conf['uVar']], x[conf['vVar']]), axis=1)
        log('Using uVar and vVar', verbose)
        data['u'] = df[conf['uVar']]
        data['v'] = df[conf['vVar']]

    # Pass through any specified extra variables
    if conf.get('extraVars'):
        extraVars = conf.get('extraVars')
        if isinstance(extraVars, str):
            extraVars = [extraVars]
        for var in extraVars:
            log(f'Including {var}', verbose)
            data[var] = df[var]

    # Convert to geopandas dataframe
    log('Converting to GeoJSON', verbose)
    gdf = geopandas.GeoDataFrame(data, geometry=geopandas.points_from_xy(df[conf['lonVar']], df[conf['latVar']]))
    gdf.crs = "EPSG:4326"
    log(gdf, verbose)

    return gdf


def read_config(config_file):
    """
    Parses a JSON dataset config file
    Parameters
    ----------
    config_file = path to configuration file
    """
    config = None
    with open(config_file) as config_f:
        config = json.load(config_f)
    return config


def uv2direction(u, v):
    """
    Calculates direction from u and v components
    Parameters
    ----------
    u = west/east direction
    v = south/north direction
    """
    direction = (270 - np.rad2deg(np.arctan2(v, u))) % 360
    return direction


def uv2magnitude(u, v):
    """
    Calculates magnitude from u and v components
    Parameters
    ----------
    u = west/east direction
    v = south/north direction
    """
    magnitude = np.sqrt(np.square(u) + np.square(v))
    return magnitude


def magdir2u(magnitude, direction):
    """
    Calculates u component from magnitude and direction
    Parameters
    ----------
    magnitude
    direction
    """
    rad = 4.0 * np.arctan(1)/180.
    u = -magnitude * np.sin(rad*direction)
    return u


def magdir2v(magnitude, direction):
    """
    Calculates v component from magnitude and direction
    Parameters
    ----------
    magnitude
    direction
    """
    rad = 4.0 * np.arctan(1)/180.
    v = -magnitude * np.cos(rad*direction)
    return v

def run(INPUT, OUTPUT, CONFIG, VERBOSE=True, MAX_RECORDS=None):
    return netcdf2geojson(CONFIG,
                           INPUT,
                           OUTPUT,
                           VERBOSE,
                           MAX_RECORDS)