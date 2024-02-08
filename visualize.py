#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Define the absolute path to ffmpeg if it is not on your PATH
plt.rcParams['animation.ffmpeg_path'] ='C:\\CS\\disease_modeling_MLOS2\\ffmpeg\\bin\\ffmpeg.exe'


def load_from_pickle(path, crs):
    p_df = pd.read_pickle(path)
    return gpd.GeoDataFrame(p_df, geometry=p_df.geometry, crs = crs)


def load_all_from_pickle(path, pattern, crs):
    return [load_from_pickle(f, crs) for f in Path(path).glob(pattern)]


def visualize(DATA_DIR, OUT_DIR, COLUMN, MIN_VALUE, MAX_VALUE, FPS, CRS):
    gdf_list = load_all_from_pickle(DATA_DIR, "*.pkl", CRS)

    def idx_generator(gdf_list):
        for i in range(len(gdf_list)): yield i

    def update(frame):
        plt.clf()
        data = gdf_list[frame]
        data.plot(column=COLUMN, cmap='Greens', vmin=MIN_VALUE, vmax=MAX_VALUE, legend=True, ax=plt.gca())
        plt.title(f'Date: {frame}')
    
    # Create the animation
    animation = FuncAnimation(plt.gcf(), update, frames=idx_generator(gdf_list), repeat=False)
    animation.save(f'{OUT_DIR}.mp4', writer='ffmpeg', fps=FPS)


# Define command line arguments
parser = argparse.ArgumentParser(description='Visualize a sequence of pickled GeoJSONs')
parser.add_argument(
    '-d',
    '--data_dir',
    default='',
    dest='DATA_DIR',
    action='store',
    help='Directory where pkl files are stored.')
parser.add_argument(
    '-c',
    '--column',
    dest='COLUMN',
    help='Target column to be visualized.',
    action='store')
parser.add_argument(
    '-o',
    '--out',
    default='',
    dest='OUT_DIR',
    help='Path to Output Directory',
    action='store')
parser.add_argument(
    '-f',
    '--fps',
    default=5,
    dest='FPS',
    help='Visualization frames per second.',
    action='store')
parser.add_argument(
    '-l',
    '--lower_bound',
    default=0,
    dest='MIN_VALUE',
    help='Lower bound of target feature.',
    action='store')
parser.add_argument(
    '-u',
    '--upper_bound',
    default=1,
    dest="MAX_VALUE",
    help="Upper bound for target feature.",
    action='store'
)
parser.add_argument(
    '-g',
    '--crs',
    default="EPSG:4326",
    dest="CRS",
    help="Coordinate Reference System of the given data.",
    action='store'
)

# Make visualization
args = parser.parse_args()
if not args.COLUMN:
    raise ValueError("Target Column Must Be Defined")
visualize(args.DATA_DIR,
          args.OUT_DIR,
          args.COLUMN,
          args.MIN_VALUE,
          args.MAX_VALUE,
          args.FPS,
          args.CRS)