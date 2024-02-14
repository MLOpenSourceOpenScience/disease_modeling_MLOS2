#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors


# Define the absolute path to ffmpeg if it is not on your PATH

plt.rcParams["animation.ffmpeg_path"] = (
    "C:\\CS\\disease_modeling_MLOS2\\ffmpeg\\bin\\ffmpeg.exe"
)

# If APPLY is defined, applies a transformation to the designated column. Otherwise, leave as None.

APPLY = None
NDVI_COLORMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", [(0, "gray"), (0.5, "white"), (1, "green")]
)
GPM_COLORMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "",
    [
        (0, "white"),
        (0.1, "lightsteelblue"),
        (0.2, "cornflowerblue"),
        (0.5, "royalblue"),
        (1, "midnightblue"),
    ],
)
GLDAS_COLORMAP = "Reds"
POP_DEN_COLORMAP = "YlOrRd"
CMAP = POP_DEN_COLORMAP


def load_from_pickle(path, crs):
    p_df = pd.read_pickle(path)
    return gpd.GeoDataFrame(p_df, geometry=p_df.geometry, crs = crs)


def get_date(f):
    ymd = str(f)[:-4].split('_')[-1]
    return f"{ymd[4:6]}/{ymd[6:]}/{ymd[:4]}"


def load_all_from_pickle(path, pattern, crs):
    return [(get_date(f), load_from_pickle(f, crs)) for f in Path(path).glob(pattern)]


def visualize(
    data_dir,
    out_dir,
    column,
    min_value,
    max_value,
    fps,
    crs,
    x_label,
    y_label,
    legend_label,
    title,
):
    gdf_list = load_all_from_pickle(data_dir, "*.pkl", crs)

    def idx_generator(gdf_list):
        for i in range(len(gdf_list)):
            yield i

    def update(frame):
        plt.clf()
        f, data = gdf_list[frame]
        if APPLY:
            data[column] = data[column].apply(APPLY)
        data.plot(
            column=column,
            cmap=CMAP,
            vmin=min_value,
            vmax=max_value,
            legend=True,
            legend_kwds={"label": f"{legend_label}"},
            ax=plt.gca(),
            edgecolor="black",
        )
        plt.title(f"{title} {f}\n")
        plt.xlabel(f"{x_label}")
        plt.ylabel(f"{y_label}")

    # Create the animation

    animation = FuncAnimation(
        plt.gcf(), update, frames=idx_generator(gdf_list), repeat=False
    )
    animation.save(f"{out_dir}.mp4", writer="ffmpeg", fps=fps)


# Define command line arguments
    
parser = argparse.ArgumentParser(description='Visualize a sequence of pickled GeoJSONs')
parser.add_argument(
    '-t',
    '--title',
    default="",
    dest="TITLE",
    help="A title for the visualization.",
    action='store'
)
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
parser.add_argument(
    '-xl',
    '--xlabel',
    default="Longitude",
    dest="XLABEL",
    help="A label for the x axis of the data.",
    action='store'
)
parser.add_argument(
    '-yl',
    '--ylabel',
    default="Latitude",
    dest="YLABEL",
    help="A label for the y axis of the data.",
    action='store'
)
parser.add_argument(
    '-ll',
    '--legendlabel',
    default="",
    dest="LEGEND",
    help="A label for the legend.",
    action='store'
)


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.COLUMN:
        raise ValueError("Target Column Must Be Defined")
    visualize(
        args.DATA_DIR,
        args.OUT_DIR,
        args.COLUMN,
        float(args.MIN_VALUE),
        float(args.MAX_VALUE),
        int(args.FPS),
        args.CRS,
        args.XLABEL,
        args.YLABEL,
        args.LEGEND,
        args.TITLE,
    )