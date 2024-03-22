# Disease Modeling
------------------------------------------------------------------------------------------------------------------------------
Data preprocessing and model training for near real time disease forecasting.

## Installation:

Download this repository.
```shell
git clone https://github.com/MLOpenSourceOpenScience/disease_modeling_MLOS2
cd disease_modeling_MLOS2
```

Set up a python environment.
```shell
conda env create -f environment.yml
conda activate dm_mlos2
```

## Gathering Data:
1. For consistent results, a standardized directory structure must be defined as follows:
   * A directory should be created for each data source, i.e., every category of satellite data.
   * A directory should be created for storing geographical data. 
2. The data used in this project is from NASA's Earth Observational Satellite Data, specifically Giovanni, NOAH, and MODIS by following the instructions on [NASA Earthdata](https://www.earthdata.nasa.gov/). A detailed guide is available [here](https://towardsdatascience.com/getting-nasa-data-for-your-next-geo-project-9d621243b8f3).
   * This project will use GPM, GLDAS, and NDVI data.
   * Satellite data will come in HDF5, netCDF, and netCDF4 formats. Our code does not support other formats.
   * Satellite data comes at varying resolutions; Our preprocessor is resolution agnostic.
3. Geographical data in the form of GeoJSONs is also required.
   * In our case, we will be using Sri Lanka's GeoJSON data retrieved from [gadm](https://gadm.org/download_country.html). Simply download the desired resolution.
4. Disease data will be obtained from our NLP team. Their work is available [here](https://github.com/MLOpenSourceOpenScience/disease_data_parser_MLOS2).
   * We provide solutions for integrating their data into the dataset as part of our [package](https://pypi.org/project/mlossp/). 

Below is a sample directory tree:
```text
├─ Data
|   ├─ Geographical
|   |   ├─ Sri_Lanka.json
|   ├─ GPM
|   |   ├─ GPM.nc
|   |   ├─ GPM2.nc
|   ├─ GLDAS
|   |   ├─ GLDAS.nc
|   |   ├─ GLDAS2.nc
|   ├─ NDVI
|   |   ├─ NDVI.nc4
|   |   ├─ NDVI2.nc4
```

## Preprocessing Data
1. Once data is downloaded, configuration files must be made to specify how the data will be processed.
2. Your preprocessing script and configuration files can be placed anywhere, so long as you provide the relative Path to your data.
3. Documentation for these files is available [here](https://github.com/MLOpenSourceOpenScience/disease_modeling_MLOS2/tree/main/MLOSSP).

## Visualizing Data
1. To visualize the data, you must have saved pickle files during the preprocessing step by defining compress_to in your config file.
2. Download [ffpmeg](https://ffmpeg.org/download.html).
3. Set ```plt.rcParams['animation.ffmpeg_path']``` in visualize.py to the absolute path to ffmpeg.exe in the ffmpeg directory.
4. Run the visualize.py script, providing the following command line arguments:

| Flag | Description                                          | Default               |
|------|------------------------------------------------------|-----------------------|
| -t   | Graph Title                                          | None                  |
| -d   | Path to your pickle files.                           | The working directory |
| -c   | Column name of the target feature for visualization. | The working directory |
| -o   | Output file path.                                    | The working directory |
| -f   | Frames per second of the visualization.              | 5                     |
| -l   | Lower bound of feature value.                        | 0                     |
| -u   | Upper bound of feature value.                        | 1                     |
| -g   | The data's coordinate reference system.              | EPSG:4326             |
| -xl  | The graph's x axis label.                            | Longitude             |
| -yl  | The graph's y axis label.                            | Latitude              |
| -ll  | The graph's legend label.                            | None                  |

Given the following directory structure:
```text
    ├─ Data
    |   ├─ Geographical
    |   ├─ GPM
    |   ├─ GPM_OUT
    |       ├─ GPM_OUT1.pkl
    |       ├─ GPM_OUT2.pkl
    |   ├─ Visualizations
    ├─ visualize.py
```

The following command line command can be run:
```text
python ./visualize.py -d Data\GPM_OUT -c Feature_Name -o Data\GPM_Visualization -f 1 -l 0 -u 10 -g EPSG:4326
```
