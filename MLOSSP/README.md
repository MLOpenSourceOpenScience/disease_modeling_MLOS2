# Satellite Data Preprocessor

A package built for preprocessing netCDF, netCDF4, GeoJSON, or **atomic** CSV files using spatial joins and concatenation to create time-series numpy and csv datasets.

### Keyword Arguments

| Kwarg        | Description                                                                                                          |
|--------------|----------------------------------------------------------------------------------------------------------------------|
| extra_config | Path to the configuration file used for loading nc files. (Only provide this if you are loading netCDF or CSV files. |
| record_out   | A function to generate a label from a file name.                                                                     |
| region_out   | A function to generate a label from a region file name.                                                              |
| suffix_out   | A function that returns a string to be appended to the output file.                                                  |

### Usage:
* Define a directory structure and create the required configuration files. (see Configuration Files)
* Create an instance of Preprocessor and provide the path to a config json file along with keyword arguments.
* Call Preprocessor.preprocess to preprocess data according to the provided config.
* To define configurations for different data sources, use Preprocessor.preprocess_multi. Config_list[i] and Kwargs_list[i] will apply to the ith data source.

```python
import mlossp as sp

config_path = "config.json"
extra_config_path = "nc_config.json"
f = lambda : "_1"

preprocessor = sp.Preprocessor(
    config_path,
    extra_config=extra_config_path,
    suffix_out=f
)
preprocessor.preprocess()

config_list = ["config_1.json", "../configs/config_2.json"]
kwargs_list = [{"extra_config": extra_config_path, "out_out": f}, {}]
preprocessor.preprocess_multi(config_list, kwargs_list)
```

### Configuration Files

Every category of data, i.e., data within a directory should have a json configuration file. The following keys should be defined within the configuration file:

| Key              | Description                                                                                             | Optional |
|------------------|---------------------------------------------------------------------------------------------------------|----------|
| regions_dir      | The path to the directory where the geographical data is stored                                         | No       |
| data_dir         | The path to the directory where the satellite data is stored                                            | No       |
| out_dir          | The path to the directory where the output files will be stored                                         | No       |
| crs              | A geopandas supported coordinate reference system                                                       | Yes      |
| regions_file_map | A mapping of region name to their file path                                                             | No       |
| selected_regions | The list of regions to extract from the satellite data. Must be in regions_file_map.                    | No       |
| join_on          | The feature that duplicate data entries in a timestamp will be aggregated on                            | No       |
| joins            | The aggregate metrics that will be stored from join_on's aggregation.                                   | No       |
| file_extension   | The format of all data being layered. Can be one of *.pkl, *.json, *.geojson, *.nc4, or *.nc            | No       |
| compress_to      | The path to the directory where visualization files will be stored.                                     | Yes      |
| chunk            | This will aggregate every ```chunk``` files together. E.g. 365 chunked into 7 will result in 53 chunks. | Yes      |

Given the following directory structure:
```text
- Data
    - Visualizations
    - EO
        - data1.nc
        - data2.nc
    - EO_OUT
    - Countries
        - gadm41_LKA_1.json
        - gadm41_USA_1.json
- preprocesser.py
```

And region and satellite data with the following features:
```text
Region: NAME, NAME_1, NAME_2, POINT, ...
Satellite Data: longitude, latitude, precipitationCal, ...
```

A sample configuration would be defined as follows:
```json
{
  "compress_to": "Data/Visualizations",
  "crs": "EPSG:4326",
  "regions_dir": "Data/Countries",
  "data_dir": "Data/EO",
  "out_dir": "Data/EO_OUT",
  "regions_file_map":  {
    "Sri Lanka": "gadm41_LKA_1.json",
    "USA": "gadm41_USA_1.json"
  },
  "selected_regions": ["Sri Lanka"],
  "join_on": "NAME_1",
  "joins": {
    "precipitationCal": ["mean", "min", "max"]
  },
  "file_extension": "*.nc",
  "chunk": 5
}
```

### Preprocessing netCDF data
To use this package with netCDF data, a separate configuration must be provided for each data source. A sample configuration is shown below.
```json
{
	"latVar":"latitude",
	"lonVar":"longitude",
	"is360":false,
	"extraVars": ["NDVI"]
}
```

#### Configuration Keys:

| Key       | Description                                                    |
|-----------|----------------------------------------------------------------|
| latVar    | The name of the latitude column                                |
| lonVar    | The name of the longitude column                               |
| is360     | Set to true if the netCDF contains data within 0 - 360 degrees |
| extraVars | Desired features to be retrieved from the netcdf file.         |

### Preprocessing CSV data
To use this package to preprocess CSV data, a configuration file must be defined for each data source.
CSV data provided will be aggregated across the given time period; Information across higher temporal resolutions will be **lost**. 

The number of csv files provided will be the final number of timestamps.
If you would like to atomicize your csv files, see [Formatting CSVs](#Formatting CSVs).

```json
{
	"latVar":"Lattitude",
	"lonVar":"Longitude",
	"is360":false,
	"filter": [["Disease Name", "Dengue Fever", "e"], ["Location Name", "SRILANKA", "n"], ["Location Name", "Kalmune", "n"]],
	"extraVars": ["Cases", "Location Name"]
}
```

#### Configuration Keys:

| Key       | Description                                                                                                                                                                                      |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| latVar    | The name of the latitude column                                                                                                                                                                  |
| lonVar    | The name of the longitude column                                                                                                                                                                 |
| is360     | Set to true if the longitude and latitude are measured in 360 degrees                                                                                                                            |
| filter    | A list of length 3 arrays of [Column Name, Target Value, Equals (e) or Not Equals (n)]. All rows whose column name has a value that is either equal or not equal to the target will be excluded. |
| extraVars | Features to keep in the output.                                                                                                                                                                  |

### Formatting CSVs
CSVs are inherently designed to support tabular, 2D data. Hence, storing higher-dimensional data such as tensors and ndarrays as CSVs results in inconsistent formats.

We offer ```mlossp.formatters``` as a solution to parse a (or a list of) 3D CSV files into a 3D numpy tensor, which is formatted then written into a .npy file and a .csv file that is supported by ```align_csv``` and ```align_npy```. 

### Citations
This package incorporates modified code from https://github.com/podaac/netcdf_to_geojson_vectors, which is licensed under Apache 2.0. You may obtain a copy of the license at: http://www.apache.org/licenses/LICENSE-2.0.