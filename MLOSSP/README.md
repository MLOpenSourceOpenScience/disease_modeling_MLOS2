# Satellite Data Preprocessor

A package built for preprocessing netCDF, netCDF4, or GeoJSON files using spatial joins and concatenation to create time-series numpy and csv datasets.

### Keyword Arguments
* nc_config - Path to the configuration file used for loading nc files (only provide this if you are loading netCDF files.)
* record_out - A function to generate a label from a file name.
* region_out - A function to generate a label from a region data file name.
* out_out - A function that returns a string that is added to your output file.

### Usage:
* Define a directory structure and create the required configuration files. (see Configuration Files)
* Create an instance of Preprocessor and provide the path to a config json file along with keyword arguments.
* Call Preprocessor.preprocess to preprocess data according to the provided config.
* To define configurations for different data sources, use Preprocessor.preprocess_multi. Config_list[i] and Kwargs_list[i] will apply to the ith data source.

```python
import mlossp as sp

config_path = "config.json"
nc_config_path = "nc_config.json"
f = lambda : "_1"

preprocessor = sp.Preprocessor(
    config_path,
    nc_config=nc_config_path,
    out_out=f
)
preprocessor.preprocess()

config_list = ["config_1.json", "../configs/config_2.json"]
kwargs_list = [{"nc_config": nc_config_path, "out_out": f}, {}]
preprocessor.preprocess_multi(config_list, kwargs_list)
```

### Configuration Files

Every category of data - that is, data within a subdirectory - should have a json configuration file. The following keys should be defined within the configuration file:

* data_path - The path to the data directory, data_dir, regions_dir, and out_dir are the subdirectories of data_path that contain the satellite data, location data, and ouput data respectively.
* compress - Set to True to store all intermediate geojson files. Must be set to True for visualizations.
* crs - A geopandas supported coordinate reference system.
* regions_file_map - A mapping of region name to their file name in regions_dir.
* selected_regions - The list of regions to extract from the satellite data.
* join_on - The feature that duplicate data entries will be aggregated by.
* joins - The aggregate metrics that will be stored in the aggregate row. Each column will be named as the concatenation of the aggregate function and the original feature.
* file_extension - The format of all data being layered. This can be either *.pkl, *.json, *.nc4, or *.nc.

Given the following directory structure:
```text
- Data
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
  "compress": true,
  "crs": "EPSG:4326",
  "data_path": "Data/",
  "regions_dir": "Countries",
  "data_dir": "EO",
  "out_dir": "EO_OUT",
  "regions_file_map":  {
    "Sri Lanka": "gadm41_LKA_1.json",
    "USA": "gadm41_USA_1.json"
  },
  "selected_regions": ["Sri Lanka"],
  "join_on": "NAME_1",
  "joins": {
    "precipitationCal": ["mean", "min", "max"]
  },
  "file_extension": "*.nc"
}
```

### Configuration Files for netCDF data
To use this package with netCDF data, a separate configuration must be provided for each data source. A sample configuration is shown below.
```json
{
	"latVar":"latitude",
	"lonVar":"longitude",
	"is360":false,
	"extraVars": ["NDVI"]
}
```

* latVar: The name of the latitude column
* lonVar: The name of the longitude column
* is360: Set to true if the netCDF contains data within 0 - 360 degrees
* extraVars: Desired features to be retrieved from the netcdf file.

For additional keys, see https://github.com/podaac/netcdf_to_geojson_vectors.

### Citations
This package incorporates modified code from https://github.com/podaac/netcdf_to_geojson_vectors, which is licensed under Apache 2.0. You may obtain a copy of the license at: http://www.apache.org/licenses/LICENSE-2.0.