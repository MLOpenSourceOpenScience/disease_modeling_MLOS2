# disease_modeling_MLOS2
------------------------------------------------------------------------------------------------------------------------------
## Project Description:

## Gathering Data:
1. For consistent results, a standardized directory structure must be defined as follows:
   1. A directory must be created to store all retrieved data.
   2. A subdirectory should be created for each data source, i.e., every category of satellite data.
   3. A subdirectory should be created for storing geographical data. 
2. The data used in this project is from NASA's Earth Observational Satellite Data. This data can be retrieved from Nasa Giovanni, NOAH, and MODIS by following the instructions on NASA Earthdata: https://www.earthdata.nasa.gov/. A detailed guide is available at https://towardsdatascience.com/getting-nasa-data-for-your-next-geo-project-9d621243b8f3.
   1. This project will use GPM, GLDAS, and NDVI data.
   2. Satellite data will come in HDF5, netCDF, and netCDF4 formats. Our use case only requires netCDF and netCDF4 data, hence, our code does not support other formats.
   3. Satellite data comes at varying resolutions. Our preprocessor implementation is resolution agnostic.
3. Geographical data in the form of GeoJSONs is also required.
   1. In our case, we will be using Sri Lanka's GeoJSON data retrieved from https://gadm.org/download_country.html. Simply download the desired resolution.

Below is a sample directory tree:
```text
- Data
    - Geographical
        - Sri_Lanka.json
    - GPM
        - GPM.nc
        - GPM2.nc
    - GLDAS
        - GLDAS.nc
        - GLDAS2.nc
    - NDVI
        - NDVI.nc4
        - NDVI2.nc4
```

## Preprocessing Data
1. Once your desired data is downloaded, configuration files must be made to specify how the data will be processed with our package mlossp.
2. Your preprocessing script and configuration files can be placed anywhere, so long as you provide the relative Path to your data.
3. Instructions and documentation to create these files can be found at https://pypi.org/project/mlossp/.

## Visualizing Data
1. To visualize the data, you must have saved pickle files during the preprocessing step by setting compress to True. These should be located within your designated out directory for each category of satellite data.
2. Download ffmpeg from https://ffmpeg.org/download.html.
   1. Set the variable ```plt.rcParams['animation.ffmpeg_path']``` to the absolute path to the ffmpeg.exe executable file that is in the ffmpeg directory.
3. Run the visualize.py script, providing the following command line arguments:
   1. -t - The title of your graph.
   2. -d - The path to your data directory as defined in the Gathering Data section.
   3. -c - The column name of the target feature you would like to visualize.
   4. -o - The path to the file where the resulting visualization will be saved.
   5. -f - The frames per second of your visualization. Lower fps will result in a slower animation.
   6. -l - The lower bound of the possible values of your target feature. Defaults to 0.
   7. -u - The upper bound of the possible values of your target feature. Defaults to 1.
   8. -g - The coordinate reference system used by your data. Defaults to EPSG:4326.
   9. -xl - The x label of the graph. Defaults to None.
   10. -yl - The y label of the graph. Defaults to None.
   11. -ll - The label for the graph's legend. Defaults to none.
   

Given the following directory structure:
```text
    - Data
        - Geographical
        - GPM
        - GPM_OUT
            - GPM_OUT1.pkl
            - GPM_OUT2.pkl
        - Visualizations
    - visualize.py
```

The following command line command can be run:
```text
python ./visualize.py -d Data\GPM_OUT -c Feature_Name -o Data\GPM_Visualization -f 1 -l 0 -u 10 -g EPSG:4326
```