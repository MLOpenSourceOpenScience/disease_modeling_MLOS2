# disease_modeling_MLOS2
------------------------------------------------------------------------------------------------------------------------------
## Project Description:

## Gathering And Visualizing Data:
1. To access satellite data (e.g., prescriptions), utilize resources from NASA's satellite collections such as Giovanni. Identify and visit the specific resource website for the desired data.
2. For this project, we used links from NASA for NDVI, GPM, and GLDAS. One can always follow the respective instructions on their websites. The data resolution used in our project is flexible; any resolution size is compatible.
3. After obtaining the data, organize it by creating a main directory and then sub-directories within it, each named after the specific data category (e.g., Prescription). It's essential to maintain the data format as netCDF (nc4).
4. Repeat the organization process for additional data types like temperature and humidity. Our custom software package allows for the serialization of the downloaded data files, targeting specific geographical areas.
5. Once the data is preprocessed and stored in the respective subfolders for each data type and area, one can use our provided pip package for preprocessing it into customized areas with a given latitute and longitute bound. 
6. For any data type (NDVI, GLDAS, etc.), after preprocessing, you can apply the layering process using our package. In the script, specify the paths for the data directory, the subdirectory for the targeted data, the output directory, and the data file extension. Currently, we support Python pickle and GeoJSON file formats, emphasizing pickle for its compression benefits. Additionally, define the Coordinate Reference System (CRS).
7. After setting up all necessary variables in the script, installation of the FFmpeg package is required for data visualization. The final output will be an MP4 video showcasing the data.
