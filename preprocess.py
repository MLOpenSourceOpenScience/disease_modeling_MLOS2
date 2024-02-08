import mlossp as sp


REGION = lambda x: x.split('.')[-2][-1]
NDVI = lambda x:str(x).split('_')[-2]
GPM = lambda x: str(x).split('.')[-4][:8]
GLDAS = lambda x: str(x).split('_')[2][:~4]

CONFIG_LIST = ["NDVI_config.json", "GPM_config.json", "GLDAS_config.json"]
KWARGS_LIST = [
    {
        "record_out": NDVI,
        "region_out": REGION,
        "nc_config": "NDVI_nc.json"
    },
    {
        "record_out": GPM,
        "region_out": REGION,
        "nc_config": "GPM_nc.json"
    },
    {
        "record_out": GLDAS,
        "region_out": REGION,
        "nc_config": "GLDAS_nc.json"
    }
]


def process_one(CONFIG_FILE, KWARGS_IDX):
    preprocessor = sp.Preprocessor(
        config_file = CONFIG_FILE,
        **KWARGS_LIST[KWARGS_IDX],
    )
    preprocessor.preprocess()


def process_all(CONFIG_LIST, KWARGS_LIST):
    preprocessor = sp.Preprocessor(config_file=CONFIG_LIST[0])
    preprocessor.preprocess_multi(CONFIG_LIST, KWARGS_LIST)


if __name__ == "__main__":
    # process_one("GLDAS_config.json", 2)
    process_all(CONFIG_LIST, KWARGS_LIST)