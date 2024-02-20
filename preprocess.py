import mlossp as sp


REGION = lambda x: x.split(".")[-2][-1]
NDVI = lambda x: str(x).split("_")[-2]
GPM = lambda x: str(x).split(".")[-4][:8]
GLDAS = lambda x: str(x).split("_")[-1][4:12]
CONFIG_LIST = [
    "Configs/NDVI_config.json",
    "Configs/GPM_config.json",
    "Configs/GLDAS_config.json",
    "Configs/POP_DEN_config.json",
]
KWARGS_LIST = [
    {
        "record_out": NDVI,
        "region_out": REGION,
        "extra_config": "Configs/NDVI_nc.json",
        "suffix_out": lambda: "NDVI",
    },
    {
        "record_out": GPM,
        "region_out": REGION,
        "extra_config": "Configs/GPM_nc.json",
        "suffix_out": lambda: "GPM",
    },
    {
        "record_out": GLDAS,
        "region_out": REGION,
        "extra_config": "Configs/GLDAS_nc.json",
        "suffix_out": lambda: "GLDAS",
    },
    {"region_out": REGION, "suffix_out": lambda: "POP_DEN"},
]


def process_one(config_file, kwargs_idx):
    preprocessor = sp.Preprocessor(
        config_file=config_file,
        **KWARGS_LIST[kwargs_idx],
    )
    preprocessor.preprocess()


def process_all(config_list, kwargs_list):
    preprocessor = sp.Preprocessor(config_file=config_list[0], verbose=False)
    preprocessor.preprocess_multi(config_list, kwargs_list)


if __name__ == "__main__":
    process_all(CONFIG_LIST, KWARGS_LIST)
    sp.Preprocessor.align_npy("Data/OUT", "Data/OUT/sri_lanka_full")