"""Running distributed (over nodes) transformation of gribdata to seqrecord.

    No join operation is performed.

Returns:
    _type_: _description_

Yields:
    _type_: _description_
"""
import sys
import os
import xarray as xr
import psutil
from seqrecord.weather.seqrecord import WSeqRecord
from seqrecord.utils import distribute_loads
import os
from typing import List, Tuple, Iterator
from tqdm import tqdm
import torch.distributed as dist
from itertools import islice
from seqrecord.weather.constants import NAME_TO_VAR

NUM_PROCESS_PER_NODE = 12  # make sure each node can handle processing 12 processes in the same time (months)

single_vars = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "2m_dewpoint_temperature",
    # "total_precipitation",
    "skin_temperature",
    "sea_surface_temperature",
    "total_cloud_cover",
    "total_column_water_vapour",
]

pressure_vars = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]

LEVELS = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]
#
HOURS_PER_MONTH = 744
HOURS_PER_SHARD = 384


def frame_generator(year, local_rank, path):
    """Process one month's data of $year.

    Args:
        year (_type_): _description_
        local_rank (_type_): _description_
        path (_type_): _description_

    Yields:
        _type_: _description_
    """
    month = local_rank + 1  # local_rank : [0-12), month: [1, 13)
    ds_files = {}
    num_frames_in_month = -1
    for var in single_vars:
        var_path = os.path.join(path, f"{year}", f"{month:02d}", f"{var}_0.grib")
        code = NAME_TO_VAR[var]
        ds_files[var] = xr.open_dataset(var_path)[code]
        if num_frames_in_month == -1:
            num_frames_in_month = ds_files[var].shape[0]
    for var in pressure_vars:
        for level in LEVELS:
            var_path = os.path.join(
                path,
                f"{year}",
                f"{month:02d}",
                f"{var}_{level}.grib",
            )
            code = NAME_TO_VAR[var]
            ds_files[f"{var}_{var}"] = xr.open_dataset(var_path)[code]

    pbar = (
        tqdm(
            range(num_frames_in_month),
            desc=f"formating frames in month {month}",
            total=num_frames_in_month,
        )
        if local_rank == 0
        else range(num_frames_in_month)
    )
    if local_rank == 0:
        print(
            f"/n When rank 0 opens one month's files, the RAM usage (GB): {psutil.virtual_memory()[3]/1000000000}"
        )
    for frame_idx in pbar:
        np_vars = {}
        for key, item in ds_files.items():
            np_vars[key] = item[frame_idx].to_numpy()
        yield np_vars


def grib2np(year, local_rank, config):
    """
    Convert grib files to numpy arrays and save them to disk.
    """
    sub_wseqrecord = WSeqRecord(
        os.path.join(
            config["wseqrecord_dir"],
            f"{year}",
            WSeqRecord.subfolder_name(local_rank, NUM_PROCESS_PER_NODE),
        ),
        local_cache_dir=os.path.join(
            config["local_cache_dir"],
            f"{year}",
            WSeqRecord.subfolder_name(local_rank, NUM_PROCESS_PER_NODE),
        ),
    )

    sub_wseqrecord.put_frame(
        frame_generator(year, local_rank, config["grib_dataset_dir"])
    )
    sub_wseqrecord.dump_record(rank=local_rank)


# def initialize():
#     dist.init_process_group(backend="gloo")


def main(num_nodes: int):
    """Each node formats several years' data (one year at a time). Within each node, spawns 12 processes where
    each process takes care of one month's data.

    Args:
        dataset_mount_dir (str): _description_
        num_nodes (int): _description_
    """
    # initialize()
    # dataset: 5.625deg_equally_np_all_levels,  1.40625deg_equally_np_all_levels(ultimate goal)
    # "/datadrive/weatherdatastorage2/datasets",  # "/mnt/data",
    dataset_mount_dir = "/mnt"
    DEFAULT_CONFIG = {
        "wseqrecord_dir": os.path.join(
            dataset_mount_dir,
            f"era5seqrecord/aml_dist/",
        ),
        "local_cache_dir": "~/record_cache",
        "grib_dataset_dir": os.path.join(dataset_mount_dir, "era5"),
    }

    config = DEFAULT_CONFIG

    source_generator = range(1979, 2016)  # full-dataset: 1979-2023
    dividens = distribute_loads(len(source_generator), num_nodes)
    # each node processes one year's data. torchrun will spawn 12 processes within each node, and each process (identified by local_rank) processes one month's data.
    # ref: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-distributed-gpu
    node_rank = int(os.environ["NODE_RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    sub_datasource_generator = islice(
        source_generator, dividens[node_rank][0], dividens[node_rank][1]
    )
    for year in sub_datasource_generator:
        print(
            f"Transforming {year}'s data on the {node_rank}-th node with local rank {local_rank}"
        )
        grib2np(
            year,
            local_rank,  # local_rank: range(0, 12) corresponds month range(1, 13)
            config,
        )
    print(
        f"Transforming {year}'s data on the {node_rank}-th node with local rank {local_rank} is done."
    )
    # # combine sub-seqrecord
    # record = WSeqRecord.gather_subseqrecords(
    #     config["wseqrecord_dir"], world_size=config["num_processes"]
    # )

    # record.dump()


if __name__ == "__main__":
    num_nodes = int(sys.argv[1])
    main(num_nodes)
