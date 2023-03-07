import os
import math
import numpy as np
import xarray as xr
import psutil
from seqrecord.weather.seqrecord import WSeqRecord
from seqrecord.utils import distribute_loads
import os
from typing import List, Tuple, Iterator
import click
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
from itertools import islice
from seqrecord.weather.constants import NAME_TO_VAR

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


def frame_generator(rank: int, path, year, month_gen: Iterator):

    for m in month_gen:
        ds_files = {}
        num_frames_in_month = -1
        for var in single_vars:
            var_path = os.path.join(path, f"{year}", f"{m:02d}", f"{var}_0.grib")
            code = NAME_TO_VAR[var]
            ds_files[var] = xr.open_dataset(var_path)[code]
            if num_frames_in_month == -1:
                num_frames_in_month = ds_files[var].shape[0]
        for var in pressure_vars:
            for level in LEVELS:
                var_path = os.path.join(
                    path,
                    f"{year}",
                    f"{m:02d}",
                    f"{var}_{level}.grib",
                )
                code = NAME_TO_VAR[var]
                ds_files[f"{var}_{level}"] = xr.open_dataset(var_path)[code]

        pbar = (
            tqdm(
                range(num_frames_in_month),
                desc=f"formating frames in month {m}",
                total=num_frames_in_month,
            )
            if rank == 0
            else range(num_frames_in_month)
        )
        if rank == 0:
            print(
                f"/n When rank 0 opens one month's files, the RAM usage (GB): {psutil.virtual_memory()[3]/1000000000}"
            )
        for frame_idx in pbar:
            np_vars = {}
            for key, item in ds_files.items():
                np_vars[key] = item[frame_idx].to_numpy()
            yield np_vars


def grib2np(rank, world_size, config, year, month_gen):
    """
    Convert grib files to numpy arrays and save them to disk.
    """
    sub_wseqrecord = WSeqRecord(
        recorddir=os.path.join(
            config["wseqrecord_dir"],
            WSeqRecord.subfolder_name(rank, world_size),
        ),
        local_cache_dir=os.path.join(
            config["local_cache_dir"],
            WSeqRecord.subfolder_name(rank, world_size),
        ),
    )

    sub_wseqrecord.put_frame(
        frame_generator(rank, config["grib_dataset_dir"], year, month_gen)
    )
    print("rank", rank, " finished, dummping metadata!")
    sub_wseqrecord.dump_record(rank=rank)


@click.command()
@click.option(
    "--dataset-mount-dir", type=str, default="/datadrive/azure_storage/weathereastus"
)
@click.option("--local-cache-dir", type=str, default="~/record_cache")
@click.option("--year", type=int, required=True)
@click.option("--num-processes", type=int, default=12)
def main(dataset_mount_dir: str, local_cache_dir: str, year: int, num_processes: int):
    # dataset: 5.625deg_equally_np_all_levels,  1.40625deg_equally_np_all_levels(ultimate goal)
    # "/datadrive/weatherdatastorage2/datasets",  # "/mnt/data",
    DEFAULT_CONFIG = {
        "num_processes": num_processes,
        "local_cache_dir": local_cache_dir,
        "wseqrecord_dir": os.path.join(
            dataset_mount_dir,
            f"era5seqrecord/test/{year}",
        ),
        "grib_dataset_dir": os.path.join(dataset_mount_dir, "era5"),
    }

    config = DEFAULT_CONFIG

    month_generator = range(1, 13)
    dividens = distribute_loads(12, config["num_processes"])
    processes = []
    for i in range(config["num_processes"]):
        sub_month_generator = islice(month_generator, dividens[i][0], dividens[i][1])
        p = Process(
            target=grib2np,
            args=(i, config["num_processes"], config, year, sub_month_generator),
        )
        processes.append(p)
        p.start()
    # all process complete successfully
    for i in range(config["num_processes"]):
        processes[i].join()

    # combine sub-seqrecord
    record = WSeqRecord.gather_subseqrecords(
        config["wseqrecord_dir"], world_size=config["num_processes"]
    )

    record.dump_record()


if __name__ == "__main__":
    main()
