import os
import sys
import numpy as np
import xarray as xr
import psutil
from seqrecord.weather.seqrecord import WSeqRecord
from seqrecord.utils import distribute_loads
import os
from typing import List, Tuple, Iterator
import re
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
import click
from itertools import islice
from seqrecord.weather.format.constants import NAME_TO_VAR

single_vars = [
    "2m_temperature",
    # "10m_u_component_of_wind",
    # "10m_v_component_of_wind",
    # "mean_sea_level_pressure",
    # # "surface_pressure",
    # # "2m_dewpoint_temperature",
    # # "total_precipitation",
    # "total_cloud_cover",
    # "total_column_water_vapour",
]


HOURS_PER_MONTH = 744
HOURS_PER_SHARD = 384

FRAME_BUFFER_SIZE = 50


def frame_generator(rank: int, path, year, month_gen: Iterator):
    def get_num_frames_in_month(m):
        var = single_vars[0]
        var_path = os.path.join(path, f"{year}", f"{m:02d}", f"{var}_0.grib")
        ds = xr.open_dataset(var_path)

        return ds[NAME_TO_VAR[var]].shape[0]

    for m in month_gen:
        num_frames_in_month = get_num_frames_in_month(m)
        print(f"{num_frames_in_month=}")
        for frame_idx in range(0, num_frames_in_month, FRAME_BUFFER_SIZE):
            np_vars = {}
            end_frame_idx = min(frame_idx + FRAME_BUFFER_SIZE, num_frames_in_month)
            for var in single_vars:
                var_path = os.path.join(path, f"{year}", f"{m:02d}", f"{var}_0.grib")
                ds = xr.open_dataset(var_path)
                code = NAME_TO_VAR[var]
                assert len(ds[code].shape) == 3

                np_vars[var] = ds[code][frame_idx:end_frame_idx].to_numpy()

                del ds
                # assert np_vars[var].shape[0] == HOURS_PER_MONTH
            print(f"/n RAM usage (GB): {psutil.virtual_memory()[3]/1000000000}")
            print(
                f"memory of the np array: {sum(arr.nbytes for _, arr in np_vars.items())}"
            )
            for frame_idx in range(0, end_frame_idx - frame_idx):
                frame = {}
                for key in np_vars:
                    frame[key] = np_vars[key][frame_idx]
                print(
                    f"memory of the frame np array: {sum(arr.nbytes for _, arr in frame.items())} /n"
                )
                yield frame
                break


def grib2np(rank, world_size, config, year, month_gen):
    """
    Convert grib files to numpy arrays and save them to disk.
    """
    sub_wseqrecord = WSeqRecord(
        os.path.join(
            config["wseqrecord_dir"],
            WSeqRecord.subfolder_name(rank, world_size),
        )
    )

    sub_wseqrecord.put_frame(
        frame_generator(rank, config["grib_dataset_dir"], year, month_gen)
    )
    sub_wseqrecord.dump_record(rank=rank)


# notes: # dataset: 5.625deg_equally_np_all_levels,  1.40625deg_equally_np_all_levels(ultimate goal)
@click.command()
@click.option(
    "--dataset-mount-dir", type=str, default="/datadrive/azure_storage/weathereastus"
)
@click.option("--year", type=int, required=True)
@click.option("--num-processes", type=int, default=6)
def main(dataset_mount_dir: str, year: int, num_processes: int):
    print(f"configs {dataset_mount_dir=}, {year=}, {num_processes=}.\n")
    year = 1979
    # "/datadrive/weatherdatastorage2/datasets",  # "/mnt/data",
    DEFAULT_CONFIG = {
        "num_processes": num_processes,
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
    print("Combining meta-seqrecord")
    record = WSeqRecord.gather_subseqrecords(
        config["wseqrecord_dir"], world_size=config["num_processes"]
    )

    record.dump()


if __name__ == "__main__":
    main()
