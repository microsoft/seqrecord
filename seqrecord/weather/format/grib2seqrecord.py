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


def frame_generator(path, year):
    def get_num_frames_in_month(m: int) -> int:
        """Read a single variable of one-month data and return the number of frames in that month

        Assume:
            within a month, every variable has the same number of frames

        Args:
            m (int): _description_

        Returns:
            int: _description_
        """
        var = single_vars[0]
        var_path = os.path.join(path, f"{year}", f"{m:02d}", f"{var}_0.grib")
        ds = xr.open_dataset(var_path)

        return ds[NAME_TO_VAR[var]].shape[0]

    for m in tqdm(range(1, 13)):
        num_frames_in_month = get_num_frames_in_month(m)
        np_vars = {}
        for var in single_vars:
            var_path = os.path.join(path, f"{year}", f"{m:02d}", f"{var}_0.grib")
            ds = xr.open_dataset(var_path)
            code = NAME_TO_VAR[var]
            assert len(ds[code].shape) == 3

            np_vars[var] = ds[code].to_numpy()

            del ds
            # assert np_vars[var].shape[0] == HOURS_PER_MONTH

        for var in pressure_vars:
            for level in LEVELS:
                var_path = os.path.join(
                    path,
                    f"{year}",
                    f"{m:02d}",
                    f"{var}_{level}.grib",
                )
                ds = xr.open_dataset(var_path)
                code = NAME_TO_VAR[var]
                np_vars[f"{var}_{level}"] = ds[code].to_numpy()
                del ds
                # assert np_vars[f"{var}_{level}"].shape[0] == HOURS_PER_MONTH

        for i in range(num_frames_in_month):
            frame = {}
            for key in np_vars:
                frame[key] = np_vars[key][i]
            yield frame


def grib2np(record, config, year):
    """
    Convert grib files to numpy arrays and save them to disk.
    """
    record.put_frame(frame_generator(config["grib_dataset_dir"], year))


@click.command()
@click.option(
    "--dataset-mount-dir", type=str, default="/datadrive/azure_storage/weathereastus"
)
@click.option("--year", type=int, required=True)
def main(dataset_mount_dir: str, year: int):
    # dataset: 5.625deg_equally_np_all_levels,  1.40625deg_equally_np_all_levels(ultimate goal)
    # "/datadrive/weatherdatastorage2/datasets",  # "/mnt/data",
    DEFAULT_CONFIG = {
        "wseqrecord_dir": os.path.join(
            dataset_mount_dir,
            f"era5seqrecord/aml/{year}",
        ),
        "grib_dataset_dir": os.path.join(dataset_mount_dir, "era5"),
    }

    config = DEFAULT_CONFIG

    record = WSeqRecord(config["wseqrecord_dir"])
    grib2np(record, config, year)
    print("Dumping")
    record.dump_record()


if __name__ == "__main__":
    main()
