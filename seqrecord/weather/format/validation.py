from seqrecord import WSeqRecord, build_wdatapipe
import torch
import numpy as np
from tqdm import tqdm
from torchdata.dataloader2 import (
    DataLoader2,
    DistributedReadingService,
    MultiProcessingReadingService,
    SequentialReadingService,
)
from time import perf_counter
import click
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys


MASTER_ADDR = "127.0.0.1"
WORLD_SIZE = 2
import socket


def _get_open_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)


os.environ["MASTER_ADDR"] = MASTER_ADDR
os.environ["MASTER_PORT"] = _get_open_port()


# constants
VAR = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_cloud_cover",
    "total_column_water_vapour",
    "geopotential_1",
    "geopotential_2",
    "geopotential_3",
    "geopotential_5",
    "geopotential_7",
    "geopotential_10",
    "geopotential_20",
    "geopotential_30",
    "geopotential_70",
    "geopotential_125",
    "geopotential_175",
    "geopotential_225",
    "geopotential_350",
    "geopotential_450",
    "geopotential_550",
    "geopotential_650",
    "geopotential_750",
    "geopotential_775",
    "geopotential_800",
    "geopotential_825",
    "geopotential_875",
    "geopotential_900",
    "geopotential_950",
    "geopotential_975",
    "specific_humidity_1",
    "specific_humidity_2",
    "specific_humidity_3",
    "specific_humidity_5",
    "specific_humidity_7",
    "specific_humidity_10",
    "specific_humidity_20",
    "specific_humidity_30",
    "specific_humidity_70",
    "specific_humidity_125",
    "specific_humidity_175",
    "specific_humidity_225",
    "specific_humidity_350",
    "specific_humidity_450",
    "specific_humidity_550",
    "specific_humidity_650",
    "specific_humidity_750",
    "specific_humidity_775",
    "specific_humidity_800",
    "specific_humidity_825",
    "specific_humidity_875",
    "specific_humidity_900",
    "specific_humidity_950",
    "specific_humidity_975",
    "temperature_1",
    "temperature_2",
    "temperature_3",
    "temperature_5",
    "temperature_7",
    "temperature_10",
    "temperature_20",
    "temperature_30",
    "temperature_70",
    "temperature_125",
    "temperature_175",
    "temperature_225",
    "temperature_350",
    "temperature_450",
    "temperature_550",
    "temperature_650",
    "temperature_750",
    "temperature_775",
    "temperature_800",
    "temperature_825",
    "temperature_875",
    "temperature_900",
    "temperature_950",
    "temperature_975",
    "u_component_of_wind_1",
    "u_component_of_wind_2",
    "u_component_of_wind_3",
    "u_component_of_wind_5",
    "u_component_of_wind_7",
    "u_component_of_wind_10",
    "u_component_of_wind_20",
    "u_component_of_wind_30",
    "u_component_of_wind_70",
    "u_component_of_wind_125",
    "u_component_of_wind_175",
    "u_component_of_wind_225",
    "u_component_of_wind_350",
    "u_component_of_wind_450",
    "u_component_of_wind_550",
    "u_component_of_wind_650",
    "u_component_of_wind_750",
    "u_component_of_wind_775",
    "u_component_of_wind_800",
    "u_component_of_wind_825",
    "u_component_of_wind_875",
    "u_component_of_wind_900",
    "u_component_of_wind_950",
    "u_component_of_wind_975",
    "v_component_of_wind_1",
    "v_component_of_wind_2",
    "v_component_of_wind_3",
    "v_component_of_wind_5",
    "v_component_of_wind_7",
    "v_component_of_wind_10",
    "v_component_of_wind_20",
    "v_component_of_wind_30",
    "v_component_of_wind_70",
    "v_component_of_wind_125",
    "v_component_of_wind_175",
    "v_component_of_wind_225",
    "v_component_of_wind_350",
    "v_component_of_wind_450",
    "v_component_of_wind_550",
    "v_component_of_wind_650",
    "v_component_of_wind_750",
    "v_component_of_wind_775",
    "v_component_of_wind_800",
    "v_component_of_wind_825",
    "v_component_of_wind_875",
    "v_component_of_wind_900",
    "v_component_of_wind_950",
    "v_component_of_wind_975",
]
VAR4TEST = [
    "2m_temperature",
    "geopotential_1",
]


def identity(x):
    return x


def mp_loader(dp):
    rs = MultiProcessingReadingService(num_workers=4)
    dl = DataLoader2(dp, reading_service=rs)
    print("MP reading serivce")
    num_frames = 0
    for i, batch in tqdm(enumerate(dl)):
        # batch_size is 1
        num_frames += 1
        item = batch[0]
        print("\n")
        for key, val in item.items():
            if isinstance(val, dict):
                print(key, val.keys())
            elif isinstance(val, torch.Tensor):
                print(key, val.size())
            elif isinstance(val, np.ndarray):
                print(key, val.shape)
            else:
                print(key, val)
    print(f"num_frames: {num_frames}")
    dl.shutdown()


def dist_loader(rank, world_size, dp, q):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    rs = DistributedReadingService()
    dl = DataLoader2(dp, reading_service=rs)
    cnt = 0
    for d in tqdm(dl, desc=f"loading on rank {rank}"):
        cnt += 1
        # Mimic distributed training step
        dist.barrier()
    q.put(cnt)
    dl.shutdown()


def mp_dist_training(rank, world_size, dp, q):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    mp_rs = MultiProcessingReadingService(num_workers=3)
    dist_rs = DistributedReadingService()
    rs = SequentialReadingService(dist_rs, mp_rs)
    dl = DataLoader2(dp, reading_service=rs)
    cnt = 0
    for d in tqdm(dl, desc=f"loading on rank {rank} with mp reading service"):
        cnt += 1
        # Mimic distributed training step
        dist.barrier()
    q.put(cnt)
    dl.shutdown()


def dist_run(loader, dp):
    ctx = mp.get_context("fork")  # Notebook doesn't work well with spawn
    pqs = []
    for rank in range(WORLD_SIZE):
        q = ctx.Queue()
        p = ctx.Process(target=loader, args=(rank, WORLD_SIZE, dp, q))
        pqs.append((p, q))
        p.start()

    for rank in range(WORLD_SIZE):
        cnt = pqs[rank][1].get()
        print(f"DataLoader2 on rank {rank} received {cnt} data")
        pqs[rank][0].join()


@click.command()
@click.option(
    "--container-dir", default="/datadrive/azure_storage/weathereastus/era5seqrecord"
)
@click.option(
    "--reading-service", required=True, type=click.Choice(["mp", "dist", "mpdist"])
)
@click.option("--testing", required=True, type=bool)
def main(container_dir: str, reading_service: str, testing: bool = True):
    recorddir = f"{container_dir}/local/1980"
    record = WSeqRecord.load_record(recorddir=recorddir)
    var_list = VAR4TEST if testing else VAR
    record.set_framereader_args(
        {
            "input_features": var_list,
            "target_features": var_list,
            "hours_per_step": 1,
            "max_pred_steps": 10,
        }
    )

    dp = build_wdatapipe(
        [record], None, None, batch_size=1, mappings=[], collate_fn=identity
    )
    print(f"Testing: {testing}")
    if reading_service == "mp":
        print("Testing mp reading with 4 workers:")
        mp_loader(dp)
    elif reading_service == "dist":
        print(f"Testing dist reading with {WORLD_SIZE} nodes:")
        dist_run(dist_loader, dp)
    else:
        print(f"Testing dist mp reading with {WORLD_SIZE} nodes and 3 workers:")
        dist_run(mp_dist_training, dp)


if __name__ == "__main__":
    main()
