import os
from torchdata.dataloader2 import (
    DataLoader2,
    MultiProcessingReadingService,
    DistributedReadingService,
    SequentialReadingService,
)
import torchdata
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
import torch.distributed as dist
import torch.multiprocessing as mp
import socket
import torch

from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES

MASTER_ADDR = "127.0.0.1"
WORLD_SIZE = 2


@functional_datapipe("square_dp")
class SqaureDP(IterDataPipe):
    def __init__(self, source_datapipe) -> None:
        self.source_datapipe = source_datapipe
        self.acc = []

    def __iter__(self):
        for num in self.source_datapipe:
            print(f"getting number {num} and square it to get {num*num}")
            self.acc.append(num)
            yield num * num
        print(f"\n {self.acc} \n")


def _get_open_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)


os.environ["MASTER_ADDR"] = MASTER_ADDR
os.environ["MASTER_PORT"] = _get_open_port()


def dist_dl(rank, world_size):
    # using the old apis of torch data to shard
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dp = IterableWrapper([i for i in range(10)]).sharding_filter().square_dp()
    mp_rs = MultiProcessingReadingService(num_workers=3)
    dist_rs = DistributedReadingService()
    rs = SequentialReadingService(dist_rs, mp_rs)
    dl = DataLoader2(dp, reading_service=rs)
    for d in dl:
        d
    dl.shutdown()


def mp_dl():
    dp = (
        IterableWrapper([i for i in range(10)])
        .sharding_filter(sharding_group_filter=SHARDING_PRIORITIES.DEFAULT)
        .sharding_filter(sharding_group_filter=SHARDING_PRIORITIES.MULTIPROCESSING)
        .square_dp()
    )
    mp_rs = MultiProcessingReadingService(num_workers=3)
    dist_rs = DistributedReadingService()
    rs = SequentialReadingService(dist_rs, mp_rs)
    dl = DataLoader2(dp, reading_service=rs)
    for i, x in enumerate(dl):
        # print(x)
        x
    dl.shutdown()


if __name__ == "__main__":

    ctx = mp.get_context("fork")  # Notebook doesn't work well with spawn

for rank in range(WORLD_SIZE):
    p = ctx.Process(target=dist_dl, args=(rank, WORLD_SIZE))
    p.start()
