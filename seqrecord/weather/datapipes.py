"""Iterative datapipes toread weather dataset in seqrecord format"""

from typing import Callable, Dict, Iterator, List, Optional, Tuple
from time import perf_counter
import numpy as np
import torch
import torchdata.datapipes as datapipe
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
from tqdm import tqdm

from .seqrecord import WSeqRecord

# todo: let collate_fn be a parameter of build_wdatapipe!


@datapipe.functional_datapipe("gen_framepair")
class FramePairsFromWSeqRecord(datapipe.iter.IterDataPipe):
    """A torch datapiple class that iteratively read frame pairs from weather dataset (encoded by WSeqRecord)."""

    def __init__(
        self,
        source_dp: datapipe.iter.IterDataPipe,
    ) -> None:
        super().__init__()
        self.source_dp = source_dp

    def __iter__(self):
        yield from WSeqRecord.iterate_framepairs_from_files(
            self.source_dp,
        )


@datapipe.functional_datapipe("gen_fileidx")
class FileidxFromWSeqRecord(datapipe.iter.IterDataPipe):
    """A torch datapiple class that iteratively read fileidx from weather dataset (encoded by WSeqRecord)."""

    def __init__(
        self,
        record_dp: datapipe.iter.IterDataPipe,
    ) -> None:
        super().__init__()
        self.record_dp = record_dp

    def __iter__(self):
        for record in self.record_dp:
            for fileidx in range(record.num_files):
                yield record, fileidx


def build_wdatapipe(
    records: List[WSeqRecord],
    file_shuffle_buffer_size: Optional[int],
    data_shuffle_buffer_size: Optional[int],
    batch_size: int,
    mappings: List[Callable],
    collate_fn: Callable,
) -> datapipe.iter.IterDataPipe:
    """Iteratively apply operations to datapipe: shuffle, sharding, map, batch, collator

    Args:
        datapipe (datapipe.datapipe.IterDataPipe): entry datapipe
        shuffle_buffer_size (Optional[int]): buffer size for pseudo-shuffle
        batch_size (int):
        mappings (List[Callable]): a list of transforms applied to datapipe, between sharding and batch

    Returns:
        datapipe.datapipe.IterDataPipe: transformed datapipe ready to be sent to dataloader
    """
    dp = IterableWrapper(records).gen_fileidx()
    # Shuffle will happen as long as you do NOT set `shuffle=False` later in the DataLoader
    # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
    if file_shuffle_buffer_size is not None:
        dp = dp.shuffle(buffer_size=file_shuffle_buffer_size)
    # sharding: Place ShardingFilter (datapipe.sharding_filter) as early as possible in the pipeline,
    # especially before expensive operations such as decoding, in order to avoid repeating these expensive operations across worker/distributed processes.
    # output will be a sequence of file_idx(s) distributed to different workers (
    dp = dp.sharding_filter()
    dp = dp.gen_framepair()
    # dp = dp.shuffle(buffer_size=data_shuffle_buffer_size)
    for i, mapping in enumerate(mappings):
        dp = dp.map(fn=mapping)
    # Note that if you choose to use Batcher while setting batch_size > 1 for DataLoader,
    # your samples will be batched more than once. You should choose one or the other.
    # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
    dp = dp.batch(batch_size=batch_size, drop_last=True)
    dp = dp.collate(collate_fn=collate_fn)
    return dp
