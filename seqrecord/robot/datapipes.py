"""Iterative datapipes built from seqrecord"""


from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from tqdm import tqdm

from .seqrecord import SeqRecord


@dp.functional_datapipe("video_datapipe")
class VideoDatapipeFromSeqRecord(dp.iter.IterDataPipe):
    """A torch datapiple class that iteratively read video(episode) segment from record files."""

    def __init__(
        self,
        record: SeqRecord,
        segment_len: int,
        features_rename: Dict[str, str],
        shuffle_recordfiles: bool = False,
    ) -> None:
        super().__init__()
        self.segmentproto = record.get_proto4segment(
            segment_len, [feature for feature in features_rename]
        )
        self.record = record
        self.features_rename = features_rename
        self.shuffle_recordfiles = shuffle_recordfiles

    def __iter__(self):
        for segment in self.record.read_segments(
            self.segmentproto, shuffle_recordfiles=self.shuffle_recordfiles
        ):
            res = {}
            for feature in self.features_rename:
                res[self.features_rename[feature]] = segment[feature]
            yield res


@dp.functional_datapipe("item_datapipe")
class ItemDatapipeFromSeqRecord(dp.iter.IterDataPipe):
    """A torch datapiple class that iteratively read item (frame) from record files."""

    def __init__(
        self,
        record: SeqRecord,
        features_rename: Dict[str, str],
        shuffle_recordfiles: bool = False,
    ) -> None:
        super().__init__()
        self.record = record
        self.features_rename = features_rename
        self.shuffle_recordfiles = shuffle_recordfiles

    def __iter__(self):
        res = {}
        for item in self.record.read_items(
            features=[feature for feature in self.features_rename],
            shuffle_recordfiles=self.shuffle_recordfiles,
        ):
            res = {}
            for feature in self.features_rename:
                res[self.features_rename[feature]] = item[feature]
            yield res


def collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    collated_batch: Dict[str, torch.Tensor] = {}
    for feature in batch[0]:
        collated_batch[feature] = torch.from_numpy(
            np.stack([batch[i][feature] for i in range(len(batch))], axis=0)
        )
    return collated_batch


def list2array(data_list: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    """transform data from list of np.array to a single numpy array. Only needed for video datapipes.
    Args:
        data_np (Dict[str, List[np.ndarray]]): _description_
    Returns:
        Dict[str, np.ndarray]: _description_
    """
    data_array: Dict[str, np.ndarray] = {}
    for feature in data_list:
        data_array[feature] = np.stack(data_list[feature], axis=0)
    return data_array


def build_datapipes(
    datapipe: dp.iter.IterDataPipe,
    shuffle_buffer_size: Optional[int],
    batch_size: int,
    mappings: List[Callable],
) -> dp.iter.IterDataPipe:
    """Iteratively apply operations to datapipe: shuffle, sharding, map, batch, collator

    Args:
        datapipe (dp.datapipe.IterDataPipe): entry datapipe
        shuffle_buffer_size (Optional[int]): buffer size for pseudo-shuffle
        batch_size (int):
        mappings (List[Callable]): a list of transforms applied to datapipe, between sharding and batch

    Returns:
        dp.datapipe.IterDataPipe: transformed datapipe ready to be sent to dataloader
    """
    # Shuffle will happen as long as you do NOT set `shuffle=False` later in the DataLoader
    # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
    if shuffle_buffer_size is not None:
        datapipe = datapipe.shuffle(buffer_size=shuffle_buffer_size)
    # sharding: Place ShardingFilter (datapipe.sharding_filter) as early as possible in the pipeline,
    # especially before expensive operations such as decoding, in order to avoid repeating these expensive operations across worker/distributed processes.
    datapipe = datapipe.sharding_filter()
    for i, mapping in enumerate(mappings):
        datapipe = datapipe.map(fn=mapping)
    # Note that if you choose to use Batcher while setting batch_size > 1 for DataLoader,
    # your samples will be batched more than once. You should choose one or the other.
    # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
    datapipe = datapipe.batch(batch_size=batch_size, drop_last=True)
    datapipe = datapipe.collate(collate_fn=collate_fn)
    return datapipe
