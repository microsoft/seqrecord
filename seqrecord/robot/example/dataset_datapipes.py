from seqrecorder.seqrecord import SeqRecord
from seqrecorder.datapipes import VideoDatapipeFromSeqRecord

import numpy as np
from typing import Callable, Dict, List, Optional
import torch.utils.data.datapipes as dp
import torch


def list2array(data_list: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    """transform data from list of np.array to a single numpy array.
    Args:
        data_np (Dict[str, List[np.ndarray]]): _description_
    Returns:
        Dict[str, np.ndarray]: _description_
    """
    data_array: Dict[str, np.ndarray] = {}
    for feature in data_list:
        data_array[feature] = np.stack(data_list[feature], axis=0)
    return data_array


def collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    collated_batch: Dict[str, torch.Tensor] = {}
    for feature in batch[0]:
        collated_batch[feature] = torch.from_numpy(
            np.stack([batch[i][feature] for i in range(len(batch))], axis=0)
        )
    return collated_batch


def build_iter_datapipe(
    recorddir, segment_len, features, transform: Optional[Callable]
):
    record = SeqRecord.load_record_from_dict(recorddir)
    datapipe = VideoDatapipeFromSeqRecord(record, segment_len, features)
    # Shuffle will happen as long as you do NOT set `shuffle=False` later in the DataLoader
    # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
    datapipe = dp.iter.Shuffler(datapipe, buffer_size=1000)
    # sharding: Place ShardingFilter (datapipe.sharding_filter) as early as possible in the pipeline,
    # especially before expensive operations such as decoding, in order to avoid repeating these expensive operations across worker/distributed processes.
    datapipe = dp.iter.ShardingFilter(datapipe)
    datapipe = dp.iter.Mapper(datapipe, fn=list2array)
    if transform is not None:
        datapipe = dp.iter.Mapper(datapipe, fn=transform)
    # Note that if you choose to use Batcher while setting batch_size > 1 for DataLoader,
    # your samples will be batched more than once. You should choose one or the other.
    # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
    datapipe = dp.iter.Batcher(datapipe, batch_size=2, drop_last=True)
    datapipe = dp.iter.Collator(datapipe, collate_fn=collate_fn)
    return datapipe


if __name__ == "__main__":

    recorddir = "./output/recorddataset/"
    segment_len = 3
    features = ["image_left", "image_right"]

    datapipe = build_iter_datapipe(recorddir, segment_len, features, None)

    for seq in datapipe:
        print(seq.keys())
