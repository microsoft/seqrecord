"""Create a dummy dataset and transform it into SeqRecord format.
"""

import numpy as np
from tqdm import tqdm
from dataio.seqrecord import SeqRecord


def encode_dummy_dataset(
    recorddir: str,
) -> None:
    """create and transform an artificial dataset to SeqRecord format.

    Args:
        recorddir (str): directory where the seqrecord files will be saved
    """
    # attributes of dataset
    num_seq = 10
    seq_len = 7
    features = {"image_left": "RGBImage", "image_right": "RGBImage"}
    record = SeqRecord(
        recorddir=recorddir,
        features=features,
        pretransform_module_path="dataio.example.dataset_transform",
    )

    for _ in tqdm(range(num_seq)):
        for j in range(seq_len):
            item = {
                "image_left": np.random.rand(224, 224, 3),
                "image_right": np.random.rand(224, 224, 3),
            }
            record.write_item(item, (j == 0))
    record.close_recordfile()
    record.dump()


if __name__ == "__main__":
    recorddir = "./output/recorddataset/"
    encode_dummy_dataset(recorddir)
