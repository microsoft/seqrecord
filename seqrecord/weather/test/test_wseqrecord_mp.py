"""test parallel write, gather meta information, and read frame pairs"""

import unittest
from typing import Dict, List
import random
import os
import numpy as np
import numpy.testing as nptest
import seqrecord.weather.seqrecord as wseqrecord
from seqrecord.weather.seqrecord import WSeqRecord
from seqrecord.weather.format.weathernp2seq_mp import distribute_loads
from itertools import islice
from multiprocessing import Process


class TestWSeqRecord(unittest.TestCase):
    def test_read_frame(self):

        print("Test read frame")
        dataset, rootdir, features = build_weather_dataset()

        record = parallel_write(dataset, rootdir, num_processes=3)
        for i, item in enumerate(record.iterate_frames(features=features)):
            for feature in features:
                nptest.assert_equal(
                    item[feature], dataset[i][feature], err_msg="", verbose=True
                )

    def test_read_frame_pair(self):

        print("Test read frame pairs")
        dataset, rootdir, features = build_weather_dataset()

        record = parallel_write(dataset, rootdir, num_processes=3)

        max_pred_steps = 100
        random_partition = random.randint(1, len(features) - 2)

        input_features = features[:random_partition]
        output_features = features[random_partition:]
        # read frame pairs
        for i, item in enumerate(
            record.iterate_framepairs(
                input_features, output_features, max_pred_steps=max_pred_steps
            )
        ):
            x, y, lookahead_steps = (
                item["input"],
                item["target"],
                item["lookahead_steps"],
            )
            x_target = np.vstack([dataset[i][feature] for feature in input_features])
            y_target = np.vstack(
                [dataset[i + lookahead_steps][feature] for feature in output_features]
            )
            nptest.assert_equal(x, x_target, err_msg="", verbose=True)
            nptest.assert_equal(
                y,
                y_target,
                err_msg="",
                verbose=True,
            )

    def test_fetch_frame_pair(self):

        print("Test fetch frame pairs")
        dataset, rootdir, features = build_weather_dataset()

        record = parallel_write(dataset, rootdir, num_processes=3)

        max_pred_steps = 100
        random_partition = random.randint(1, len(features) - 2)

        input_features = features[:random_partition]
        output_features = features[random_partition:]
        # read frame pairs
        for i, item in enumerate(
            record.fetch_framepairs(
                input_features, output_features, max_pred_steps=max_pred_steps
            )
        ):
            x, y, lookahead_steps = (
                item["input"],
                item["target"],
                item["lookahead_steps"],
            )
            x_target = np.vstack([dataset[i][feature] for feature in input_features])
            y_target = np.vstack(
                [dataset[i + lookahead_steps][feature] for feature in output_features]
            )
            nptest.assert_equal(x, x_target, err_msg="", verbose=True)
            nptest.assert_equal(
                y,
                y_target,
                err_msg="",
                verbose=True,
            )

    def test_async_iterate_frame_pair(self):

        print("Test async iterate frame pairs")
        dataset, rootdir, features = build_weather_dataset()

        record = parallel_write(dataset, rootdir, num_processes=3)

        max_pred_steps = 100
        random_partition = random.randint(1, len(features) - 2)

        input_features = features[:random_partition]
        output_features = features[random_partition:]
        # read frame pairs
        for i, item in enumerate(
            record.async_iterate_framepairs(
                input_features, output_features, max_pred_steps=max_pred_steps
            )
        ):
            x, y, lookahead_steps = (
                item["input"],
                item["target"],
                item["lookahead_steps"],
            )
            x_target = np.vstack([dataset[i][feature] for feature in input_features])
            y_target = np.vstack(
                [dataset[i + lookahead_steps][feature] for feature in output_features]
            )
            nptest.assert_equal(x, x_target, err_msg="", verbose=True)
            nptest.assert_equal(
                y,
                y_target,
                err_msg="",
                verbose=True,
            )

    def test_iterate_frame_pair_from_files(self):

        print("Test iterate frame pairs from files")
        dataset, rootdir, features = build_weather_dataset()

        record = parallel_write(dataset, rootdir, num_processes=3)

        max_pred_steps = 100
        random_partition = random.randint(1, len(features) - 2)

        input_features = features[:random_partition]
        output_features = features[random_partition:]
        read_args = {
            "input_features": features[:random_partition],
            "target_features": features[random_partition:],
            "max_pred_steps": max_pred_steps,
        }
        record.add_read_args(read_args)
        fileidx_generator = ((record, i) for i in range(record.num_files))
        # read frame pairs
        for i, item in enumerate(
            record.iterate_framepairs_from_files(fileidx_generator)
        ):
            x, y, lookahead_steps = (
                item["input"],
                item["target"],
                item["lookahead_steps"],
            )
            x_target = np.vstack([dataset[i][feature] for feature in input_features])
            y_target = np.vstack(
                [dataset[i + lookahead_steps][feature] for feature in output_features]
            )
            nptest.assert_equal(x, x_target, err_msg="", verbose=True)
            nptest.assert_equal(
                y,
                y_target,
                err_msg="",
                verbose=True,
            )

    def test_local_cache_writer(self):
        print("Test iterate frame pairs from files")

        wseqrecord.MAX_RECORDFILE_SIZE = 1e8
        dataset, rootdir, features = build_weather_dataset()

        record = parallel_write(dataset, rootdir, num_processes=3)


def build_weather_dataset():
    """Generate an aritificial weather dataset so that each feature has the same size"""
    rootdir = "./output/wseqrecord_test/"
    features = {"s1": None, "i5": None, "i7": None, "v100": None, "A100": None}
    time_horizon = 10000
    dataset = [{} for _ in range(time_horizon)]
    shape = (np.random.randint(1, 100), np.random.randint(1, 100))
    dtype = np.random.choice(["float32", "int", "bool"])
    for feature in features:
        for i in range(time_horizon):
            dataset[i][feature] = (
                np.random.rand(shape[0], shape[1])
                if dtype == "float32"
                else np.random.randint(low=0, high=255, size=shape)
            )
    features = [feature for feature in features]
    return dataset, rootdir, features


def write_frame(rank, world_size, rootdir, frame_gen, load_amound):
    sub_wsrecord = WSeqRecord(
        os.path.join(rootdir, WSeqRecord.subfolder_name(rank, world_size)),
        local_cache_dir=os.path.join("./output/cache/", str(rank)),
    )
    sub_wsrecord.put_frame(frame_gen)
    sub_wsrecord.dump_record(rank)


def parallel_write(dataset, rootdir, num_processes):

    dividens = distribute_loads(len(dataset), num_processes)
    processes = []
    for i in range(num_processes):
        sub_dataset_generator = islice(iter(dataset), dividens[i][0], dividens[i][1])
        p = Process(
            target=write_frame,
            args=(
                i,
                num_processes,
                rootdir,
                sub_dataset_generator,
                dividens[i][1] - dividens[i][0],
            ),
        )
        processes.append(p)
        p.start()
    # all process complete successfully
    for i in range(num_processes):
        processes[i].join()

    # combine sub-seqrecord
    record = WSeqRecord.gather_subseqrecords(rootdir, world_size=num_processes)
    record.dump_record()
    return record


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
