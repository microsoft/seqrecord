"""Unit test for Wseqrecord's correctness and speed"""
import unittest
from typing import Dict, List
import random
import numpy as np
import numpy.testing as nptest
import seqrecord.weather.seqrecord as wseqrecord
from seqrecord.weather.seqrecord import WSeqRecord


def test_iter_frames(file_size: int):
    dataset, record, features = build_dataset()

    wseqrecord.MAX_RECORDFILE_SIZE = file_size
    # encode dataset
    record.put_frame(frame_generator=iter(dataset))
    record.dump_record()

    # loaded_record = WSeqRecord.load_record_from_dict("./output/wseqrecord_test/")
    # decode dataset
    for i, item in enumerate(record.iterate_frames(features=features)):
        for feature in features:
            nptest.assert_equal(
                item[feature], dataset[i][feature], err_msg="", verbose=True
            )


def test_iter_frame_pairs(file_size: int, max_pred_steps: int):
    assert max_pred_steps > 1, "maximum prediction steps need to be greater than 1"
    dataset, record, features = build_weather_dataset()
    lookahead_steps_stats = [0 for _ in range(max_pred_steps)]
    wseqrecord.MAX_RECORDFILE_SIZE = file_size
    # encode dataset
    record.put_frame(frame_generator=iter(dataset))
    record.dump_record()

    loaded_record = WSeqRecord.load_record("./output/wseqrecord_test/")
    random_partition = random.randint(1, len(features) - 2)
    input_features = features[:random_partition]
    output_features = features[random_partition:]
    # read frame pairs
    for i, item in enumerate(
        loaded_record.iterate_framepairs(
            input_features, output_features, max_pred_steps=max_pred_steps
        )
    ):
        x, y, lookahead_steps = (
            item["input"],
            item["target"],
            item["lookahead_steps"],
        )
        lookahead_steps_stats[lookahead_steps - 1] += 1
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
    return [d / sum(lookahead_steps_stats) for d in lookahead_steps_stats]


def test_fetch_frame_pairs(file_size: int, max_pred_steps: int):
    assert max_pred_steps > 1, "maximum prediction steps need to be greater than 1"
    dataset, record, features = build_weather_dataset()
    lookahead_steps_stats = [0 for _ in range(max_pred_steps)]
    wseqrecord.MAX_RECORDFILE_SIZE = file_size
    # encode dataset

    record.put_frame(frame_generator=iter(dataset))
    record.dump_record()

    loaded_record = WSeqRecord.load_record("./output/wseqrecord_test/")
    random_partition = random.randint(1, len(features) - 2)
    input_features = features[:random_partition]
    output_features = features[random_partition:]
    # read frame pairs
    for i, item in enumerate(
        loaded_record.fetch_framepairs(
            input_features,
            output_features,
            max_pred_steps=max_pred_steps,
            prefetch_buffer_size=10,
        )
    ):
        x, y, lookahead_steps = (
            item["input"],
            item["target"],
            item["lookahead_steps"],
        )
        lookahead_steps_stats[lookahead_steps - 1] += 1
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
    return [d / sum(lookahead_steps_stats) for d in lookahead_steps_stats]


def test_async_read_frame_pairs(file_size: int, max_pred_steps: int):
    assert max_pred_steps > 1, "maximum prediction steps need to be greater than 1"
    dataset, record, features = build_weather_dataset()
    lookahead_steps_stats = [0 for _ in range(max_pred_steps)]
    wseqrecord.MAX_RECORDFILE_SIZE = file_size
    # encode dataset
    record.put_frame(frame_generator=iter(dataset))
    record.dump_record()

    loaded_record = WSeqRecord.load_record("./output/wseqrecord_test/")
    random_partition = random.randint(1, len(features) - 2)
    input_features = features[:random_partition]
    output_features = features[random_partition:]
    # read frame pairs
    for i, item in enumerate(
        loaded_record.async_iterate_framepairs(
            input_features, output_features, max_pred_steps=max_pred_steps
        )
    ):
        x, y, lookahead_steps = (
            item["input"],
            item["target"],
            item["lookahead_steps"],
        )
        lookahead_steps_stats[lookahead_steps - 1] += 1
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
    return [d / sum(lookahead_steps_stats) for d in lookahead_steps_stats]


def test_threaded_write_frame(file_size: int, max_pred_steps: int):
    assert max_pred_steps > 1, "maximum prediction steps need to be greater than 1"
    dataset, record, features = build_weather_dataset()
    lookahead_steps_stats = [0 for _ in range(max_pred_steps)]
    wseqrecord.MAX_RECORDFILE_SIZE = file_size
    # encode dataset
    record.put_frame(iter(dataset), 5)
    record.dump_record()

    # loaded_record = WSeqRecord.load_record_from_dict("./output/wseqrecord_test/")
    # decode dataset
    for i, item in enumerate(record.iterate_frames(features=features)):
        for feature in features:
            nptest.assert_equal(
                item[feature], dataset[i][feature], err_msg="", verbose=True
            )


def test_read_framepairs_from_file(file_size: int, max_pred_steps: int):
    assert max_pred_steps > 1, "maximum prediction steps need to be greater than 1"
    dataset, record, features = build_weather_dataset()
    lookahead_steps_stats = [0 for _ in range(max_pred_steps)]
    wseqrecord.MAX_RECORDFILE_SIZE = file_size
    # encode dataset

    record.put_frame(iter(dataset), 5)
    record.dump_record()

    loaded_record = WSeqRecord.load_record("./output/wseqrecord_test/")
    random_partition = random.randint(1, len(features) - 2)
    read_args = {
        "input_features": features[:random_partition],
        "target_features": features[random_partition:],
        "max_pred_steps": max_pred_steps,
    }
    loaded_record.add_read_args(read_args)
    # read frame pairs
    for i, item in enumerate(
        WSeqRecord.iterate_framepairs_from_files(
            ((loaded_record, i) for i in range(loaded_record.num_files))
        )
    ):
        x, y, lookahead_steps = (
            item["input"],
            item["target"],
            item["lookahead_steps"],
        )
        lookahead_steps_stats[lookahead_steps - 1] += 1
        x_target = np.vstack(
            [dataset[i][feature] for feature in read_args["input_features"]]
        )
        y_target = np.vstack(
            [
                dataset[i + lookahead_steps][feature]
                for feature in read_args["target_features"]
            ]
        )
        nptest.assert_equal(x, x_target, err_msg="", verbose=True)
        nptest.assert_equal(
            y,
            y_target,
            err_msg="",
            verbose=True,
        )
    return [d / sum(lookahead_steps_stats) for d in lookahead_steps_stats]


def test_read_frames_from_file(
    num_frames_in_input: int,
    max_pred_steps: int,
    num_frames: int,
    file_size: int,
):
    assert max_pred_steps > 1, "maximum prediction steps need to be greater than 1"
    dataset, record, features = build_weather_dataset(num_frames)
    lookahead_steps_stats = [0 for _ in range(max_pred_steps)]
    wseqrecord.MAX_RECORDFILE_SIZE = file_size
    # encode dataset

    record.put_frame(iter(dataset), 5)
    record.dump_record()

    loaded_record = WSeqRecord.load_record("./output/wseqrecord_test/")

    read_args = {
        "input_features": features,
        "target_features": features,
        "max_pred_steps": max_pred_steps,
        "hours_per_step": 1,
        "num_frames_in_input": num_frames_in_input,
    }
    loaded_record.set_framereader_args(read_args)
    # read frame pairs
    num_input_target_pairs = 0
    for i, item in enumerate(
        WSeqRecord.iterate_frames_from_file(
            ((loaded_record, i) for i in range(loaded_record.num_files))
        )
    ):
        print(f"testing the {i}-th input-target pair")
        num_input_target_pairs += 1
        x, y, lookahead_steps = (
            item["input"],
            item["target"],
            item["lead_time"],
        )
        x_target = [
            np.stack(
                [dataset[i + j][feature] for feature in read_args["input_features"]],
                axis=0,
            )
            for j in range(num_frames_in_input)
        ]

        for j in range(num_frames_in_input):
            frame_idx, frame = x[j]
            print(
                f"inside {i}-th input_frames, testing the {j}-th (rel) and {frame_idx}-th (abs) frame"
            )
            nptest.assert_equal(
                frame,
                x_target[j],
                err_msg="",
                verbose=True,
            )
        print(
            f"with {num_frames_in_input=} and {num_frames}, {lookahead_steps=} and target frame idx is {i + num_frames_in_input+ lookahead_steps-1}"
        )
        y_target = np.stack(
            [
                dataset[i + num_frames_in_input + lookahead_steps - 1][feature]
                for feature in read_args["target_features"]
            ]
        )
        nptest.assert_equal(
            y,
            y_target,
            err_msg="",
            verbose=True,
        )
    print("num_input_target_pairs: ", num_input_target_pairs)
    return num_input_target_pairs


class TestWSeqRecord(unittest.TestCase):
    def test_encode_decode(self):
        """Testing encode and decode of frames."""
        print("testing reading frame")
        test_iter_frames(1e4)
        test_iter_frames(1e6)
        test_iter_frames(1e8)
        test_iter_frames(1e10)

    def test_read_frame_pairs(self):
        """Testing iteratively read frame pairs"""
        # todo: fix feature bug and max_steps bugs
        print("testing reading frame pairs")
        lookahead_stats = test_iter_frame_pairs(1e6, max_pred_steps=10)
        lookahead_stats = test_iter_frame_pairs(1e8, max_pred_steps=5)
        lookahead_stats = test_iter_frame_pairs(1e10, max_pred_steps=13)

        lookahead_stats = test_iter_frame_pairs(1e6, max_pred_steps=100)
        lookahead_stats = test_iter_frame_pairs(1e8, max_pred_steps=1000)
        lookahead_stats = test_iter_frame_pairs(1e10, max_pred_steps=10000)

    def test_fetch_frame_pairs(self):
        """Tesing reading frame pairs with prefetching"""
        print("testing fetching frame pairs")
        test_fetch_frame_pairs(1e6, max_pred_steps=10)
        test_fetch_frame_pairs(1e8, max_pred_steps=5)
        test_fetch_frame_pairs(1e10, max_pred_steps=13)

        test_fetch_frame_pairs(1e6, max_pred_steps=100)
        test_fetch_frame_pairs(1e8, max_pred_steps=1000)
        test_fetch_frame_pairs(1e10, max_pred_steps=10000)

    def test_async_read_frame_pairs(self):
        """Testing reading frame pairs using asyncio"""

        print("testing async read frame pairs")
        test_async_read_frame_pairs(1e6, max_pred_steps=10)
        test_async_read_frame_pairs(1e8, max_pred_steps=5)
        test_async_read_frame_pairs(1e10, max_pred_steps=13)
        test_async_read_frame_pairs(1e6, max_pred_steps=100)
        test_async_read_frame_pairs(1e8, max_pred_steps=1000)
        test_async_read_frame_pairs(1e10, max_pred_steps=10000)

    def test_single_async(self):

        print("Single test")
        test_async_read_frame_pairs(1e8, max_pred_steps=1000)

    def test_threaded_write(self):

        test_threaded_write_frame(1e6, max_pred_steps=100)

    def test_read_framepairs_from_file(self):
        print("Testing reading frame files")

        test_read_framepairs_from_file(1e6, max_pred_steps=100)

    def test_read_frames_from_file(self):
        print("Testing reading frames from file")
        num_frames_in_input = 5
        max_pred_steps = 10
        num_frames = 15
        num_inputtarget_pairs = test_read_frames_from_file(
            num_frames_in_input, max_pred_steps, num_frames, 1e6
        )
        self.assertEqual(num_inputtarget_pairs, num_frames - num_frames_in_input)

        # todo: test with inconsecutive frames


def build_dataset():
    """Generate an aritificial dataset to test methods of SeqRecord, w"""
    rootdir = "./output/wseqrecord_test/"
    features = {"s1": None, "i5": None, "i7": None, "v100": None, "A100": None}
    time_horizon = 1000
    dataset = [{} for _ in range(time_horizon)]
    for feature in features:
        shape = (np.random.randint(1, 100), np.random.randint(1, 100))
        dtype = np.random.choice(["float32", "int", "bool"])
        for i in range(time_horizon):
            dataset[i][feature] = (
                np.random.rand(shape[0], shape[1])
                if dtype == "float32"
                else np.random.randint(low=0, high=255, size=shape)
            )
    record = WSeqRecord(rootdir)
    features = [feature for feature in features]
    return dataset, record, features


def build_weather_dataset(time_horizon: int = 1000):
    """Generate an aritificial weather dataset so that each feature has the same size"""
    rootdir = "./output/wseqrecord_test/"
    features = {"s1": None, "i5": None, "i7": None, "v100": None, "A100": None}
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
    record = WSeqRecord(rootdir)
    features = [feature for feature in features]
    return dataset, record, features


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
