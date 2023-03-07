"""Unit test for seqrecord's functionalities"""
import unittest
from typing import Dict, List

import numpy as np
import numpy.testing as nptest
from seqrecord.robot.seqrecord import RSeqRecord


def concate_list(file2segment_item: Dict[str, list]):
    res = []
    for key in sorted(file2segment_item):
        res = res + file2segment_item[key]
    return res


class Test_RSeqRecord(unittest.TestCase):
    def test_encode_decode(self):
        """Testing encode and decode of items, no segment involved."""
        record, dataset, features = build_simple_dataset()
        # encode dataset
        for i, item in enumerate(dataset):
            if i % 4 == 0:
                # mock start of a sequence
                record.write_item(item, True)
            else:
                record.write_item(item, False)
        record.close_recordfile()
        record.dump()
        # decode dataset
        for i, item in enumerate(record.read_frames(features=features)):
            for feature in features:
                nptest.assert_equal(
                    item[feature], dataset[i][feature], err_msg="", verbose=True
                )
        loaded_record = RSeqRecord.load_record_from_dict("./output/seqrecord_test/")

    def test_idx4segment(self):
        """Having the record written (and various attributes setup), generate an index protocal for
        specific segment len."""
        record, dataset, features = build_simple_dataset()
        seq_len = 4
        # encode dataset
        for i, item in enumerate(dataset):
            if i % seq_len == 0:
                # mock start of a sequence
                record.write_item(item, True)
            else:
                record.write_item(item, False)
        record.close_recordfile()

        # segment len =2, sequence len =4, full features
        seg_len = 2
        idx4segment = record.get_metadata4segment(segment_len=seg_len)
        items = concate_list(idx4segment["file2segment_items"])
        self.assertEqual(len(items), 10)
        ids = [item_idx for _, item_idx in items]
        self.assertListEqual(ids, list(range(10)))
        for i, (is_segment_start, item) in enumerate(items):
            if i in [0, 1, 2, 4, 5, 6, 8]:
                self.assertTrue(is_segment_start)
            else:
                self.assertFalse(is_segment_start)
        heads = idx4segment["head4segment"]
        for i, segment in enumerate(record.read_segments(idx4segment)):
            for j in range(seg_len):
                for feature in features:
                    nptest.assert_equal(
                        dataset[heads[i] + j][feature],
                        segment[feature][j],
                        err_msg="",
                        verbose=True,
                    )

        # segment len =4, sequence len =4, full features
        seg_len = 4
        idx4segment = record.get_metadata4segment(segment_len=seg_len)
        items = concate_list(idx4segment["file2segment_items"])
        ids = [item_idx for _, item_idx in items]
        self.assertEqual(len(ids), 8)
        ids = [item_idx for _, item_idx in items]
        self.assertListEqual(ids, [0, 1, 2, 3, 4, 5, 6, 7])
        for i, (is_segment_start, item) in enumerate(items):
            if i in [0, 4]:
                self.assertTrue(is_segment_start)
            else:
                self.assertFalse(is_segment_start)
        heads = idx4segment["head4segment"]
        for i, segment in enumerate(record.read_segments(idx4segment)):
            for j in range(seg_len):
                for feature in features:
                    nptest.assert_equal(
                        dataset[heads[i] + j][feature],
                        segment[feature][j],
                        err_msg="",
                        verbose=True,
                    )

        # segment len =3, sequence len =4, full feature
        seg_len = 3
        idx4segment = record.get_metadata4segment(segment_len=seg_len)
        self.assertEqual(len(ids), 8)
        items = concate_list(idx4segment["file2segment_items"])
        ids = [item_idx for _, item_idx in items]
        self.assertListEqual(ids, [0, 1, 2, 3, 4, 5, 6, 7])
        for i, (is_segment_start, item) in enumerate(items):
            if i in [0, 1, 4, 5]:
                self.assertTrue(is_segment_start)
            else:
                self.assertFalse(is_segment_start)
        heads = idx4segment["head4segment"]
        for i, segment in enumerate(record.read_segments(idx4segment)):
            for j in range(seg_len):
                for feature in features:
                    nptest.assert_equal(
                        dataset[heads[i] + j][feature],
                        segment[feature][j],
                        err_msg="",
                        verbose=True,
                    )

    def test_idx4segment_brokenfeatures(self):
        """Test the idx4segment with some features from dataset missing."""
        # segment len = 3, sequence len =4, break two features
        record, dataset, features = build_broken_dataset([3, 4])
        seq_len = 4
        # encode dataset
        for i, item in enumerate(dataset):
            if i % seq_len == 0:
                # mock start of a sequence
                record.write_item(item, True)
            else:
                record.write_item(item, False)
        record.close_recordfile()

        seg_len = 3
        metadata_segment = record.get_metadata4segment(segment_len=seg_len)
        items = concate_list(metadata_segment["file2segment_items"])
        ids = [item_idx for _, item_idx in items]
        self.assertEqual(len(ids), 6)
        self.assertListEqual(ids, [0, 1, 2, 5, 6, 7])
        for i, (is_segment_start, item) in enumerate(items):
            if item in [0, 5]:
                self.assertTrue(is_segment_start)
            else:
                self.assertFalse(is_segment_start)
        heads = metadata_segment["head4segment"]
        for i, segment in enumerate(record.read_segments(metadata_segment)):
            for j in range(seg_len):
                for feature in features:
                    nptest.assert_equal(
                        dataset[heads[i] + j][feature],
                        segment[feature][j],
                        err_msg="",
                        verbose=True,
                    )

        seg_len = 3
        metadata_segment = record.get_metadata4segment(
            segment_len=seg_len, sub_features=["A100"]
        )
        items = concate_list(metadata_segment["file2segment_items"])
        ids = [item_idx for _, item_idx in items]
        self.assertEqual(len(items), 6)
        self.assertListEqual(ids, [0, 1, 2, 5, 6, 7])
        for i, (is_seg_start, item) in enumerate(items):
            if item in [0, 5]:
                self.assertTrue(is_seg_start)
            else:
                self.assertFalse(is_seg_start)

                seg_len = 3
        heads = metadata_segment["head4segment"]
        for i, segment in enumerate(record.read_segments(metadata_segment)):
            for j in range(seg_len):
                for feature in ["A100"]:
                    nptest.assert_equal(
                        dataset[heads[i] + j][feature],
                        segment[feature][j],
                        err_msg="",
                        verbose=True,
                    )

        metadata_segment = record.get_metadata4segment(
            segment_len=seg_len, sub_features=["i5", "s1"]
        )
        items = concate_list(metadata_segment["file2segment_items"])
        ids = [item_idx for _, item_idx in items]
        self.assertEqual(len(ids), 8)
        self.assertListEqual(ids, [0, 1, 2, 3, 4, 5, 6, 7])
        for i, (is_seg_start, item) in enumerate(items):
            if item in [0, 1, 4, 5]:
                self.assertTrue(is_seg_start)
            else:
                self.assertFalse(is_seg_start)

        heads = metadata_segment["head4segment"]
        for i, segment in enumerate(record.read_segments(metadata_segment)):
            for j in range(seg_len):
                for feature in ["i5", "s1"]:
                    nptest.assert_equal(
                        dataset[heads[i] + j][feature],
                        segment[feature][j],
                        err_msg="",
                        verbose=True,
                    )
        # segment len = 3, sequence len =4, break two features
        record, dataset, features = build_broken_dataset([3, 6])
        seq_len = 4
        # encode dataset
        for i, item in enumerate(dataset):
            if i % seq_len == 0:
                # mock start of a sequence
                record.write_item(item, True)
            else:
                record.write_item(item, False)
        record.close_recordfile()

        seg_len = 3
        metadata_segment = record.get_metadata4segment(segment_len=seg_len)
        items = concate_list(metadata_segment["file2segment_items"])
        ids = [item_idx for _, item_idx in items]
        self.assertEqual(len(ids), 3)
        self.assertListEqual(ids, [0, 1, 2])
        for i, (is_seg_start, item) in enumerate(items):
            if item in [0, 5]:
                self.assertTrue(is_seg_start)
            else:
                self.assertFalse(is_seg_start)
        heads = metadata_segment["head4segment"]
        for i, segment in enumerate(record.read_segments(metadata_segment)):
            for j in range(seg_len):
                for feature in features:
                    nptest.assert_equal(
                        dataset[heads[i] + j][feature],
                        segment[feature][j],
                        err_msg="",
                        verbose=True,
                    )

        # segment len = 3, sequence len =4, break two features
        record, dataset, features = build_broken_dataset([2, 6])
        seq_len = 4
        # encode dataset
        for i, item in enumerate(dataset):
            if i % seq_len == 0:
                # mock start of a sequence
                record.write_item(item, True)
            else:
                record.write_item(item, False)
        record.close_recordfile()

        seg_len = 3
        metadata_segment = record.get_metadata4segment(segment_len=seg_len)
        items = concate_list(metadata_segment["file2segment_items"])
        ids = [item_idx for _, item_idx in items]
        self.assertEqual(len(ids), 0)


def build_simple_dataset():
    """Generate a fake dataset to test methods of Record.
    Returns:
        _type_: _description_
    """
    rootdir = "./output/seqrecord_test/"
    seq_len = 10
    features = {"s1": None, "i5": None, "i7": None, "v100": None, "A100": None}
    dataset = [{} for _ in range(seq_len)]
    for i in range(seq_len):
        for feature in features:
            shape = (np.random.randint(1, 5), np.random.randint(2, 7))
            dtype = np.random.choice(["float32", "int", "bool"])
            dataset[i][feature] = (
                np.random.rand(shape[0], shape[1])
                if dtype == "float32"
                else np.ones(shape=shape)
            )
    record = RSeqRecord(rootdir)
    return record, dataset, features


def build_broken_dataset(feature_is_none_list: List[int]):
    """Generate a fake dataset to test methods of SeqRecord where some features does not exist.
    Params:
        feature_is_none_list (List[str]): indices of data-frame that have missing features
    Returns:
        None
    """
    rootdir = "./output/seqrecord_test/"
    seq_len = 10
    features = {"s1": None, "i5": None, "i7": None, "v100": None, "A100": None}
    dataset = [{} for _ in range(seq_len)]
    for i in range(seq_len):
        for feature in features:
            shape = (np.random.randint(1, 5), np.random.randint(2, 7))
            dtype = np.random.choice(["float32", "int", "bool"])
            if feature != "A100" or (i not in feature_is_none_list):
                dataset[i][feature] = (
                    np.random.rand(shape[0], shape[1])
                    if dtype == "float32"
                    else np.ones(shape=shape)
                )
            else:
                dataset[i][feature] = np.array(None)
    record = RSeqRecord(rootdir)
    return record, dataset, features


def build_seq_dataset():
    """Generate an aritificial dataset to test methods of SeqRecord, w"""
    rootdir = "./output/seqrecord_test/"
    seq_len = 10
    features = {"s1": None, "i5": None, "i7": None, "v100": None, "A100": None}
    dataset = [{} for _ in range(seq_len)]
    for feature in features:
        shape = (np.random.randint(1, 5), np.random.randint(2, 7))
        dtype = np.random.choice(["float32", "int", "bool"])
        for i in range(seq_len):
            dataset[i][feature] = (
                np.random.rand(shape[0], shape[1])
                if dtype == "float32"
                else np.ones(shape=shape)
            )
    record = RSeqRecord(rootdir)
    return record, dataset, features


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
