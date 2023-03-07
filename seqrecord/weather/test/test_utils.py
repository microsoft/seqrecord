from seqrecord.utils import distribute_loads
import unittest


class TestUtils(unittest.TestCase):
    def test_distribute_loads(self):
        works = 10
        num_processes = 2
        print(
            f"{works=}, {num_processes=}, the result is: {distribute_loads(works, num_processes)}"
        )

        num_processes = 3
        print(
            f"{works=}, {num_processes=}, the result is: {distribute_loads(works, num_processes)}"
        )

        num_processes = 4
        print(
            f"{works=}, {num_processes=}, the result is: {distribute_loads(works, num_processes)}"
        )

        works = 11
        num_processes = 2
        print(
            f"{works=}, {num_processes=}, the result is: {distribute_loads(works, num_processes)}"
        )

        works = 19
        num_processes = 2
        print(
            f"{works=}, {num_processes=}, the result is: {distribute_loads(works, num_processes)}"
        )
        works = 11
        num_processes = 2
        print(
            f"{works=}, {num_processes=}, the result is: {distribute_loads(works, num_processes)}"
        )
        works = 19
        num_processes = 10
        print(
            f"{works=}, {num_processes=}, the result is: {distribute_loads(works, num_processes)}"
        )
        works = 11
        num_processes = 5
        print(
            f"{works=}, {num_processes=}, the result is: {distribute_loads(works, num_processes)}"
        )

        works = 44
        num_processes = 10
        res = distribute_loads(works, num_processes)
        print(
            f"{works=}, {num_processes=}, the result is: {res}, with maximum load {max(r[1]-r[0] for r in res)}"
        )

        works = 44
        num_processes = 21
        res = distribute_loads(works, num_processes)
        print(
            f"{works=}, {num_processes=}, the result is: {res}, with maximum load {max(r[1]-r[0] for r in res)}"
        )


if __name__ == "__main__":
    unittest.main()
