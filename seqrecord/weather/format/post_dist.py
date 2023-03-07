import os
from seqrecord.weather.seqrecord import WSeqRecord


def main():
    years = range(1979, 2016)
    recorddir = "/datadrive/azure_storage/weathereastus/era5seqrecord/aml_dist"
    for year in years:
        print(f"Gathering {year}'s data")
        sub_dir = os.path.join(recorddir, str(year))
        WSeqRecord.gather_subseqrecords(sub_dir, 12).dump(rank=year - 1979)

    print("Gathering all years' data")
    WSeqRecord.gather_subseqrecords(
        recorddir,
        len(years),
        rank2folder={i: str(year) for i, year in enumerate(years)},
    ).dump()


if __name__ == "__main__":
    main()
