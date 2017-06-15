# coding: utf-8
from argparse import ArgumentParser

from utils import read_as_pandas, save_pandas, map_by_chunk


def csv2hdf5(infile, outfile, append):
    """
    csv to hdf5
    :param infile:
    :param outfile:
    :return:
    """
    if not append:
        df = read_as_pandas(infile)
        save_pandas(df, outfile)

    else:
        map_by_chunk(
            filename=infile,
            read_func=lambda filename: read_as_pandas(filename=filename, by_chunk=True),
            save_func=lambda df: save_pandas(df, filename=outfile, append=True),
            map_func=None,
        )


def main(args):
    csv2hdf5(args.input_file, args.out_file, args.by_chunk)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("input_file", required=True)
    parser.add_argument('output_file', required=True)
    parser.add_argument('by_chunk', type=bool)
    args = parser.parse_args()
    main(args)
