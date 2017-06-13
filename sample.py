# coding:utf-8
from argparse import ArgumentParser
from random import random
import sys


def random_sample_file(filename, out_name, rate=0.01):
    """
    对文件进行下采样
    :param filename:
    :param out_name:
    :param rate:
    :return:
    """
    with open(filename, 'r') as in_file:
        with open(out_name, 'w') as out_file:
            for line in in_file.readlines():
                r = random()
                if r < rate:
                    out_file.write(line)


def random_sample_file(args):
    random_sample_file(args.src, args.dest, args.rate)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--dest', required=True)
    parser.add_argument('--rate', required=True, type=float)

    args = parser.parse_args()
    random_sample_file(args)
