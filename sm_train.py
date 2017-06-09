# coding: utf-8
import random


def split(filename, outfile):
    with open(filename, 'r') as fm:
        with open(outfile, 'w') as out:
            for line in fm.readlines():
                if random.random() < 0.1:
                    out.write(line)


if __name__ == '__main__':
    split('train.csv', 'train.sm.csv')
    split('test.csv', 'test.sm.csv')
