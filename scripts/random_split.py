import argparse
import gzip
import random
import logging
import math

logger = logging.getLogger("random_split")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--outputs", dest="outputs", nargs="+", help="Output files")
    parser.add_argument("--proportions", dest="proportions", type=float, nargs="+", help="Proportions of data for each output file")
    parser.add_argument("--random_seed", dest="random_seed", type=int, default=None)
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    
    if len(args.outputs) != len(args.proportions):
        raise Exception("The number of output files and the number of proportions must be the same.")

    if sum(args.proportions) > 1 or any([x <= 0 for x in args.proportions]):
        raise Exception("The specified proportions must all be between 0 and 1, and sum to less than or equal to 1.")
    
    if args.random_seed:
        random.seed(args.random_seed)

    logger.info("Reading data from '%s'", args.input)
    lines = []
    with gzip.open(args.input, "rt") as ifd:
        for line in ifd:
            lines.append(line)
            
    random.shuffle(lines)
    line_count = len(lines)
    
    logger.info("Read %d lines", line_count)
    
    for prop, fname in zip(args.proportions, args.outputs):
        num = math.ceil(prop * line_count)
        logger.info("Writing %d items to file '%s'", num, fname)
        with gzip.open(fname, "wt") as ofd:
            for line in lines[:num]:       
                ofd.write(line)
            lines = lines[num:]

