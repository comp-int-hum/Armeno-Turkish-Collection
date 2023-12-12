import json
import gzip
import logging
import argparse

logger = logging.getLogger("doc_id")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--output", dest="output", nargs = "+", help="Output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    
    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        for line in ifd:
            j = json.loads(line)
            logger.info("ID'ing '%s'", j["htid"])
            logger.info(f"{j.keys()}")
            logger.info(f"{j['lid'][0].keys()}")
            break