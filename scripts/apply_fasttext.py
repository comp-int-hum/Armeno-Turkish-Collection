import json
import gzip
import logging
import argparse
import fasttext

logger = logging.getLogger("apply_fasttext")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--window_size", dest="window_size", type=int, default=400)
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    model = fasttext.load_model(args.model)

    label_count = len(model.get_labels())
    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        for line in ifd:
            j = json.loads(line)
            logger.info("Processing '%s'", j["htid"])
            toks = j["content"].split()
            del j["content"]
            j["lid"] = []
            while len(toks) > 0:
                chunk = " ".join(toks[:args.window_size])
                labels, probs = model.predict(chunk, k=label_count)
                lang_probs = {}
                for k, v in zip(labels, probs):
                    lang_probs[k[9:]] = v
                j["lid"].append(lang_probs)
                toks = toks[args.window_size:]                
            ofd.write(json.dumps(j) + "\n")
