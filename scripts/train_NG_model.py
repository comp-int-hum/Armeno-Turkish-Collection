import argparse
import pickle
import logging
import gzip
import json
from nltk.lm import KneserNeyInterpolated, WittenBellInterpolated, Laplace, AbsoluteDiscountingInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline


logger = logging.getLogger("train_ngram_model")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", help="Output file for pickled model")
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--ngram", dest="ngram", type = int, help="Ngram number")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    
    train = {}
    with gzip.open(args.input, "rt") as ifd:
        for lang, texts in json.loads(ifd.read()).items():
            train[lang] = []
            for htid, text in texts:
                train[lang].append(text)
    train = {k : padded_everygram_pipeline(args.ngram, v) for k, v in train.items()}
    
    logger.info(
        "Training new ngram model with n=%d on %d languages",
        args.ngram,
        len(train),
    )

    models = {}
    for lang, (seq, vocab) in train.items():
        logger.info("Training model for %s", lang)
        models[lang] = AbsoluteDiscountingInterpolated(args.ngram)
        models[lang].fit(seq, vocab)

    with gzip.open(args.model, "wb") as ofd:
        ofd.write(pickle.dumps(models))
