import argparse
import pickle
import logging
import gzip
import json
from nltk.lm import KneserNeyInterpolated, WittenBellInterpolated, Laplace, AbsoluteDiscountingInterpolated
from nltk.util import pad_sequence, everygrams, ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from sklearn.metrics import accuracy_score,f1_score


logger = logging.getLogger("test_model")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", help="Input file for pickled model")
    parser.add_argument("--scores", dest="scores", help="Output file for scores")
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--ngram", dest="ngram", type = int, help="Ngram number")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    with gzip.open(args.model, "rb") as ifd:
        models = pickle.loads(ifd.read())

    logger.info("Loaded model with %d languages", len(models))
    
    golds = []
    dists = []    
    guesses = []
    with gzip.open(args.input, "rt") as ifd:
        for lang, texts in json.loads(ifd.read()).items():
            logger.info("Processing language %s", lang)
            total, correct = 0, 0
            for htid, text in texts:
                seq = list(list(padded_everygram_pipeline(args.ngram, [text])[0])[0])
                dist = {k : v.perplexity(seq) for k, v in models.items()}
                dists.append(dist)
                golds.append(lang)
                guess = sorted(dist.items(), key=lambda x : x[1])[0][0]
                guesses.append(guess)
                total += 1
                if lang == guess:
                    correct += 1
            logger.info("Accuracy for %s: %.3f", lang, correct / total)
            
    metrics  = {
        "accuracy":  accuracy_score(golds, guesses),
        "fscore" : f1_score(golds, guesses, average='macro')
    }

    with open(args.scores, "wt") as ofd:
        ofd.write(json.dumps(metrics))
