import argparse
import pickle
import logging
import gzip
import json
from nltk.lm import KneserNeyInterpolated, WittenBellInterpolated, Laplace, AbsoluteDiscountingInterpolated
from nltk.util import pad_sequence, everygrams, ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from sklearn.metrics import accuracy_score,f1_score
from collections import namedtuple


logger = logging.getLogger("test_model")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", help="Input file for pickled model")
    parser.add_argument("--scores", dest="scores", help="Output file for scores")
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--ngram", dest="ngram", type = int, help="Ngram number")
    parser.add_argument("--gen_results", dest="gen_results", help = "Output file for all language heatmap data")
    parser.add_argument("--at_results", dest="at_results", help="Output file for tur_Armenian heatmap data")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    with gzip.open(args.model, "rb") as ifd:
        models = pickle.loads(ifd.read())

    logger.info("Loaded model with %d languages", len(models))
    
    TestResult = namedtuple('TestResult', ['htid', 'gold', 'guess', 'dist', 'text'])
    ATTestResult = namedtuple('ATTestResult', ['htid', 'gold', 'guess', 'dist', 'text'])
    
    golds = []
    htids = []
    dists = []    
    guesses = []
    contents = []
    # one set of tuples for tur_Armenian
    # one set of tuples for everything
    # (htid, gold, guess, dists, "text")
    with gzip.open(args.input, "rt") as ifd:
        for lang, texts in json.loads(ifd.read()).items():
            logger.info("Processing language %s", lang)
            total, correct = 0, 0
            for htid, text in texts:
                htids.append(htid)
                seq = list(list(padded_everygram_pipeline(args.ngram, [text])[0])[0])
                dist = {k : v.perplexity(seq) for k, v in models.items()}
                dists.append(dist)
                golds.append(lang)
                guess = sorted(dist.items(), key=lambda x : x[1])[0][0]
                guesses.append(guess)
                contents.append(text)
                total += 1
                if lang == guess:
                    correct += 1
            logger.info("Accuracy for %s: %.3f", lang, correct / total)
    zipped = zip(htids, golds, guesses, dists, contents)
    test_results = [TestResult(*item) for item in zipped]

    at = filter(lambda x: x.gold == "tur_Armenian", test_results)
    at_test_results = [ATTestResult(*item) for item in at]

    with open(args.gen_results, "wt") as gen_output:
        json.dump(test_results, gen_output)

    with open(args.at_results, "wt") as at_output:
        json.dump(at_test_results, at_output)

    metrics  = {
        "accuracy":  accuracy_score(golds, guesses),
        "fscore" : f1_score(golds, guesses, average='macro')
    }
    # chunk
    # 

    with open(args.scores, "wt") as ofd:
        ofd.write(json.dumps(metrics))
