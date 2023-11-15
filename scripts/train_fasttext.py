import argparse
import gzip
import json
import os.path
import os
import fasttext
import tempfile


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", help="Input file")
    parser.add_argument("--dev", dest="dev", help="Input file")
    parser.add_argument("--model", dest="model", help="Output file")
    parser.add_argument("--min_char_ngram", dest="min_char_ngram", default=2, type=int)
    parser.add_argument("--max_char_ngram", dest="max_char_ngram", default=4, type=int)
    parser.add_argument("--learning_rate", dest="learning_rate", default=0.1, type=float)
    parser.add_argument("--rep_size", dest="rep_size", default=16, type=int)
    parser.add_argument("--word_context_size", dest="word_context_size", default=5, type=int)    
    parser.add_argument("--epochs", dest="epochs", default=5, type=int)    
    parser.add_argument("--min_word_occurrences", dest="min_word_occurrences", default=3, type=int)
    parser.add_argument("--min_label_occurrences", dest="min_label_occurrences", default=1, type=int)    
    parser.add_argument("--max_word_ngram", dest="max_word_ngram", default=2, type=int)
    args, rest = parser.parse_known_args()

    _, train_fname = tempfile.mkstemp()
    _, dev_fname = tempfile.mkstemp()

    try:
        for ifn, ofn in [(args.train, train_fname), (args.dev, dev_fname)]:
            with gzip.open(ifn, "rt") as ifd, open(ofn, "wt") as ofd:
                for line in ifd:
                    j = json.loads(line)
                    label = j["label"]
                    ofd.write("__label__{} {}".format(j["label"], j["content"]).strip().replace("\n", " ") + "\n")


        model = fasttext.train_supervised(
            train_fname,
            minn=args.min_char_ngram,
            maxn=args.max_char_ngram,
            minCount=args.min_word_occurrences,
            #wordNgrams=args.max_word_ngram,
            dim=args.rep_size,
            loss="hs"
            #epoch=args.epochs,
            #autotuneValidationFile=dev_fname
        )
        print(model.test(dev_fname))
        model.save_model(args.model)
    except Exception as e:
        raise e
    finally:
        for fname in [train_fname, dev_fname]:
           if os.path.exists(fname):
               os.remove(fname)
