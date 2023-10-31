import argparse
import pickle
import gzip
import warnings
import json
import nltk
from nltk import FreqDist
import random
from nltk.lm import KneserNeyInterpolated, WittenBellInterpolated, Laplace
from nltk.util import pad_sequence, everygrams, ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline, flatten
from nltk.lm import Vocabulary
from collections import Counter, defaultdict


nltk.download('punkt')


warnings.simplefilter("ignore")


def train_preprocessing(ngram_num, text):
    train, vocab_base = padded_everygram_pipeline(ngram_num, text)
    vocab = Vocabulary(vocab_base, unk_cutoff = 2)
    return train, vocab

def create_lang_vocabs(train_dict, ngram_num=3):
    lang_vocabs = {}
    print(f"NUmber of languages: {len(train_dict.items())}")
    for i, (k, v) in enumerate(train_dict.items()):
        print(f"Language: {k}")
        print(f"Type of subdoc: {type(v[0])}")
        text_list = [list(subdoc) for (_, subdoc) in v]
        train, vocab = train_preprocessing(ngram_num, text_list)
        lang_model = KneserNeyInterpolated(order = ngram_num)
        lang_model.fit(train, vocab)
        lang_vocabs[k] = lang_model
    return lang_vocabs


def predict_language_from_vocabs(text, lang_vocabs, ngram_num):
    tokenized_text = list(text)
    padded_text = list(pad_sequence(tokenized_text, ngram_num, pad_left = True, pad_right = True, left_pad_symbol = "<s>", right_pad_symbol="</s>"))
    test_data = list(ngrams(padded_text, ngram_num))
    scores = {}
    print(f"before perplexity: {len(lang_vocabs.items())}")
    i = 0
    for lang, vocab in lang_vocabs.items():
        print(f"Vocab items {i}")
        i+= 1
        scores[lang] = vocab.perplexity(test_data)
    return min(scores, key=scores.get)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", help="Output file for pickled model")
    parser.add_argument("--scores", dest="scores", help="Output file for scores")
    parser.add_argument("--input", dest="input", nargs = 3, help="Input file")
    parser.add_argument("--ngram", dest="ngram", type = int, help="Ngram number")
    parser.add_argument("--ranked", dest="ranked", type = int, help = "If 0, then perplexity else rank list size")
    parser.add_argument("--load_model", dest="pretrained", default=None)
    args, rest = parser.parse_known_args()

#Creating content, label and title lists
X = []
y = []
ids = []
print("NEW RUN")

lang_dict = {}
with open(args.input[0], "r") as train_input:
    train = json.load(train_input)
    
with open(args.input[1], "r") as test_input:
    test = json.load(test_input)

if args.pretrained != "None":
    print("Loaded ngram model")
    with gzip.open(args.model, 'rb') as ifd:
        models = pickle.loads(ifd.read())
else:
    print("Creating new ngram model")
    models = create_lang_vocabs(train, args.ngram)



total = 0
from sklearn.metrics import accuracy_score,f1_score


y_labels = []
y_preds = []
i = 0
length = len(test.items())
for lang, docs in test.items():
    print(f"Test num: {i} out of {length}")
    i+= 1
    for (_, doc) in docs:
        y_labels.append(lang)
        y_preds.append(predict_language_from_vocabs(doc, models, args.ngram))

metrics  = {
    "ac":  accuracy_score(y_labels, y_preds),                                                                                        
    "fscore" : f1_score(y_labels, y_preds, average='macro')
    }

metrics = json.dumps(metrics)
        
        
with gzip.open(args.model, "wb") as ofd:
    print("Wrote out model")
    ofd.write(pickle.dumps(models))

with open(args.scores, "wt") as ofd:
    ofd.write(metrics)
