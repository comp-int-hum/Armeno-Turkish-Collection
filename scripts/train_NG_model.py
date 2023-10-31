import argparse
import pickle
import gzip
import warnings
import json
import nltk
from nltk import FreqDist
import random
from nltk.lm import KneserNeyInterpolated, WittenBellInterpolated, Laplace
from nltk.util import pad_sequence, everygrams
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline, flatten
from nltk.lm import Vocabulary
from collections import Counter, defaultdict


nltk.download('punkt')

def generate_nested_counts(data):
    counts = defaultdict(lambda: defaultdict(int))
    for ngram in data:
        context, token = tuple(ngram[:-1]), ngram[-1]
        counts[context][token] += 1
    return counts

warnings.simplefilter("ignore")

def update_lang_profile(profile, doc, ngram_num):
    for n in range(1, ngram_num+1):
        profile.update(character_ngram_as_tuple(doc, n))

def create_lang_profiles(train_dict, ngram_num):
    lang_profiles = {}

    for k, v in train_dict.items():
        lang_prof = FreqDist()
        for (_, subdoc) in v:
            update_lang_profile(lang_prof, subdoc, ngram_num)
        lang_profiles[k] = lang_prof

    return lang_profiles


def train_preprocessing(ngram_num, text):
    train, vocab_base = padded_everygram_pipeline(ngram_num, text)
    vocab = Vocabulary(vocab_base, unk_cutoff = 2)
    return train, vocab

def get_ngrams(text, ngram_num):
    return [tuple(text[i:i+ngram_num]) for i in range(len(text)-ngram_num+1)]

def create_lang_vocabs(train_dict, ngram_num=3):
    lang_vocabs = {}
    print(f"NUmber of languages: {len(train_dict.items())}")
    for i, (k, v) in enumerate(train_dict.items()):
        print(f"Language: {k}")
        # print(f"language: {i}")
        #if k == "spa_Latin":
        #     print(f"Result: {v}")

        text_list = [list(subdoc) for (_, subdoc) in v]
        train, vocab = train_preprocessing(ngram_num, text_list)
        

        # flat_train_data = list(flatten(train))
        # counter = generate_nested_counts(flat_train_data)
        # flat_vocab_data = list(flatten(vocab))

        # vocabulary = Vocabulary(flat_vocab_data, unk_cutoff=2)

        # Initialize the KneserNey model
        # lang_model = KneserNeyInterpolated(vocabulary=vocabulary, counter=counter, order=3, discount=0.1)
        lang_model = Laplace(order = ngram_num)
        lang_model.fit(train, vocab)
        # lang_model = KneserNeyInterpolated(ngram_num)
        # lang_model.fit(train, vocab)
        lang_vocabs[k] = lang_model
    return lang_vocabs

def get_rank_dict(lang_profile, max_size):
    ngrams_sorted = lang_profile.most_common(max_size)
    rank_dict = {}
    for rank, (ngram, _) in enumerate(ngrams_sorted):
        rank_dict[ngram] = rank
    
    return rank_dict

def out_of_place_measure(text, lang_profile, ngram_num, max_size = 300):
    text_profile = FreqDist()
    update_lang_profile(text_profile, text, ngram_num)
    text_ranks = get_rank_dict(text_profile, max_size)
    lang_ranks = get_rank_dict(lang_profile, max_size)

    oop_measure = 0
    for ngram, rank in text_ranks.items():
        if ngram in lang_ranks:
            oop_measure += abs(lang_ranks[ngram] - rank)
        else:
            oop_measure += max_size

    return oop_measure

def predict_language_from_profiles(text, lang_profiles, ngram_num, max_size):
    scores = {}
    for lang, profile in lang_profiles.items():
        scores[lang] = out_of_place_measure(text, profile, ngram_num, max_size)
    return min(scores, key=scores.get)

def predict_language_from_vocabs(text, lang_vocabs, ngram_num):
    # tokenized_text = [word_tokenize(sent) for sent in sent_tokenize(text)]

    # text_data = list(pad_sequence(tokenized_text, pad_left = True, 
    # left_pad_symbol = "<s>",
    #                                                 pad_right = True,
    #                                                 right_pad_symbol = "</s>",
    #                                                 n = ngram_num))
    tokenized_text = list(text)
    padded_text = list(pad_sequence(tokenized_text, ngram_num, pad_left = True, pad_right = True, left_pad_symbol = "<s>", right_pad_symbol="</s>"))
    test_data = get_ngrams(padded_text, ngram_num)
    # print(f"Test data type: {type(test_data)}")
    # print(f"Element type: {type(test_data[0])}")
    # print(f"Padded text data: {test_data}")
    scores = {}
    for lang, vocab in lang_vocabs.items():
        scores[lang] = vocab.perplexity(test_data)
    return min(scores, key=scores.get)

def extract_features(content, ngram_num):
    tokens = nltk.word_tokenize(content)
    ngrams = list(nltk.ngrams(tokens, ngram_num)) # left and right padding?
    return {ngram: True for ngram in ngrams}

# currently no (pre)processing other than padding
def processing(content):
    return content

def character_ngram_as_str(content, ngram_num):
    return [''.join(ngram) for ngram in nltk.ngrams(content, ngram_num)] # left right padding?

def character_ngram_as_tuple(content, ngram_num):
    return list(nltk.ngrams(content, ngram_num)) # left right padding?

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

with open(args.input[2], "r") as ta_input:
    ta_test = json.load(ta_input)
if args.pretrained != "None":
    print("Loaded ngram model")
    with gzip.open(args.model, 'rb') as ifd:
        models = pickle.loads(ifd.read())
else:
    print("Creating new ngram model")
    if args.ranked == 0:
        models = create_lang_vocabs(train, args.ngram)
    else:
        models = create_lang_profiles(train, args.ngram)
    # for line in train_input:
    #     data = json.loads(line)
    #     label = data['label']
    #     processed_content = (data['content']).lower().strip()
    #     if label not in lang_dict:
    #         lang_dict[label] = []
    #     lang_dict[label].append(processed_content)


# for k, v in lang_dict.items():
#     if not v:
#         print("Empty key: ", k)




# precision = tp / (tp + fp)
# recall = tp / p
# f-score = 2 * (precision x recall) / (precision + recall)
# total = 0
# correct = 0
# tp = 0
# fp = 0
# pos = 0
total = 0
from sklearn.metrics import accuracy_score,f1_score

# to_predict = "Yo, Juan Gallo de Andrada, escribano de Cámara del Rey nuestro señor, de los que residen en su Consejo, certifico y doy fe que, habiendo visto por los señores dél#  un libro intitulado El ingenioso hidalgo de la Mancha, compuesto por Miguel de Cervantes Saavedra, tasaron cada pliego del dicho libro a tres maravedís y medio; el cual tiene##  ochenta y tres pliegos, que al dicho precio monta el dicho libro docientos y noventa maravedís y medio, en que se ha de vender en papel; y dieron licencia para que a este prec# io se pueda vender, y mandaron que esta tasa se ponga al principio del dicho libro, y no se pueda vender sin ella. Y, para que dello conste, di la presente en Valladolid, a vei# nte días del mes de deciembre de mil y seiscientos y cuatro años."
# prediction = predict_language_from_vocabs(to_predict, models, args.ngram)
# print(f"Prediction: {prediction}")

y_labels = []
y_preds = []
i = 0
length = len(test.items())
# for lang, docs in test.items():
#     print(f"Test num: {i} out of {length}")
#     i+= 1
#     for (_, doc) in docs:
#         y_labels.append(lang)
#         if args.ranked == 0:
#             y_preds.append(predict_language_from_vocabs(doc, models, args.ngram))
#         else:
#             y_preds.append(predict_language_from_profiles(doc, models, args.ngram, args.ranked))
    
        # correct += 1
#     total += 1

metrics  = {
    "ac":  accuracy_score(y_labels, y_preds),
     # cm: confusion_matrix(y_test, y_pred)                                                                                           
    "fscore" : f1_score(y_labels, y_preds, average='macro')
    }

metrics = json.dumps(metrics)

ta_correct = 0

print(f"ta test type: {type(ta_test)}")
pos_len = len(ta_test["positive"])
for i in range(len(ta_test["positive"])):
    doc = ta_test["positive"][i]
    if args.ranked == 0:
        pred = predict_language_from_vocabs(doc, models, args.ngram)
    else:
        pred = predict_language_from_profiles(doc, models, args.ngram, args.ranked)
    if pred == "tur_Armenian":
        ta_correct += 1

not_ta_correct = 0

for i in range(len(ta_test["negative"])):
    doc = ta_test["negative"][i]
    if args.ranked == 0:
        pred = predict_language_from_vocabs(doc, models, args.ngram)
    else:
        pred = predict_language_from_profiles(doc, models, args.ngram, args.ranked)
    if pred != "tur_Armenian":
        not_ta_correct += 1

print(f"TA correct: {ta_correct} {pos_len}")
print(f"Non_TA correct: {not_ta_correct} {pos_len}")

        
        
with gzip.open(args.model, "wb") as ofd:
    print("Wrote out model")
    ofd.write(pickle.dumps(models))
    
# with gzip.open("vectorizer.pickle.gz", "wb") as ofd:
#     ofd.write(pickle.dumps())

#pickle.dump(cv, open("vectorizer.pickle", "wb"))
#saving the scores

with open(args.scores, "wt") as ofd:
    ofd.write(metrics)

# with open("work/armeno-turkish-results.txt", "wt") as filewriter:
#     filewriter.write(
