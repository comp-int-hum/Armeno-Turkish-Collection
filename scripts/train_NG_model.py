import argparse
import pickle
import gzip
import warnings
import json
import nltk
from nltk import FreqDist
import random
from nltk.lm import MLE
from nltk.util import pad_sequence, everygrams
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline

nltk.download('punkt')

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

def create_lang_vocabs(train_dict, ngram_num):
    lang_vocabs = {}
    for k, v in train_dict.items():
        lang_model = MLE(ngram_num)
        char_doclist = [list(subdoc) for (_, subdoc) in v]
        print(f"Char doc list len: {len(char_doclist)}")
        # padded_text = list(pad_sequence(subdoc, pad_left = True, 
        #                                  left_pad_symbol = "<s>",
        #                                  pad_right = True,
        #                                  right_pad_symbol = "</s>",
        #                                  n = ngram_num))
        # padded_ngrams = list(everygrams(padded_text, max_len = ngram_num))
        train, vocab = padded_everygram_pipeline(ngram_num, char_doclist)
        lang_model.fit(train, vocab)
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
    tokenized_text = list(text)
    text_data = list(pad_sequence(tokenized_text, pad_left = True, 
                                                    left_pad_symbol = "<s>",
                                                    pad_right = True,
                                                    right_pad_symbol = "</s>",
                                                    n = ngram_num))
    scores = {}
    for lang, vocab in lang_vocabs.items():
        scores[lang] = vocab.perplexity(text_data)
    return min(scores, key=scores.get)

def extract_features(content, ngram_num):
    tokens = nltk.word_tokenize(content)
    ngrams = list(nltk.ngrams(tokens, ngram_num)) # left and right padding?
    return {ngram: True for ngram in ngrams}

# currently no preprocessing
def preprocessing(content):
    return content

def character_ngram_as_str(content, ngram_num):
    return [''.join(ngram) for ngram in nltk.ngrams(content, ngram_num)] # left right padding?

def character_ngram_as_tuple(content, ngram_num):
    return list(nltk.ngrams(content, ngram_num)) # left right padding?

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", help="Output file for pickled model")
    parser.add_argument("--scores", dest="scores", help="Output file for scores")
    parser.add_argument("--input", dest="input", nargs = 2, help="Input file")
    parser.add_argument("--ngram", dest="ngram", type = int, help="Ngram number")
    parser.add_argument("--ranked", dest="ranked", type = int, help = "If 0, then perplexity else rank list size")
    args, rest = parser.parse_known_args()

#Creating content, label and title lists
X = []
y = []
ids = []

lang_dict = {}
with open(args.input[0], "r") as train_input:
    train = json.load(train_input)
    
with open(args.input[1], "r") as test_input:
    test = json.load(test_input)

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


print("NEW RUN")
if args.ranked == 0:
    lang_vocabs = create_lang_vocabs(train, args.ngram)
	
else:
    lang_profiles = create_lang_profiles(train, args.ngram)


# precision = tp / (tp + fp)
# recall = tp / p
# f-score = 2 * (precision x recall) / (precision + recall)
# total = 0
# correct = 0
# tp = 0
# fp = 0
# pos = 0
from sklearn.metrics import accuracy_score,f1_score

y_labels = []
y_preds = []
for lang, docs in test.items():
    for (_, doc) in docs:
        y_labels.append(lang)
        if args.ranked == 0:
            y_preds.append(predict_language_from_vocabs(doc, lang_vocabs, args.ngram))
        else:
            y_preds.append(predict_language_from_profiles(doc, lang_profiles, args.ngram, args.ranked))
    
    #     correct += 1
    # total += 1

model = lang_vocabs if args.ranked == 0 else lang_profiles
metrics  = {
    "ac":  accuracy_score(y_labels, y_preds),
#   cm: confusion_matrix(y_test, y_pred)                                                                                           
    "fscore" : f1_score(y_labels, y_preds, average='macro')
    }

metrics = json.dumps(metrics)
with gzip.open(args.model, "wb") as ofd:
    ofd.write(pickle.dumps(model))
    
# with gzip.open("vectorizer.pickle.gz", "wb") as ofd:
#     ofd.write(pickle.dumps())

#pickle.dump(cv, open("vectorizer.pickle", "wb"))
#saving the scores

with open(args.scores, "wt") as ofd:
    ofd.write(metrics)
