import argparse
import pickle
import gzip
import warnings
import json
import nltk
from nltk import FreqDist
import random

warnings.simplefilter("ignore")
def train_test_split(lang_dict):
    train_set = {}
    test_set = {}
    min_subdoc_num = get_min_subdocs(lang_dict)
    train_size = int(0.8 * min_subdoc_num)
    for k, v in lang_dict.items():
        train_set[k] = []
        test_set[k] = []
        random.shuffle(v)
        for i in range(min_subdoc_num):
            if i < train_size:
                train_set[k].append(v[i])
            else:
                test_set[k].append(v[i])
    
    return train_set, test_set

def update_lang_profile(profile, doc, ngram_num):
    for n in range(1, ngram_num+1):
        profile.update(character_ngram_as_tuple(doc, n))

def create_lang_profiles(train_dict, ngram_num):
    lang_profiles = {}

    for k, v in train_dict.items():
        lang_prof = FreqDist()
        for subdoc in v:
            update_lang_profile(lang_prof, subdoc, ngram_num)
        lang_profiles[k] = lang_prof

    return lang_profiles

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

def get_min_subdocs(lang_dict):
    return min(len(subdocs) for subdocs in lang_dict.values())

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
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--ngram", dest="ngram", type = int, help="Ngram number")
    parser.add_argument("--ranked", dest="ranked", type = int, help = "If 0, then perplexity else rank list size")
    args, rest = parser.parse_known_args()

#Creating content, label and title lists
X = []
y = []
ids = []

lang_dict = {}
with gzip.open(args.input, "rt") as ifd:
    for line in ifd:
        data = json.loads(line)
        label = data['label']
        lang_dict[label] = lang_dict.get(label, [])
        processed_content = (data['content']).lower().strip()
        lang_dict[label].append(processed_content)

train, test = train_test_split(lang_dict)
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
for lang, doc in test.items():
    y_labels.append(lang)
    y_preds.append(predict_language_from_profiles(doc, lang_profiles, args.ngram, args.ranked))
    
    #     correct += 1
    # total += 1


metrics  = {
    "ac":  accuracy_score(y_labels, y_preds),
#   cm: confusion_matrix(y_test, y_pred)                                                                                           
    "fscore" : f1_score(y_labels, y_preds, average='macro')
    }

metrics = json.dumps(metrics)
with gzip.open(args.model, "wb") as ofd:
    ofd.write(pickle.dumps(lang_profiles))
    
# with gzip.open("vectorizer.pickle.gz", "wb") as ofd:
#     ofd.write(pickle.dumps())

#pickle.dump(cv, open("vectorizer.pickle", "wb"))
#saving the scores

with open(args.scores, "wt") as ofd:
    ofd.write(metrics)
