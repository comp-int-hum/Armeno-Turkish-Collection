import argparse
import pickle
import gzip
import warnings
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
warnings.simplefilter("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", help="Output file for pickled model")
    parser.add_argument("--scores", dest="scores", help="Output file for scores") 
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--vectorizer", dest="vectorizer", help="Output file for vectorizer")
    parser.add_argument("--ngram_lower", dest="ng_lower", type = int, help="Min ngram to use")
    parser.add_argument("--ngram_upper", dest="ng_upper", type = int, help="Max ngram to use")
#   parser.add_argument("--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()

    
#Creating content, label and title lists
X = []
y = []
ids = []
with gzip.open(args.input, "rt") as ifd:
    for line in ifd:
        data = json.loads(line)
        y.append(data['label'])
        processed_content = (data['content']).lower().strip()
        X.append(processed_content)
        htid = data['htid']
        ids.append(htid)

le = LabelEncoder()
y = le.fit_transform(y)
y = list(zip(y, ids))


# Not understanding why there is a token pattern and also analyzer = 'char'
cv = CountVectorizer(ngram_range=(args.ng_lower, args.ng_upper), token_pattern = r"(?u)\b\w+\b", analyzer='char')
X = cv.fit_transform(X)

#train test splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train_tuples, y_test_tuples = train_test_split(X, y, random_state=1, test_size = 0.20)

#model creation and prediction
y_train =[label for label, title in y_train_tuples]
y_test = [label for label, title in y_test_tuples]
title_test = [title for lable, title in y_test_tuples]

model = MultinomialNB()
model.fit(x_train, y_train)
# prediction 
y_pred = model.predict(x_test)
# model evaluation
from sklearn.metrics import accuracy_score,f1_score

experiment_name = f"Ngrams from {str(args.ng_lower)} to {str(args.ng_upper)}"
metrics  = {
    "experiment_name" : experiment_name,
    "ac":  accuracy_score(y_test, y_pred),
#   cm: confusion_matrix(y_test, y_pred)                                                                                           
    "fscore" : f1_score(y_test, y_pred, average='macro')
    }

metrics = json.dumps(metrics)
with gzip.open(args.model, "wb") as ofd:
    ofd.write(pickle.dumps(model))
    
with gzip.open(args.vectorizer, "wb") as ofd:
    ofd.write(pickle.dumps(cv))

#pickle.dump(cv, open("vectorizer.pickle", "wb"))
#saving the scores

with open(args.scores, "at") as ofd:
    ofd.write(f"\n{metrics}")
   
   
   
   
   

   
        








    






"""
def split_text(text, max_length):
    while len(text) > max_length:
        yield text[:max_length]
        text = text[max_length:]
    if len(text) > 0:
        yield text

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="data", help="Data input file")
    parser.add_argument("--train", dest="train", help="Train input file (list of line-numbers in the data file)")
    parser.add_argument("--dev", dest="dev", help="Dev input file (list of line-numbers in the data file)")
    parser.add_argument("--model", dest="model", help="Output file for pickled model")
    parser.add_argument("--scores", dest="scores", help="Output file for dev scores")
    parser.add_argument("--n", dest="n", default=3, type=int, help="Value of 'n', as in 'n-gram'")
    parser.add_argument("--max_length", dest="max_length", default=0, type=int, help="Max length of texts to evaluate on")
    args, rest = parser.parse_known_args()

    # Get lists of train/dev item-numbers
    train = []
    dev = []

    with gzip.open(args.train, "rt") as ifd:
        for line in ifd:
            train.append(json.loads(line))

    with gzip.open(args.dev, "rt") as ifd:
        for line in ifd:
            dev.append(json.loads(line))            

    models = {}

    # Train model and get dev instances
    devs = []
    with gzip.open(args.data, "rt") as ifd:
        for i, line in enumerate(ifd):
           j = json.loads(line)
           if i in train:
               label = j["label"]
               # Initialize model for this label if one doesn't exist yet
               # models[label] = models.get(label, None)
               
               # Add counts from j["contents"] to the model[label]
               # ...
               pass
           elif i in dev:
               devs.append(j)
    
    # Apply models to (chunks of) dev
    guesses = []
    golds = []
    for j in devs:
        gold = j["label"]
        # Apply models to j["content"], whichever is more likely is the "guess"
        #
        for chunk in (j["content"] if args.max_length == 0 else split_text(j["content"], args.max_length)):
            # guesses.append(guess)
            golds.append(gold)
            pass

    # compute scores
    scores = {
        # "fscore" : sklearn.metrics.f_measure(...)
    }

    # save model
    with gzip.open(args.model, "wb") as ofd:
        ofd.write(pickle.dumps(models))

    # save scores
    with gzip.open(args.scores, "wb") as ofd:
        ofd.write(pickle.dumps(scores))
"""
