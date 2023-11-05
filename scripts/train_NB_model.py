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
y_train =[label for label, htid in y_train_tuples]
y_test = [label for label, htid in y_test_tuples]
title_test = [title for lable, title in y_test_tuples]

model = MultinomialNB()
model.fit(x_train, y_train)
# prediction 
y_pred = model.predict(x_test)

y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

# model evaluation
from sklearn.metrics import accuracy_score,f1_score

# convert to binary tur_Armenian classification task
at_test = [test if test == "tur_Armenian" else "non_tur_Armenian" for test in y_test_labels]
at_preds = [pred if pred == "tur_Armenian" else "non_tur_Armenian" for pred in y_pred_labels]
print(f"at_test: {at_test}")
print(f"at_pred: {at_preds}")

at_errors = [(at_test[i], at_preds[i]) for i in range(len(at_test)) if at_test[i] != at_preds[i]]

metrics  = {
    "overall_ac":  accuracy_score(y_test, y_pred),                                                                                  
    "overall_fscore" : f1_score(y_test, y_pred, average='macro'),
    "at_recall" : accuracy_score(at_test, at_preds),
    "at_fscore": f1_score(at_test, at_preds, average='binary', pos_label="tur_Armenian")
    }

print(at_errors)

metrics = json.dumps(metrics)
with gzip.open("model.pk1.gz", "wb") as ofd:
    ofd.write(pickle.dumps(model))
    
with gzip.open("vectorizer.pickle.gz", "wb") as ofd:
    ofd.write(pickle.dumps(cv))

#pickle.dump(cv, open("vectorizer.pickle", "wb"))
#saving the scores

with open(args.scores, "wt") as ofd:
    ofd.write(metrics)