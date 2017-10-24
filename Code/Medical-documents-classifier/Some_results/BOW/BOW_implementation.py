import argparse
import json
import csv
import os
import sys

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

## USE AS:
## python BOW_implementation.py --K 10 --span 80 --papers_set ""
## if tf-idf: --tf-idf


parser = argparse.ArgumentParser(description="Classify papers.")
parser.add_argument("--K", type=int, required=True, help="Number of nearest neighbours to consider")
parser.add_argument("--span", help="Number of words used for classification, counting from "
                                   "the start of the abstract", type=int, default=80)
parser.add_argument("--papers_set", help="Path to the paper set.",
                    required=True)

parser.add_argument("--distance_metric", default="minkowski", help="Metric to use to select nearest neighbours. "
                                                                   "Currently Minkowsky and dot product are "
                                                                   "implemented.")

tf_idf_parser = parser.add_mutually_exclusive_group(required=False)
tf_idf_parser.add_argument('--tf_idf', dest='tf_idf', action='store_true')


parser.set_defaults(tf_idf=False)
args = parser.parse_args()
span = args.span
path_to_papers = args.papers_set
join_path = os.path.join

sufix = ("_tfidf" if args.tf_idf else "")

##LOAD DATA

print("loading data")

with open("fold_2011_span_80.json", encoding="utf-8") as fold_file:
    fold_docs = json.load(fold_file)
    year_docs = fold_docs["2011"]["docs"]
    other_docs = fold_docs["others"]["docs"]


train_papers = [' '.join(t["abstract"][:span]) for t in other_docs]
train_labels = [t["classification"] for t in other_docs]

test_papers = [' '.join(t["abstract"][:span]) for t in year_docs]
test_labels = [t["classification"] for t in year_docs]


##VECTORIZE

print("vectorizing")

index_split = len(train_papers)

vectorizer = CountVectorizer()

train_data = vectorizer.fit_transform(train_papers)
test_data = vectorizer.transform(test_papers)

if args.tf_idf:
    print("computing tf_idf")
    transformer = TfidfTransformer(norm="l2")
    train_data = transformer.fit_transform(train_data)
    test_data = transformer.transform(test_data)


## Decomposition, first pca and then tsne

pca = PCA(n_components = 250)
print("starting PCA transformation")
pca_first_decomp = pca.fit_transform(test_data.toarray()[:5000])
print("ended PCA transformation")

tsne = TSNE(perplexity=30, n_components=2, n_iter=5000)
print("starting TSNE transformation")
low_dim_embs = tsne.fit_transform(pca_first_decomp)
print("ended TSNE transformation")

# PLOT SCATTER WITH PYPLOT

for i in range(len(low_dim_embs)):
    try:
        if test_labels[i] == "systematic-review":
            c = "b"
        else:
            c = "r"
        x = low_dim_embs[i][0]
        y = low_dim_embs[i][1]
        plt.scatter(x, y, c=c)
    except:
        pass
#plt.show()

color = {"systematic-review": 6, "primary-study": -6}

with open("tsne_bow{}_scatter_250pca.csv".format(sufix), "w") as csv_out:
    writer = csv.writer(csv_out, delimiter = ",")
    writer.writerow(["x", "y", "label", "color"]) #header
    for i in range(len(low_dim_embs)):
        r = [low_dim_embs[i][0], low_dim_embs[i][1], test_labels[i], color[test_labels[i]]]
        writer.writerow(r)
        if i % 1000 == 0:
            print("Saved: {}".format(i))



classifier = KNeighborsClassifier(n_neighbors=args.K, metric=args.distance_metric)
print("fitting")
classifier.fit(train_data, train_labels)
print("predicting")
prediction = list()
count = 0
for paper in test_data:
    prediction.append(classifier.predict(paper))
    count += 1
    if count % 1000 == 0:
        print("{}/{}".format(count, len(test_labels)), end = "\r")
classes = ["primary-study", "systematic-review"]

if len(test_labels) != len(prediction):
    print("dimensions error. labels: {}, predictions: {}".format(len(test_labels),
                                                                  len(prediction)))


log = open("BOW_LOG{}".format(sufix), "w")
log.write("Calculating metrics...\n")
print("Calculating metrics...\n")
accuracy = metrics.accuracy_score(test_labels, prediction)
log.write("Accuracy: {}\n".format(accuracy))
print("Accuracy: {}\n".format(accuracy))

precision_ps, precision_sr = metrics.precision_score(test_labels, prediction, average=None, labels=["primary-study", "systematic-review"])
log.write("Precision PS: {}\nPrecision SR: {}\n".format(precision_ps, precision_sr))
print("Precision PS: {}\nPrecision SR: {}\n".format(precision_ps, precision_sr))

recall_ps, recall_sr = metrics.recall_score(test_labels, prediction, average=None, labels=["primary-study", "systematic-review"])
log.write("Recall PS: {}\nRecall SR: {}\n".format(recall_ps, recall_sr))
print("Recall PS: {}\nRecall SR: {}\n".format(recall_ps, recall_sr))

f1_ps, f1_sr = metrics.f1_score(test_labels, prediction, average=None, labels=["primary-study", "systematic-review"])
log.write("F1 PS: {}\nF1 SR: {}\n".format(f1_ps, f1_sr))
print("F1 PS: {}\nF1 SR: {}\n".format(f1_ps, f1_sr))

conf_mtx = metrics.confusion_matrix(test_labels, prediction, labels=["primary-study", "systematic-review"])
log.write("Confusion matrix:\n\n")
print("Confusion matrix:\n\n")
tab_conf_mtx = str(conf_mtx)#, headers=["primary-study", "systematic-review"],showindex=["primary-study", "systematic-review"]
log.write(tab_conf_mtx + "\n\n")
print(tab_conf_mtx + "\n\n")

log.write("Saving results...\n")
print("Saving results...\n")

log.close()
