import json
from collections import Counter, defaultdict

class MaxPoolLab:

    def __init__(self):

        self.results = dict() #save results, keys referring to document
        self.meta = dict()

    def add_document(self, doc_identifier):

        if doc_identifier in self.results:
            return("Identifier already exists")
        else:
            self.results[doc_identifier] = list()
            return("Added succesfully")

    def add_document_info(self, doc_identifier, key, value):

        if doc_identifier not in self.meta:
            self.meta[doc_identifier] = dict()
        self.meta[doc_identifier][key] = value

    def delete_document(self, doc_identifier):

        if doc_identifier in self.results:
            del self.results[doc_identifier]
            return("Deleted succesfully")
        else:
            return("Couldn't find identifier")

    def add_word_occurrence(self, word, index, doc_identifier):

        if doc_identifier in self.results:
            self.results[doc_identifier].append((index, word))
        else:
            return("Couldn't find identifier")

    def obtain_results_tuples(self, doc_identifier):

        if doc_identifier in self.results:
            return self.results[doc_identifier]
        else:
            return("Couldn't find identifier")

    def obtain_results(self, doc_identifier = None):

        processed_results = dict()
        if doc_identifier:
            result_tuples = self.obtain_results_tuples(doc_identifier)
            indexes, words = self.process_results(result_tuples)
            processed_results[doc_identifier] = {"indexes_occurrence": indexes,
                                                 "words_occurrence": words,
                                                 "meta": self.meta[doc_identifier]}
        else:
            for doc_identifier, result_tuples in self.results.items():
                indexes, words = self.process_results(result_tuples)
                processed_results[doc_identifier] = {"indexes_occurrence": indexes,
                                                     "words_occurrence": words,
                                                     "meta": self.meta[doc_identifier]}

        return processed_results

    def infographic_from_results(self, doc_identifier = None):

        results = self.obtain_results(doc_identifier)
        infographic = ""
        for _id, result in results.items():
            infographic += "\nid: {}\n".format(_id)

            sorted_by_appareance = sorted(result["indexes_occurrence"].items(),
                                          key = lambda t: t[0])
            infographic += "\n"
            total_words = 0
            total_occurrences = 0
            for _tuple in sorted_by_appareance:
                infographic += "|{}".format(_tuple[1])
                total_occurrences += _tuple[1]
                total_words += 1
            infographic += "|"

            sorted_by_occurrence = sorted(result["words_occurrence"].items(),
                                          key = lambda t: t[1])
            infographic += "\n"
            i = 1
            for _tuple in sorted_by_occurrence[:5]:
                infographic += "{}. {}\n".format(i, _tuple[0])
                i += 1

            infographic += "Total words = {}\n".format(total_words)
            infographic += "Total occurrences = {}\n".format(total_occurrences)


        return infographic

    def process_results(self, result_tuples):

        index_occurrence_dict = dict()
        word_occurrence_dict = dict()

        for _tuple in result_tuples:

            if _tuple[0] in index_occurrence_dict:
                index_occurrence_dict[_tuple[0]] += 1
            else:
                index_occurrence_dict[_tuple[0]] = 1

            if _tuple[1] in word_occurrence_dict:
                word_occurrence_dict[_tuple[1]] += 1
            else:
                word_occurrence_dict[_tuple[1]] = 1

        return index_occurrence_dict, word_occurrence_dict

class DocLab:

    def __init__(self, span, vocab = []):

        self.results = {} #save results, keys referring to document
        self.paper_is = {} #key is id and value is "train" or "test"
        self.vocab = vocab #to count UNK words
        self.span = span

    def add_doc(self, doc_id, paper, _is):
        if paper["abstract"]:
            self.results[doc_id] = paper
            self.paper_is[doc_id] = _is

    def most_used_words(self, top = 10, save = False):
        counts = {word:0 for word in self.vocab}

        for _id, paper in self.results.items():
            for word in self.vocab:
                counts[word] += paper["abstract"][:self.span].count(word)

        if save:
            with open("word_count_per_doc.json", "w") as json_out:
                json.dump(counts, json_out)

        c = Counter(counts)
        for k, v in d.most_common(top):
            print('{}: {}'.format(k, v))

        return counts

    def update_vocab(self, vocab):
        self.vocab = vocab
        self.default_vocab = defaultdict(lambda: 1) #to count unk, form word:0 for every word in the vocab
        for word in self.vocab:
            self.default_vocab[word] = 0

    def reset(self):
        self.results = {}
        self.paper_is = {}

    def count_unk_per_doc(self, save = False):
        counts = {}

        print("Start unk count")
        i = 0
        for _id, paper in self.results.items():
            counts[_id] = sum((self.default_vocab[word] for word in paper["abstract"][:self.span]))

            if i % 100 == 0:
                print("Processed: {}".format(i), end = "\r")

            i += 1

        if save:
            with open("unk_per_doc.json", "w") as json_out:
                json.dump(counts, json_out)

        return counts

    def verbose_unk_lab(self):
        log = "-- <UNK> LAB --\n"


        counts = self.count_unk_per_doc()


        ### unk and doc count per class
        class_unk = {"systematic-review": {"train": 0, "test": 0}, "primary-study": {"train": 0, "test": 0}}
        class_count = {"systematic-review": {"train": 0, "test": 0}, "primary-study": {"train": 0, "test": 0}}

        for _id, count in counts.items():
            doc_class = self.results[_id]["classification"]
            doc_is = self.paper_is[_id]
            class_unk[doc_class][doc_is] += count
            class_count[doc_class][doc_is] += 1

        log += "Systematic review train: {} documents, {} <unk> count, {} average <unk> per document\n".format(
                                                                                                    class_count["systematic-review"]["train"],
                                                                                                    class_unk["systematic-review"]["train"],
                                                                                                    class_unk["systematic-review"]["train"] / class_count["systematic-review"]["train"]
                                                                                                        )
        log += "Systematic review test: {} documents, {} <unk> count, {} average <unk> per document\n".format(
                                                                                                    class_count["systematic-review"]["test"],
                                                                                                    class_unk["systematic-review"]["test"],
                                                                                                    class_unk["systematic-review"]["test"] / class_count["systematic-review"]["test"]
                                                                                                        )

        log += "Primary study train: {} documents, {} <unk> count, {} average <unk> per document\n".format(
                                                                                                    class_count["primary-study"]["train"],
                                                                                                    class_unk["primary-study"]["train"],
                                                                                                    class_unk["primary-study"]["train"] / class_count["primary-study"]["train"]
                                                                                                        )
        log += "Primary study test: {} documents, {} <unk> count, {} average <unk> per document\n".format(
                                                                                                    class_count["primary-study"]["test"],
                                                                                                    class_unk["primary-study"]["test"],
                                                                                                    class_unk["primary-study"]["test"] / class_count["primary-study"]["test"]
                                                                                                        )
        return log


    def save(self, save_id = ""):
        save_id = "_" + str(save_id)
        with open("doc_lab_out{}.json".format(save_id), "w") as out:
            json.dump(self.results, out)

    def reset(self):
        self.results = {}
