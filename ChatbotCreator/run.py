from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from keras.models import load_model
import os
import warnings
import spacy
import sys
import time
import pickle
import random
import numpy as np


class Run():
    def __init__(self, model_file_namem, lang_model="en_core_web_md"):
        self.model_name = model_file_name
        self.stemmer = PorterStemmer()
        try:
            self.nlp = spacy.load(lang_model)
        except:
            self.executable = sys.executable
            self.executable = str(self.executable)
            os.system(self.executable + " -m spacy download %s" % lang_model)
            print("[INFO] Installled %s" % lang_model)
            print("[INFO] Please re-run the program")
            time.sleep(5)
            sys.exit()
        with open("./data123.pickle", "rb") as f:
            self.x_vectors, self.train_y, self.classes = pickle.load(f)
        with open("./data321.pickle", "rb") as f:
            self.data = pickle.load(f)
        self.model = load_model("./" + self.model_name)

    def run(self, input_variable):
        self.phrase = input_variable
        try:
            self.words = word_tokenize(self.phrase)
            self.stemmed_words = []
            for self.word in self.words:
                self.stemmed_words.append(self.stemmer.stem(self.word))
            self.var = " ".join(self.text for self.text in self.stemmed_words)
            self.phrase = self.var
        except:
            print("[INFO] Installing packages for nltk")
            import nltk

            nltk.download("wordnet")
            nltk.download("stopwords")
            nltk.download("punkt")
            print("[INFO] Please re-run the program")
            time.sleep(5)
            sys.exit()

        self.to_predict = []
        self.to_predict.append(self.phrase)
        self.pred_docs = [self.nlp(self.text) for self.text in self.to_predict]
        self.pred_word_vectors = [self.x.vector for self.x in self.pred_docs]
        self.pred_word_vectors = np.array(self.pred_word_vectors)
        self.results = self.model.predict(self.pred_word_vectors)
        self.pred = np.argmax(self.results)
        self.results_index = self.pred
        self.pred = self.classes[self.pred]
        for self.cl in self.data:
            if self.cl['class'] == self.pred and self.results[0][self.results_index] > 0.7:
                self.responses = random.choice(self.cl['responses'])
            if self.cl['class'] == self.pred and self.results[0][self.results_index] < 0.7:
                self.responses = random.choice(self.cl['responses'])
                try:
                    with open("./low-confidence-patterns.txt", "r") as file:
                        self.old_data = file.read()
                    with open("./low-confidence-patterns.txt", "w") as file:
                        file.write(self.old_data + "\n" + self.phrase)
                        file.close()
                except:
                    with open("./low-confidence-patterns.txt", "w") as file:
                        file.write(self.phrase)
                        file.close()

        return self.pred, self.responses, self.results, self.results_index
