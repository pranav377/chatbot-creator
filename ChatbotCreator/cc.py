import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import time
import sys
import pickle
import spacy
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")
class ChatbotCreator():

    def __init__(
            self,
            model_file_name_to_save,
            lang_model="en_core_web_md"
        ):
        self.model_name = model_file_name_to_save
        self.lang_model = lang_model

    def createData(self, json_file_path):
        self.train_x = []
        self.train_y = []

        self.file_path = json_file_path
        self.data = pd.read_json(self.file_path)
        self.data = self.data['data']
        self.x = []
        self.y = []

        for self.keys in self.data:
            self.sentences = self.keys['patterns']
            self.cls = self.keys['class']
            self.x.append(self.sentences)
            self.y.append(self.cls)

        self.new_x = []
        self.new_y = []
        self.length_of_patterns = []

        for self.patterns in self.x:
            self.len_of_patterns = len(self.patterns)
            self.length_of_patterns.append(self.len_of_patterns)
            for self.sentences in self.patterns:
                self.new_x.append(self.sentences)

        self.stemmer = PorterStemmer()

        for self.i in range(len(self.length_of_patterns)):
            self.num = self.length_of_patterns[self.i]
            for self.sentences in self.new_x[:self.num]:
                self.train_y.append(self.y[self.i])

        for self.phrase in self.new_x:
            try:
                self.words = word_tokenize(self.phrase)
                self.stemmed_words = []
                for self.word in self.words:
                    self.stemmed_words.append(self.stemmer.stem(self.word))
                self.var = " ".join(
                    self.text for self.text in self.stemmed_words)
                self.train_x.append(self.var)
            except:
                print("[INFO] Installing packages for nltk")
                import nltk

                nltk.download("wordnet")
                nltk.download("stopwords")
                nltk.download("punkt")
                print("[INFO] PLease re-run the program")
                time.sleep(5)
                sys.exit()

        try:
            self.nlp = spacy.load(self.lang_model)
        except:
            print("[INFO] Installing spacy model")
            self.executable = sys.executable
            self.executable = str(self.executable)
            os.system(self.executable + " -m spacy download %s" % self.lang_model)
            print("[INFO] Successfully installed spacy model")
            print("[INFO] Please re-run the program")
            time.sleep(5)
            sys.exit()

        self.docs = [self.nlp(self.text) for self.text in self.train_x]
        self.x_vectors = [self.x.vector for self.x in self.docs]
        self.x_vectors = np.array(self.x_vectors)
        self.shape1 = self.x_vectors.shape[1]
        self.shape2 = self.x_vectors.shape[0]
        self.le = LabelEncoder()
        self.train_y = self.le.fit_transform(self.train_y)
        self.train_y = np.array(self.train_y)
        self.classes = self.le.classes_
        with open("./data123.pickle", "wb") as f:
            pickle.dump((self.x_vectors, self.train_y, self.classes), f)
        with open("./data321.pickle", "wb") as f:
            pickle.dump(self.data, f)

    def createModel(self, init_lr=0.0001, epochs=500, batch_size=32):
        with open("./data123.pickle", "rb") as f:
            self.x_vectors, self.train_y, self.classes = pickle.load(f)
        with open("./data321.pickle", "rb") as f:
            self.data = pickle.load(f)

        print("[INFO] Importing packages for neural network")
        import tensorflow as tf
        import keras.models as Models
        from keras.layers import Dense
        from keras.optimizers import Adam
        from keras.utils import to_categorical
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        print("[INFO] Import successful")
        print("[INFO] defining variables")
        self.train_y = to_categorical(self.train_y)
        self.INIT_LR = init_lr
        self.EPOCHS = epochs
        self.BS = batch_size
        print("[INFO] Creating Neural network")
        tf.config.run_functions_eagerly(True)
        print(self.x_vectors)
        print(self.train_y)
        self.model = Models.Sequential()
        self.model.add(Dense(100, input_shape=(
            len(self.x_vectors[0]),), activation='relu'))
        self.model.add(Dense(120, activation='relu'))
        self.model.add(Dense(112, activation='relu'))
        self.model.add(Dense(len(set(self.classes)), activation='softmax'))
        self.opt = Adam(lr=self.INIT_LR, decay=self.INIT_LR / self.EPOCHS)
        self.model.compile(loss="binary_crossentropy",
                           optimizer=self.opt, metrics=["accuracy"])
        self.es = EarlyStopping(
            monitor='loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
        self.mc = ModelCheckpoint(
            self.model_name, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        print("[INFO] Training model")
        self.model.fit(self.x_vectors, self.train_y,
                       epochs=self.EPOCHS, callbacks=[self.es, self.mc])