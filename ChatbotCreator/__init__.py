try:
    import pandas as pd
    from keras.models import load_model
    import os
    from sklearn.preprocessing import LabelEncoder
    import sys
    import pickle
    from sklearn import svm
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    import spacy
    import numpy as np
    import warnings
    import random
    import time
    import discord
    from googlesearch import search
    from urllib.request import urlopen
    from bs4 import BeautifulSoup

    __version__ = '0.0.1'


    warnings.filterwarnings("ignore")
    class ChatbotCreator():

        def __init__(self, model_file_name_to_save):
            self.model_name = model_file_name_to_save

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
                self.words = word_tokenize(self.phrase)
                self.stemmed_words = []
                for self.word in self.words:
                    self.stemmed_words.append(self.stemmer.stem(self.word))
                self.var = " ".join(self.text for self.text in self.stemmed_words)
                self.train_x.append(self.var)

            try:
                self.nlp = spacy.load("en_core_web_md")
            except:
                print("[INFO] Installing spacy model")
                self.executable = sys.executable
                os.system(executable + " -m spacy download en_core_web_md")
                print("[INFO] Successfully installed spacy model")
                print("[INFO] Please re-run the program")
                time.sleep(5)
                exit()
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

        def createModel(self, use_neural_network=True, init_lr=0.0001, epochs=500, batch_size=32):
            with open("./data123.pickle", "rb") as f:
                self.x_vectors, self.train_y, self.classes = pickle.load(f)
            with open("./data321.pickle", "rb") as f:
                self.data = pickle.load(f)
            if use_neural_network == True:
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
                self.model.add(Dense(100, input_shape=(len(self.x_vectors[0]),), activation='relu'))
                self.model.add(Dense(120, activation='relu'))
                self.model.add(Dense(112, activation='relu'))
                self.model.add(Dense(len(set(self.classes)), activation='softmax'))
                self.opt = Adam(lr=self.INIT_LR, decay=self.INIT_LR / self.EPOCHS)
                self.model.compile(loss="binary_crossentropy", optimizer=self.opt,metrics=["accuracy"])
                self.es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 
                self.mc = ModelCheckpoint(self.model_name, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
                print("[INFO] Training model")
                self.model.fit(self.x_vectors, self.train_y, epochs=self.EPOCHS, callbacks=[self.es,self.mc])
            else:
                self.clf_svm = svm.SVC(kernel="linear")
                self.clf_svm.fit(self.x_vectors, self.train_y)
                self.model_file_name = self.model_name
                pickle.dump(self.clf_svm, open(self.model_file_name, "wb"))

    class Run():
        def __init__(self, model_file_name, used_neural_network):
            self.model_name = model_file_name
            self.nlp = spacy.load("en_core_web_md")
            with open("./data123.pickle", "rb") as f:
                self.x_vectors, self.train_y, self.classes = pickle.load(f)
            with open("./data321.pickle", "rb") as f:
                self.data = pickle.load(f)
            self.used_neural_network = used_neural_network
            if used_neural_network==True:
                self.model = load_model("./"+ self.model_name)
            else:
                self.model = pickle.load("./" + self.model_name)

        def run(self, input_variable):
            self.phrase = input_variable
            if self.used_neural_network == True:
                warnings.filterwarnings("ignore")
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
                    if self.cl['class'] == self.pred:
                        self.responses = random.choice(self.cl['responses'])
                return self.pred, self.responses, self.results, self.results_index
                
            else:
                self.to_predict = []
                self.model = pickle.load(open("./" + self.model_name, "rb"))
                self.phrase = input_variable
                self.to_predict.append(self.phrase)
                self.pred_docs = [self.nlp(self.text) for self.text in self.to_predict]
                self.pred_word_vectors = [self.x.vector for self.x in self.pred_docs]
                self.pred_word_vectors = np.array(self.pred_word_vectors)
                self.pred = self.model.predict(self.pred_word_vectors)
                self.pred = self.classes[self.pred]
                for self.cl in self.data:
                    if self.cl['class'] == self.pred:
                        self.responses = random.choice(self.cl['responses'])
                return self.pred, self.responses

    class CreateDiscordBot():
        def __init__(self, model_file_name_to_use, bot_token, use_wikipedia=True):
            with open("./data321.pickle", "rb") as f:
                self.data = pickle.load(f)
            with open("./data123.pickle", "rb") as f:
                self.x_vectors, self.train_y, self.classes = pickle.load(f)
            self.nlp = spacy.load("en_core_web_md")
            self.model = load_model("./"+ model_file_name_to_use)
            self.TOKEN = bot_token
            self.client = discord.Client()
            self.use_wikipedia = use_wikipedia

        def run(self):
            @self.client.event
            async def on_ready():
                print(f'{self.client.user.name} has connected to Discord!')

            @self.client.event
            async def on_member_join(member):
                await member.create_dm()
                await member.dm_channel.send(
                    f'Hi {member.name}, welcome to our Discord server!'
                )
                
            @self.client.event
            async def on_message(message):
                if message.author == self.client.user:
                    return
                    
                self.inp = message.content

                self.to_predict = []
                self.phrase = self.inp
                self.to_predict.append(self.phrase)
                self.pred_docs = [self.nlp(self.text) for self.text in self.to_predict]
                self.pred_word_vectors = [self.x.vector for self.x in self.pred_docs]
                self.pred_word_vectors = np.array(self.pred_word_vectors)
                self.results = self.model.predict(self.pred_word_vectors)
                self.pred = np.argmax(self.results)
                self.results_index = self.pred
                self.pred = self.classes[self.pred]
                for self.cl in self.data:
                    if self.cl['class'] == self.pred:
                        self.responses = random.choice(self.cl['responses'])
                if self.use_wikipedia==True:
                    if self.results[0][self.results_index] > 0.8:
                        await message.channel.send(self.responses)
                    
                    if self.results[0][self.results_index] < 0.8:
                        self.query = self.inp + " wikipedia"

                        self.URLs = []

                        for self.j in search(self.query, tld="co.in", num=2, stop=2, pause=2): 
                            self.URLs.append(self.j)

                        self.url = self.URLs[0]

                        self.soup = BeautifulSoup(urlopen(self.url), features="html.parser")

                        self.paragraphs = []

                        for self.p in self.soup.find_all("p"):
                            if len(self.p.get_text())>10:
                                self.paragraphs.append(self.p.get_text())
                                    

                        self.paragraph = self.paragraphs[:2]
                        self.message_user = str(message.author)
                        await message.channel.send("@"+self.message_user + " " + "According to wikipedia: ")
                        for self.pr in self.paragraph:
                            await message.channel.send(self.pr)
                else:
                    await message.channel.send(self.responses)
            self.client.run(self.TOKEN)
            
        
except:
    import sys
    import os
    import time
    print("[INFO] Installing packages...")
    executable = sys.executable
    os.system(executable + " -m pip install pandas")
    os.system(executable + " -m pip install sklearn")
    os.system(executable + " -m pip install spacy")
    os.system(executable + " -m spacy download en_core_web_md")
    os.system(executable + " -m pip install nltk")
    os.system(executable + " -m pip install discord")
    import nltk
    nltk.download("wordnet")
    nltk.download("stopwords")
    nltk.download("punkt")
    print("[INFO] Please Re-run the program")
    time.sleep(5)
