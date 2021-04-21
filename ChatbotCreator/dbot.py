from nltk.stem import PorterStemmer
import pickle
import spacy
from keras.models import load_model
import discord
from nltk.tokenize import word_tokenize
import time
import sys
import random
import numpy as np
from googlesearch import search
from urllib.request import urlopen
from bs4 import BeautifulSoup

class CreateDiscordBot():
    def __init__(
            self,
            model_file_name_to_use,
            bot_token,
            use_wikipedia=True,
            lang_model="en_core_web_md"):
        self.stemmer = PorterStemmer()
        with open("./data321.pickle", "rb") as f:
            self.data = pickle.load(f)
        with open("./data123.pickle", "rb") as f:
            self.x_vectors, self.train_y, self.classes = pickle.load(f)
        self.nlp = spacy.load(lang_model)
        self.model = load_model("./" + model_file_name_to_use)
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

            try:
                self.inp = message.content
                self.words = word_tokenize(self.inp)
                self.stemmed_words = []
                for self.word in self.words:
                    self.stemmed_words.append(self.stemmer.stem(self.word))
                self.var = " ".join(
                    self.text for self.text in self.stemmed_words)
                self.inp = self.var
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
            self.phrase = self.inp
            self.to_predict.append(self.phrase)
            self.pred_docs = [self.nlp(self.text)
                              for self.text in self.to_predict]
            self.pred_word_vectors = [
                self.x.vector for self.x in self.pred_docs]
            self.pred_word_vectors = np.array(self.pred_word_vectors)
            self.results = self.model.predict(self.pred_word_vectors)
            self.pred = np.argmax(self.results)
            self.results_index = self.pred
            self.pred = self.classes[self.pred]
            for self.cl in self.data:
                if self.cl['class'] == self.pred:
                    self.responses = random.choice(self.cl['responses'])
            if self.use_wikipedia == True:
                if self.results[0][self.results_index] > 0.8:
                    await message.channel.send(self.responses)

                if self.results[0][self.results_index] < 0.8:
                    self.query = self.inp + " wikipedia"

                    self.URLs = []

                    for self.j in search(self.query, tld="co.in", num=2, stop=2, pause=2):
                        self.URLs.append(self.j)

                    self.url = self.URLs[0]

                    self.soup = BeautifulSoup(
                        urlopen(self.url), features="html.parser")

                    self.paragraphs = []

                    for self.p in self.soup.find_all("p"):
                        if len(self.p.get_text()) > 10:
                            self.paragraphs.append(self.p.get_text())

                    self.paragraph = self.paragraphs[:2]
                    self.message_user = str(message.author)
                    await message.channel.send("@"+self.message_user + " " + "According to wikipedia: ")
                    for self.pr in self.paragraph:
                        await message.channel.send(self.pr)
            else:
                await message.channel.send(self.responses)
        self.client.run(self.TOKEN)
