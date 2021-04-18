import os
import subprocess
import platform
import calendar
import sys
from DateTime import Timezones
import wikipedia
from wikipedia.wikipedia import summary
from wolframalpha import Result
zones = set(Timezones())
import pytz
from DateTime import *
import speech_recognition
recognizer = speech_recognition.Recognizer()
import pyttsx3 as tts
speaker = tts.init()
rate = speaker.getProperty('rate')
speaker.setProperty('rate', 150)
# print(rate)
volume = speaker.getProperty('volume')
# print(volume)
speaker.setProperty('volume',1.0)
voices= speaker.getProperty('voices')
speaker.setProperty('voice', voices[1].id)
# speaker.say("TTS: True")
# speaker.runAndWait()



from re import T, VERBOSE
from abc import ABCMeta, abstractmethod
import re
import pprint
import warnings
import json
import datetime
from urllib.request import HTTPPasswordMgrWithDefaultRealm
import webbrowser
import wikipedia
import wolframalpha
import requests
import time
# import pyfirmata
# from pyfirmata import Arduino, util
# board = pyfirmata.Arduino('COM4')
import random
import pickle
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

nltk.download('punkt', quiet=False)
nltk.download('wordnet', quiet=False)

# speaker.say("Creating Classes")
# speaker.runAndWait()

class IAssistant(metaclass=ABCMeta):
    @abstractmethod
    def train_model(self):
        """ Implemented in child class """
    @abstractmethod
    def request_tag(self, message):
        """ Implemented in child class """
    @abstractmethod
    def get_tag_by_id(self, id):
        """ Implemented in child class """
    @abstractmethod
    def request_method(self, message):
        """ Implemented in child class """
    @abstractmethod
    def request(self, message):
        """ Implemented in child class """


class GenericAssistant(IAssistant):
    def __init__(self, intents, intent_methods={}, model_name="assistant_model"):
        self.intents = intents
        self.intent_methods = intent_methods
        self.model_name = model_name
        if intents.endswith(".json"):
            self.load_json_intents(intents)
        self.lemmatizer = WordNetLemmatizer()
    def load_json_intents(self, intents):
        self.intents = json.loads(open(intents).read())
    def train_model(self):
        self.words = []
        self.classes = []
        documents = []
        ignore_letters = ['!', '?', ',', '.']
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word = nltk.word_tokenize(pattern)
                self.words.extend(word)
                documents.append((word, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        training = []
        output_empty = [0] * len(self.classes)
        for doc in documents:
            bag = []
            word_patterns = doc[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])
        random.shuffle(training)
        training = np.array(training)
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(train_y[0]), activation='softmax'))
        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.hist = self.model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    def save_model(self, model_name=None):
        if model_name is None:
            self.model.save(f"{self.model_name}.h5", self.hist)
            pickle.dump(self.words, open(f'{self.model_name}_words.pkl', 'wb'))
            pickle.dump(self.classes, open(f'{self.model_name}_classes.pkl', 'wb'))
        else:
            self.model.save(f"{model_name}.h5", self.hist)
            pickle.dump(self.words, open(f'{model_name}_words.pkl', 'wb'))
            pickle.dump(self.classes, open(f'{model_name}_classes.pkl', 'wb'))
    def load_model(self, model_name=None):
        if model_name is None:
            self.words = pickle.load(open(f'{self.model_name}_words.pkl', 'rb'))
            self.classes = pickle.load(open(f'{self.model_name}_classes.pkl', 'rb'))
            self.model = load_model(f'{self.model_name}.h5')
        else:
            self.words = pickle.load(open(f'{model_name}_words.pkl', 'rb'))
            self.classes = pickle.load(open(f'{model_name}_classes.pkl', 'rb'))
            self.model = load_model(f'{model_name}.h5')
    def _clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    def _bag_of_words(self, sentence, words):
        sentence_words = self._clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
        return np.array(bag)
    def _predict_class(self, sentence):
        p = self._bag_of_words(sentence, self.words)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.1
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list
    def _get_response(self, ints, intents_json):
        try:
            tag = ints[0]['intent']
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if i['tag']  == tag:
                    result = random.choice(i['responses'])
                    break
        except IndexError:
            result = "I don't understand!"
        return result
    def request_tag(self, message):
        pass
    def get_tag_by_id(self, id):
        pass
    def request_method(self, message):
        pass
    def request(self, message):
        ints = self._predict_class(message)
        if ints[0]['intent'] in self.intent_methods.keys():
            self.intent_methods[ints[0]['intent']]()
        else:
            print(self._get_response(ints, self.intents))


# speaker.say("Assigning functions")
# speaker.runAndWait()

todo_list = ['Go Shopping','Clean room', 'Figure it out']

def create_note():
    global recognizer
    speaker.say("What do you want to write onto your note?")
    speaker.runAndWait()
    done = False
    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio =  recognizer.listen(mic)
                note = recognizer.recognize_google(audio)
                note = note.lower()
                speaker.say("Choose a filename!")
                speaker.runAndWait()
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic)
                filename = recognizer.recognize_google(audio)
                filename = filename.lower()
            with open(filename, 'w') as f:
                f.write(note)
                done =  True
                speaker.say(f"I successfully created the note {filename}")
                speaker.runAndWait()
        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            speaker.say("Audio Error")
            speaker.runAndWait()

def searchGoogle():
    global recognizer
    speaker.say("Keyword Please?")
    speaker.runAndWait()
    done = False
    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio =  recognizer.listen(mic)
                keyword = recognizer.recognize_google(audio)
                keyword = keyword.lower()
                webbrowser.open_new_tab("https://www.google.com/search?q=" + keyword)
                speaker.say("Roger")
                speaker.runAndWait()
        except speech_recognition.UnknownValueError:
            done = True
            recognizer = speech_recognition.Recognizer()

def searchYoutube():
    global recognizer
    speaker.say("Keyword Please?")
    speaker.runAndWait()
    done = False
    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio =  recognizer.listen(mic)
                keyword = recognizer.recognize_google(audio)
                keyword = keyword.lower()
                webbrowser.open_new_tab("https://www.youtube.com/results?search_query=" + keyword)
                speaker.say("Roger")
                speaker.runAndWait()
        except speech_recognition.UnknownValueError:
            done = True
            recognizer = speech_recognition.Recognizer()

def playPremDhillon():
    premDhillon = 'https://www.youtube.com/watch?v=TxPtDhRiROI&ab_channel=SidhuMooseWala'
    webbrowser.open_new_tab(premDhillon)

def playProphec():
    prophec = 'https://www.youtube.com/watch?v=sfbw-YvKiVI&ab_channel=ThePropheC'
    webbrowser.open_new_tab(prophec)

def killChrome():
    try:
        os.system('TASKKILL /F /IM chrome.exe')
    except:
        pass

def killVlc():
    try:
        os.system('TASKKILL /F /IM vlc.exe')
    except:
        pass

def coldWar():
    coldwarpath = "E:\COLD WAR\Call of Duty Black Ops Cold War\Black Ops Cold War Launcher.exe"
    try:
        subprocess.Popen(coldwarpath)
    except:
        pass

def add_todo():
    global recognizer
    speaker.say("What to do do you want to add?")
    speaker.runAndWait()
    done = False
    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic)
                item = recognizer.recognize__google(audio)
                item = item.lower()
                todo_list.append(item)
                done=True
                speaker.say(f"added {item} to to do list!")
                speaker.runAndWait()
        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            speaker.say("Audio Error")
            speaker.runAndWait()

def show_todos():
    speaker.say("The items on your to do list are as following")
    for item in list:
        speaker.say(item)
        speaker.runAndWait()

def audio_to_str():
    global recognizer
    speaker = speech_recognition.Recognizer()
    while True:
        with speech_recognition.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=0.1)
            audio = recognizer.listen(mic)
            print("Listening...")
            try:
                phrase =  recognizer.recognize_google(audio)
                phrase = phrase.lower()
                print(f'User said: {phrase}\n')
            except speech_recognition.UnknownValueError:
                recognizer = speech_recognition.Recognizer()

def artFromUrl(): #natural language prosessing article scrapinf from html with user input url.
    url = input('')
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    print(f'Tiltle: {article.title}')
    print(f'Authors: {article.authors}')
    print(f'Publication Date: {article.publish_date}')
    print(f'Summary: {article.summary}')
    speaker.say(article.summary)
    speaker.runAndWait()

def systemInfo():
    platform_machine = platform.machine()
    platform_machine = str(platform_machine)
    print(platform_machine)
    speaker.say(platform_machine)
    speaker.runAndWait()
    os = platform.system()
    os = str(os)
    speaker.say(os)
    speaker.runAndWait()
    print(os)
    processor = platform.processor()
    processor = str(processor)
    speaker.say(processor)
    speaker.runAndWait()
    print(processor)

def open_youtube():
    webbrowser.open_new_tab("https://www.youtube.com")
    speaker.say("youtube is open now")
    speaker.runAndWait()

def open_google():
    webbrowser.open_new_tab("https://www.google.com")
    speaker.say("search is open now")
    speaker.runAndWait()


def quit():
    speaker.say("bye")
    speaker.runAndWait()
    sys.exit(0)

def greeting():
    hour=datetime.datetime.now().hour
    if hour>=0 and hour<12:
        speaker.say("Hi, Good Morning")
        speaker.runAndWait()
        print("Hi, Good Morning")
    elif hour>=12 and hour<18:
        speaker.say("Hi, Good Afternoon")
        speaker.runAndWait()
        print("Hi, Good Afternoon")
    else:
        speaker.say("Hi, Good Evening")
        speaker.runAndWait()
        print("Hi, Good Evening")

def hello():
    greeting()

mappings = {
    "greetings": hello,
    "create_note": create_note,
    "add_todo": add_todo,
    "show_todos": show_todos,
    "exit": quit,
    "systemInfo": systemInfo,
    "artFromUrl": artFromUrl,
    "coldWar": coldWar,
    "killVlc": killVlc,
    "killChrome": killChrome,
    "playPremDhillon": playPremDhillon,
    "playProphec": playProphec,
    "searchGoogle": searchGoogle,
    "searchYoutube": searchYoutube
}

assistant = GenericAssistant('intents.json', intent_methods=mappings)
assistant.train_model()
assistant.save_model()
print('Data Structure Ready')
# speaker.say("Data Structure Ready")
# speaker.runAndWait()


speaker.say("Starting listening loop")
speaker.runAndWait()

def dayMonthTime():
    dayAndDate = datetime.datetime.now()
    message = "{:%A, %B %d, %Y, %H:%M:%S}"
    print(message.format(dayAndDate))

def timeAlone():
    time = datetime.datetime.now()
    message = "{:,%H:%M%S}"
    print(message.format())

def listening_loop():
    while True:
        debug=True
        try:
            with speech_recognition.Microphone() as mic:
                global recognizer
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic)
                dayMonthTime()
                message = recognizer.recognize_google(audio)
                message = message.lower()
                print(f'User said: {message}\n')
            assistant.request(message)
        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()


listening_loop()
# import threading
# task1 = threading.Thread(target=listening_loop)
# task1.start()
