import numpy as np, pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request
import re
from flask_ngrok import run_with_ngrok
import nltk
nltk.download('punkt')
nltk.download('all')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize
from nltk import word_tokenize
import wordrep as wr
import nounrep as nr
import adjrep as ar
from gingerit.gingerit import GingerIt
from parrot import Parrot
import torch
import warnings
import pickle
import sys
warnings.filterwarnings("ignore")

app = Flask(__name__)
run_with_ngrok(app)
model = keras.models.load_model('models/model.h5')
lpph = open('/content/gdrive/MyDrive/CapstoneFlask/trial2/pph.obj','rb')
pp = pickle.load(lpph)

train = pd.read_csv('/content/gdrive/MyDrive/CapstoneFlask/trial2/train.csv')
list_sequences_train = train["comment_text"]
max_features = 22000
tokenizer = Tokenizer(num_words=max_features)
train = tokenizer.fit_on_texts(list(list_sequences_train))

'''
def paraphrase(phrases):
    for phrase in phrases:
    #print("-"*100)
    #print("Input_phrase: ", phrase)
    #print("-"*100)
    para_phrases = pp.augment(input_phrase=phrase)
    for para_phrase in para_phrases:
      print(para_phrase)
'''

def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

def hardcode(comment):
    #print(tokens)
    occupations = [
        'terrorist',
        'terrorists',
        'terrorism',
        'rapist',
        'rapists',
        'molester',
        'molesters',
        'robbery',
        'kidnappers',
        'kidnapping',
        'kidnappings',
        'robbers',
        'theft',
        'thief',
        'thieves',
        'robbers',
        'robber']
    
    print(comment)
    tokens =list(comment.split(' '))
    posverbs = ['like','love','adore','adored']

    negverbs = ['dislike','hate','abhore','disgust','hatred','abhored']
    print(tokens)

    ## checking if our hard code criteria is fulfilled or not.
    find = 0
    for i in tokens:
        if i in occupations:
            find += 1
    #print(find)
    if find == 0:
        return 0

    #tagged = nltk.pos_tag(tokens)
    #print(tagged)
    #vrb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP']
    nc = 0
    for i in tokens:
        if i in negverbs:
            nc += 1
    print(nc)
    if nc%2 == 1:
        return 1 ## falls under criteria and not toxic
    else:
        return 2 ## falls under criteria and toxic + severe toxic = 1

def wordIdentifier(query):
    
    #fucntion of nltk package which tokenizes sentences into words
    tokens=nltk.word_tokenize(query)
    print(tokens)

    #tags tokens with associatd figure of speech(like CD,VB,NN etc)
    tagged=nltk.pos_tag(tokens)
    print(tagged)
    keywords=[]
    nouns = []
    adjective = []
    tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP']
    for i in tagged:
        if i[1] in tags:
            keywords.append(i[0])
        elif i[1] == 'NN' or i[1] == 'NNS':
            nouns.append(i[0])

    for i in tagged:
        if i[1] == 'JJ' or i[1] == 'JJR':
            adjective.append(i[0])
            
    return tokens,keywords,tagged,nouns,adjective

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = [x for x in request.form.values()]
    comment = [preprocess_text(comment[0])]

    #print(comment)
    #print(type(comment[0]))
    tokens = tokenizer.texts_to_sequences(comment)
    if hardcode(comment[0]) != 0:
        if hardcode(comment[0]) == 1:
            toxic = 0.0
            severe_toxic = 0.0
            obscene = 0.0
            threat = 0.0
            insult = 0.0
            identity_hate = 0.0
        elif hardcode(comment[0]) == 2:
            toxic = 1.0
            severe_toxic = 1.0
            obscene = 0.0
            threat = 0.0
            insult = 0.0
            identity_hate = 0.0
    else:
        test = tokenizer.texts_to_sequences(comment)
        final_test = pad_sequences(test, padding='post', maxlen=200)

        prediction = model.predict(final_test)
        toxic = float(prediction[0,0]>0.50)
        severe_toxic = float(prediction[0,1]>0.50)
        obscene = float(prediction[0,2]>0.50)
        threat = float(prediction[0,3]>0.50)
        insult = float(prediction[0,4]>0.50)
        identity_hate = float(prediction[0,5]>0.50)

    
    if toxic == 1 and hardcode(comment[0]) == 0:
        query =' '.join(comment)
        query1=' '.join(comment)
        ## converting some commonly seperated and deformed words
        query = query.replace("ass hole","asshole")
        query = query.replace("butt hole","butthole")
        query = query.replace("homo sex ual","homosexual")
        #query = query.replace("shut the fuck up","stfu")
        query = query.replace("biatch","bitch")
        query = query.replace("b!tch","bitch")
        query = query.replace("13itch","bitch")
        query = query.replace("b i t c h","bitch")
        query = query.replace("bi+ch","bitch")
        query = query.replace("@ss","ass")
        query = query.replace("@$$","ass")
        query = query.replace("a55","ass")
        query = query.replace("trans gender","transgender")
    
        tokens,keywords,tagged,nouns,adjective=wordIdentifier(query)

        if('mother fucker' in query or 'motha fucker' in query or 'motha f ' in query or 'mother f ' in query or 'motherfucker' in query or 'whore' in query or 'slut' in query):
            query = "INPUT STATEMENT TOO DEROGATORY TO SUGGEST ALTERNATIVE"
        

        ## methodically removing instances where "fuck" is used as a noun
        else:
            if('get the fuck out' in query):
                query = query.replace('get the fuck out', 'get out')

            if("don't give a fuck" in query):
                query =  query.replace("don't give a fuck", "don't care")

            if("the fuck " in query):
                query = query.replace('the fuck ','')

            ##methodically removing instances where "fuck" is used as an adverb

            if("fuck you" in query):
                query = query.replace("fuck you","screw you")

            ## replacing all obscene nouns
            for i in nouns:
                if i in nr.nounreplist:
                    query = query.replace(i,nr.nounreplist[i])

            for i in keywords:
                if i in wr.wordreplist:
                    query = query.replace(i,wr.wordreplist[i])

            for i in adjective:
                if i in ar.adjreplist:
                    query = query.replace(i,ar.adjreplist[i])
            if(query==query1):
                query="Toxicity in intent, recommendation not possible !!"
            else:
                # dereferencing too many times
                Query = GingerIt().parse(query)
                query = Query['result']
                paraphrases = pp.augment(input_phrase = query)
                query = paraphrases[0][0]
    elif toxic == 1 and hardcode(comment[0])!=0:
        query = "Toxicity in intent, recommendation not possible !!"
    else:
        query = "No change required :)"

    
    
    return render_template('index.html', toxic = toxic, severe_toxic = severe_toxic, obscene = obscene, threat = threat, insult = insult, identity_hate = identity_hate, Recommendation = query)

if __name__ == "__main__":
    app.run()

