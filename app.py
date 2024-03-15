import re
import io
import os
import nltk
import random
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
import streamlit.components.v1 as components


from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import model_from_json

from tensorflow.keras.utils import to_categorical
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Recommend", page_icon=":notes:", layout="wide")

@st.cache_data
def load_data(filename):
  df = pd.read_csv(filename)
  return df

@st.cache_data
def load_scraped_data(filename):
  df = pd.read_csv(filename, lineterminator='\n')
  return df

@st.cache_data
def load_text(filename):
    df = pd.read_csv(filename, names=['Text', 'Emotion'], sep=';')
    return df


import sys

sys.path.append( "C:/Users/Hansika/Mini Project/.vscode/streamlit_chat" )
from __init__ import message

# Might also import the data from '1_Explore.py' using importlib
albums = load_data('C:/Users/Hansika/Mini Project/Emotion-Based-Music-Recommendation-System/data/albums.csv')
artists = load_data('C:/Users/Hansika/Mini Project/Emotion-Based-Music-Recommendation-System/data/artists.csv')
filteredtracks = load_scraped_data('C:/Users/Hansika/Mini Project/Emotion-Based-Music-Recommendation-System/data/filteredtracks.csv')
 
st.sidebar.header("Recommendation")
st.sidebar.info("Recommendation page lets you ask for Text Based")
st.sidebar.caption("In case of any error, simply refresh the page once.")

st.header('Music Recommendation System')

tab1= st.tabs(['Text Based'])

def mov(res, ind, dir):

    # Defines the container format

    st.markdown("<iframe src='{}'' width='180' height='200' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>".format(dir + res.iloc[ind]['uri']), unsafe_allow_html=True)


def container(desc, dir):

    # Defines the container layout
    col1, col2, col3 = st.columns(3)
    try:
        with col1:
            mov(desc, 0, dir)
        with col2:
            mov(desc, 1, dir)
        with col3:
            mov(desc, 2, dir)
    except:
        pass    

def genreBasedTracks(gen, n):
        db = pd.read_csv('C:/Users/Hansika/Mini Project/Emotion-Based-Music-Recommendation-System/data/filteredtracks.csv', lineterminator='\n')
        db['genres'] = db.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
        db_exp = db.explode("genres")[db.explode("genres")["genres"].isin(gen)]
        db_exp.loc[db_exp["genres"]=="korean pop", "genres"] = "k-pop"
        db_exp_indices = list(db_exp.index.unique())
        db = db[db.index.isin(db_exp_indices)]
        db = db.reset_index(drop=True)
        db = db.head(500).sample(n=n)
        return db

def genreBasedAlbums(gen, n):
        db = pd.merge(albums, artists, on='artist_id', how='inner')
        db['genres'] = db.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
        db_exp = db.explode("genres")[db.explode("genres")["genres"].isin(gen)]
        db_exp.loc[db_exp["genres"]=="korean pop", "genres"] = "k-pop"
        db_exp_indices = list(db_exp.index.unique())
        db = db[db.index.isin(db_exp_indices)]
        db = db.reset_index(drop=True)
        db['uri'] = db.uri.apply(lambda x: x[14:])
        db = db.head(500).sample(n=n)
        return db

def genreBasedArtists(gen, n):
        db = artists
        db['genres'] = db.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
        db_exp = db.explode("genres")[db.explode("genres")["genres"].isin(gen)]
        db_exp.loc[db_exp["genres"]=="korean pop", "genres"] = "k-pop"
        db_exp_indices = list(db_exp.index.unique())
        db = db[db.index.isin(db_exp_indices)]
        db = db.reset_index(drop=True)
        db = db.rename(columns={'artist_id':'uri'})
        db = db.head(500).sample(n=n)
        return db

moodToGenre = {
            'joy': ['k-pop', 'pop', 'hip hop', 'rock'],
            'happy': ['k-pop', 'pop', 'hip hop', 'rock'],
            'calm': ['k-pop', 'pop', 'hip hop', 'rock'],
            'sad': ['soul', 'folk', 'classical', 'r&b'],
            'sadness': ['soul', 'folk', 'classical', 'r&b'],
            'anger': ['metal', 'hip hop', 'rock', 'grunge'],
            'angry': ['metal', 'hip hop', 'rock', 'grunge'],
            'fear': ['country', 'soul', 'jazz', 'classical', 'latin'],
            'fearful': ['country', 'soul', 'jazz', 'classical', 'latin'],
            'disgust': ['country', 'soul', 'jazz', 'classical', 'latin'],
            'love': ['electronic', 'jazz', 'k-pop', 'pop', 'r&b'],
            'neutral': ['electronic', 'jazz', 'k-pop', 'pop', 'r&b'],
            'surprise': ['electronic', 'folk', 'hip hop', 'r&b', 'soul', 'pop'],
            'surprised': ['electronic', 'folk', 'hip hop', 'r&b', 'soul', 'pop']
        }

#nltk.data.path.append("C:/Users/Hansika/AppData/Roaming/nltk_data/corpora")

stop_words = set(stopwords.words("english"))


st.subheader('Text Based Recommendation') 
def lemmatization(text):
    lemmatizer= WordNetLemmatizer()
    text = text.split()
    text=[lemmatizer.lemmatize(y) for y in text]
    return " " .join(text)

def remove_stop_words(text):

    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text=[y.lower() for y in text]
    return " " .join(text)

def removing_punctuations(text):
    # Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )
    
    # Remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan
                
def normalize_text(df):
    df.Text=df.Text.apply(lambda text : lower_case(text))
    df.Text=df.Text.apply(lambda text : remove_stop_words(text))
    df.Text=df.Text.apply(lambda text : removing_numbers(text))
    df.Text=df.Text.apply(lambda text : removing_punctuations(text))
    df.Text=df.Text.apply(lambda text : removing_urls(text))
    df.Text=df.Text.apply(lambda text : lemmatization(text))
    return df

def normalized_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= remove_stop_words(sentence)
    sentence= removing_numbers(sentence)
    sentence= removing_punctuations(sentence)
    sentence= removing_urls(sentence)
    sentence= lemmatization(sentence)
    return sentence

with open('C:/Users/Hansika/Mini Project/Emotion-Based-Music-Recommendation-System/data/text_based/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

with open('C:/Users/Hansika/Mini Project/Emotion-Based-Music-Recommendation-System/data/text_based/labelencoder.pkl', 'rb') as file:
    le = pickle.load(file)

adam = Adam(learning_rate=0.005)
text_based = load_model('C:/Users\Hansika/Mini Project/Emotion-Based-Music-Recommendation-System/data/text_based/text_based.h5', compile=False)
text_based.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

API_KEY = st.secrets["API_KEY"]

API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
headers = {"Authorization": "Bearer {}".format(API_KEY)}

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'bot' not in st.session_state:
    st.session_state['bot'] = random.randint(1, 100)

if 'user' not in st.session_state:
    st.session_state['user'] = random.randint(1, 100)

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def get_text():
    input_text = st.text_input("Enter some text:", "Hey! How you doin'?")
    return input_text 

def gen(): 
    if st.session_state['generated']:
        with chatholder.container():
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], avatar_style="lorelei", seed=st.session_state['bot'], key=str(i))
                message(st.session_state['past'][i], avatar_style="lorelei", seed=st.session_state['user'], flip='true', is_user=True, key=str(i) + '_user')

                sentence = normalized_sentence(st.session_state['past'][i])
                sentence = tokenizer.texts_to_sequences([sentence])
                sentence = pad_sequences(sentence, maxlen=229, truncating='pre')
                result = le.inverse_transform(np.argmax(text_based.predict(sentence), axis=-1))[0]
                prob =  np.max(text_based.predict(sentence))

                # st.caption(f"Mood:{result}; Probability: {prob}; Genres: {', '.join(moodToGenre[result])}")

        with placeholder.container():
            st.write('Recommended Tracks')
            container(genreBasedTracks(moodToGenre[result], 3), 'https://open.spotify.com/embed/track/')
            st.write('Recommended Albums')
            container(genreBasedAlbums(moodToGenre[result], 3), 'https://open.spotify.com/embed/album/')
            st.write('Recommended Artists')
            container(genreBasedArtists(moodToGenre[result], 3), 'https://open.spotify.com/embed/artist/')

col1, col2 = st.columns((1.5,2.5))
with col1: 
    user_input = get_text()
    chatholder = st.empty()
with col2:
    placeholder = st.empty()
if user_input:
    output = query({
        "inputs": {
            "past_user_inputs": st.session_state.past,
            "generated_responses": st.session_state.generated,
            "text": user_input,
        },"parameters": {"repetition_penalty": 1.33},
    })

    if len(st.session_state['generated']) >= 3:
        st.session_state['generated'] = st.session_state['generated'][1:]
        st.session_state['past'] = st.session_state['past'][1:]
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output["generated_text"])

if len(st.session_state['past'])>1:
    if st.session_state['past'][0] != st.session_state['past'][1]:
        gen()
    else:
        st.session_state['generated'] = st.session_state['generated'][1:]
        st.session_state['past'] = st.session_state['past'][1:]
        gen()
else:
    gen()