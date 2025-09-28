import streamlit as st
import random
import json
import pickle
import numpy as np
import time
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# ---------------- SETUP ---------------- #
st.set_page_config(page_title="Artificial Neural Network Demo", layout="wide")

lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

# Load data and model
intents = json.loads(open('intents.json').read())
words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))
model = load_model('model/chatbot_model.h5')

# ---------------- FUNCTIONS ---------------- #
def clean_up_sentence(sentence):
    sentence_words = tokenizer.tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

def predict_class(sentence):
    start_time = time.time()
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    end_time = time.time()
    response_time = end_time - start_time

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    if not return_list:
        return_list.append({'intent': 'no_match', 'probability': '0'})
    return return_list, response_time

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = "⚠️ The ANN could not classify this input."
    return result

# ---------------- STREAMLIT UI ---------------- #
st.title("🧠 Artificial Neural Network Demo")
st.markdown("""
Welcome! This demo showcases **how an Artificial Neural Network (ANN)** processes text to recognize intents.
It is **not a chatbot** — the focus is on the **ANN model** itself and its predictions.
""")

st.markdown("### Enter a sentence to see how the ANN interprets it:")

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Input:")

if st.button("Predict") and user_input:
    ints, resp_time = predict_class(user_input)
    res = get_response(ints, intents)

    # Save to history
    st.session_state.history.append({
        "input": user_input,
        "prediction": res,
        "response_time": f"{resp_time:.4f} seconds"
    })

# Display history
st.markdown("### ANN Prediction History:")
for record in st.session_state.history:
    st.markdown(f"**You:** {record['input']}")
    st.markdown(f"**ANN Prediction:** {record['prediction']}  ⏱ {record['response_time']}")
