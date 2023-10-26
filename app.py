# importing required libraries
import numpy as np
import streamlit as st
import lightgbm as lgb
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def stemming(content):
    port_stem = PorterStemmer()
    nltk.download('stopwords')
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# preprocess and make prediction with the model
def preprocess(text):
    """
    Makes predictions on given text and returns whether it is a real news or not
    """
    text = stemming(text)

    # converting the textual data to numerical data
    vectorizer = pickle.load(open("tfidf.pk", "rb"))
    process_text = vectorizer.transform([text])

    model = pickle.load(open("lightgbm.pkl", "rb"))
    preds = model.predict(process_text)

    # returning the predictions
    if preds[0] == 0:
        return "Real News"

    return "Fake News"


def main():
    st.title("Fake News Detection")
    text = st.text_area("Enter the News you saw:", placeholder="Copy & Paste the News here...",
                        help="Provide the News Article description you want to verify", height=350, max_chars=10000)

    if st.button("Submit"):
        if text == "":
            st.warning(" Please Enter some News to verify...", icon="⚠️")

        else:
            result = preprocess(text)
            if result == "Real News":
                st.success(result, icon="✅")
            else:
                st.error("Fake News", icon="❌")



if __name__ == "__main__":
    # call main function
    main()
