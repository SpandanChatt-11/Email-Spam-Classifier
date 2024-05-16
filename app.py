import streamlit as st
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import string

ps=PorterStemmer()
def transform_text(text):
    #lowecase
    text=text.lower()
    
    #tokenize
    text=nltk.word_tokenize(text)
    
    #removing special characters
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    
    #removing stopwords and punctuations
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    
    #stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)        

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("SMS/Email Spam Classifier")

input_sms=st.text_area("Enter The message")

if st.button('Predict'):

    transformed_sms=transform_text(input_sms)

    vector_input=tfidf.transform([transformed_sms])

    result= model.predict(vector_input)

    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")