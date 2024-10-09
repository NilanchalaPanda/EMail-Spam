import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords

# Load the saved SVC model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the fitted TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Load NLTK stopwords
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

# Streamlit UI
st.title('Spam Classification App')

user_input = st.text_area('Enter your email content:', '')

if st.button('Classify'):
    if user_input:
        # Preprocess the input text
        processed_text = preprocess_text(user_input)

        # Transform the input using the loaded vectorizer
        input_vector = vectorizer.transform([processed_text])

        # Predict the result using the loaded model
        prediction = model.predict(input_vector)

        # Display the result
        if prediction == 1:
            st.write('This email is *spam*.')
        else:
            st.write('This email is *ham* (not spam).')
    else:
        st.write('Please enter some email content for classification.')
