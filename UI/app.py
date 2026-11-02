
import streamlit as st
import joblib
import pandas as pd
import string

# Load trained model and TF-IDF vectorizer
model = joblib.load("SavedModel/spam_model.pkl")
tfidf = joblib.load("SavedModel/tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

# App UI
st.title("📩 SMS Spam Classifier")

st.write(
    "Enter any SMS message below. "
    "Our Machine Learning model will detect whether it is Spam or Normal."
)

st.divider()

message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You won a free iPhone...",
    height=150
)

if st.button("🔍 Check Message", use_container_width=True):

    if not message.strip():
        st.warning("Please enter a message first!")

    else:
        # Preprocessing
        message_clean = pd.Series([message]).str.lower()

        message_clean = message_clean.str.translate(
            str.maketrans("", "", string.punctuation)
        )

        # TF-IDF transformation
        message_vector = tfidf.transform(message_clean)

        # Model prediction
        prediction = model.predict(message_vector)

        st.divider()

        if prediction[0] == 1:
            st.error("🚨 SPAM MESSAGE DETECTED!")
        else:
            st.success("✅ THIS IS A NORMAL MESSAGE!")

st.divider()
st.caption("Built with Python, Scikit-learn & Streamlit")
