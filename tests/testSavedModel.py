import joblib
import pandas as pd
import string

# Load saved model and vectorizer
model = joblib.load('SavedModel/spam_model.pkl')
tfidf = joblib.load('SavedModel/tfidf_vectorizer.pkl')

# New message
message = ["Congratulations! You won a free iPhone. Claim now!"]

# Preprocessing
message_clean = pd.Series(message).str.lower()

message_clean = message_clean.str.translate(
    str.maketrans('', '', string.punctuation)
)

# Convert text into numbers
message_vector = tfidf.transform(message_clean)

# Prediction
prediction = model.predict(message_vector)

if prediction[0] == 1:
    print("🚨 SPAM MESSAGE")
else:
    print("✅ NORMAL MESSAGE")