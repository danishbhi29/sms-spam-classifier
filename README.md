# 📩 SMS Spam Classifier


<p align="center">
  A Machine Learning-powered SMS Spam Detection System built with Python, Scikit-learn, NLP, and Streamlit.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikitlearn&logoColor=white" alt="Scikit-learn" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status" />
</p>


## 📌 Overview

SMS Spam Classifier is an end-to-end Machine Learning project that identifies whether an SMS message is **Spam** or **Normal (Ham)**.

The project uses Natural Language Processing (NLP) techniques to preprocess text messages, TF-IDF to extract numerical features, and Logistic Regression to classify messages.

An interactive web application built with Streamlit allows users to enter custom SMS messages and receive instant predictions.

## ✨ Features

- 📩 SMS spam and normal message classification.
- 🧹 Text preprocessing and cleaning.
- 🔤 TF-IDF feature extraction.
- 🤖 Logistic Regression classification.
- ⚖️ Balanced class weights to improve spam detection.
- 📊 Model evaluation using classification metrics and confusion matrix.
- 💾 Saved trained model and vectorizer using Joblib.
- 🌐 Interactive Streamlit web application.
- ⚡ Real-time predictions for custom messages.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| Scikit-learn | Machine Learning algorithms |
| TF-IDF | Text feature extraction |
| Logistic Regression | Classification model |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive web interface |
| Jupyter Notebook | Model development and experimentation |

## 🧠 Machine Learning Pipeline

The project follows an end-to-end Machine Learning workflow.

**1. Data Loading and Exploration**

- Loaded the SMS Spam Collection dataset.
- Explored dataset structure, labels, and message distribution.
- Analyzed the distribution of Spam and Ham messages.

**2. Data Cleaning**

- Removed duplicate records.
- Converted text messages to lowercase.
- Removed punctuation.
- Normalized whitespace.
- Preserved numerical information in messages.

**3. Label Encoding**

Converted categorical labels into numerical values:

| Label | Encoded Value |
|---|---:|
| Ham | 0 |
| Spam | 1 |

**4. Train-Test Split**

Split the dataset into training and testing sets:

- Training: 80%
- Testing: 20%
- Stratified sampling to preserve class distribution.

**5. Feature Extraction**

Applied TF-IDF Vectorization to convert text messages into numerical feature vectors that can be processed by Machine Learning algorithms.

**6. Model Training**

Experimented with multiple classification approaches, including:

- Logistic Regression
- Balanced Logistic Regression
- TF-IDF with unigram and bigram features
- Multinomial Naive Bayes

Selected balanced Logistic Regression based on its spam detection performance and precision-recall trade-off.

**7. Model Evaluation**

Evaluated the trained model using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

**8. Model Persistence**

Saved the trained Logistic Regression model and TF-IDF vectorizer using Joblib for reuse without retraining.

**9. Application Development**

Integrated the saved model into a Streamlit application to perform predictions on new SMS messages.

## 📊 Model Performance

The selected model is **Logistic Regression with balanced class weights**.

Evaluation was performed on a held-out test dataset.

| Metric | Score |
|---|---:|
| Accuracy | 97.10% |
| Spam Precision | 86.86% |
| Spam Recall | 90.84% |
| Spam F1-Score | 88.80% |

### Confusion Matrix

| Actual / Predicted | Ham | Spam |
|---|---:|---:|
| Ham | 885 | 18 |
| Spam | 12 | 119 |

**Key observations:**

- Correctly classified 885 normal messages.
- Correctly detected 119 spam messages.
- Misclassified 18 normal messages as spam.
- Missed 12 spam messages.

The model achieved approximately 90.84% recall for the Spam class, demonstrating its ability to identify most spam messages in the test dataset.

*Note: These results are specific to the selected model and train-test split. Performance may vary on unseen real-world messages.*

## 📁 Project Structure

```text
sms-spam-classifier/
│
├── SavedModel/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── UI/
│   └── app.py
│
├── notebooks/
│   └── spam_classifier.ipynb
│
├── tests/
│   └── testSavedModel.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/danishbhi29/sms-spam-classifier.git
```

### 2. Navigate to the Project Directory

```bash
cd sms-spam-classifier
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python -m streamlit run UI/app.py
```

Open the following URL in your browser:

```text
http://localhost:8501
```

**Important:** The application expects the saved model files inside the `SavedModel/` directory. Make sure the paths in `UI/app.py` point to these files correctly.

## 🖥️ Application Usage

1. Launch the Streamlit application.
2. Enter an SMS message in the input area.
3. Click the **Check Message** button.
4. View the classification result.

The application returns one of the following:

- 🚨 **SPAM MESSAGE DETECTED**
- ✅ **THIS IS A NORMAL MESSAGE**

## 🔬 Experiments and Model Selection

Different approaches were tested during model development.

| Experiment | Purpose |
|---|---|
| Logistic Regression | Establish baseline performance |
| Balanced Logistic Regression | Improve spam-class detection |
| Unigram + Bigram TF-IDF | Explore word combinations |
| Different C values | Experiment with regularization |
| Multinomial Naive Bayes | Compare another text classification algorithm |
| Different Alpha values | Tune Naive Bayes smoothing |

Balanced Logistic Regression was selected as the final model because it provided a useful balance between spam precision and recall.

## 📚 Key Learnings

This project provided practical experience with:

- Building an end-to-end Machine Learning pipeline.
- Text preprocessing and NLP fundamentals.
- TF-IDF feature extraction.
- Training and evaluating classification algorithms.
- Working with imbalanced datasets.
- Understanding precision, recall, and F1-score.
- Improving minority-class detection using class weights.
- Saving and loading Machine Learning models.
- Integrating trained models into web applications using Streamlit.
- Organizing a Python Machine Learning project for GitHub.

## 🔮 Future Improvements

- Experiment with additional Machine Learning algorithms.
- Perform systematic hyperparameter tuning.
- Add prediction confidence scores.
- Improve text preprocessing.
- Add more automated tests.
- Deploy the application online.
- Evaluate performance on additional real-world SMS datasets.

## 👨‍💻 Author

**Danish Farman**

Software Engineering Graduate | AI & Machine Learning Engineer

- GitHub: [@danishbhi29](https://github.com/danishbhi29)

---

<p align="center">
  Built with ❤️ using Python and Machine Learning.
</p>

<p align="center">
  ⭐ If you find this project useful, consider giving it a star!
</p>
