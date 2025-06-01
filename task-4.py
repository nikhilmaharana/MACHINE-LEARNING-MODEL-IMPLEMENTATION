# ✅ Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ✅ Step 2: Load Dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
print("Sample Data:")
print(data.head())

# ✅ Step 3: Preprocessing
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})
print("\nLabel Distribution:\n", data['label'].value_counts())

# ✅ Step 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label_num'], test_size=0.2, random_state=42)

# ✅ Step 5: Vectorize Text Data
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ✅ Step 6: Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ✅ Step 7: Predict and Evaluate
y_pred = model.predict(X_test_vec)

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ✅ Step 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ✅ Step 9: Try with a custom message
custom_msg = ["Congratulations! You've won a free ticket. Call now to claim."]
custom_vec = vectorizer.transform(custom_msg)
prediction = model.predict(custom_vec)
print("\nCustom Message Prediction:", "Spam" if prediction[0] == 1 else "Ham")
