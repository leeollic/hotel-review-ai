from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


reviews = [
    "The room was clean and comfortable",
    "Excellent service and very friendly staff",
    "I loved the breakfast and the location",
    "The hotel was amazing and very luxurious",
    "Terrible experience, the room was dirty",
    "The staff was rude and unhelpful",
    "Very noisy and uncomfortable stay",
    "I hated the food and the service"
]

labels = [1, 1, 1, 1, 0, 0, 0, 0]  

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)


model = LogisticRegression()
model.fit(X, labels)


test_reviews = [
    "The room was beautiful and the staff were kind",
    "The hotel was dirty and the service was awful"
]

X_test = vectorizer.transform(test_reviews)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)


for i, review in enumerate(test_reviews):
    sentiment = "Positive" if predictions[i] == 1 else "Negative"
    confidence = round(max(probabilities[i]) * 100, 2)

    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print("Confidence:", confidence, "%")
    print("-" * 40)
