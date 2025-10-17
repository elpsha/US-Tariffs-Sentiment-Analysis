# -----------------------------------------
# Multi-Class Sentiment Analysis with Visualizations
# Classes: Positive, Neutral, Negative
# -----------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier

# For sentiment analysis
from textblob import TextBlob

# -----------------------------------------
# Step 1: Load Dataset
# -----------------------------------------
df = pd.read_csv('us_tariffs_comments_sentiment.csv')

if 'comment_body' not in df.columns:
    raise ValueError("CSV must contain 'comment_body' column!")

# -----------------------------------------
# Step 2: Create Sentiment Polarity and Labels
# -----------------------------------------
if 'sentiment_polarity' not in df.columns:
    df['sentiment_polarity'] = df['comment_body'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

def label_sentiment(polarity):
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_label'] = df['sentiment_polarity'].apply(label_sentiment)

print("Sentiment distribution before balancing:\n", df['sentiment_label'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment_label', data=df, palette='Set2')
plt.title('Sentiment Distribution Before Balancing')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# -----------------------------------------
# Step 3: TF-IDF Vectorization
# -----------------------------------------
vectorizer = TfidfVectorizer(max_features=7000, stop_words='english')
X = vectorizer.fit_transform(df['comment_body'].astype(str))
y = df['sentiment_label']

# -----------------------------------------
# Step 4: Balance Dataset
# -----------------------------------------
ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(X, y)

print("\nSentiment distribution after balancing:\n", y_balanced.value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x=y_balanced, palette='Set1')
plt.title('Sentiment Distribution After Balancing')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# -----------------------------------------
# Step 5: Encode Labels
# -----------------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y_balanced)

# -----------------------------------------
# Step 6: Track Actual Comment Text for Balanced Dataset
# -----------------------------------------
# Use ros.sample_indices_ to get the sampled comments after oversampling
balanced_comments = pd.Series(df['comment_body'].iloc[ros.sample_indices_]).reset_index(drop=True)

# Train-test split
X_train, X_test, y_train, y_test, comments_train, comments_test = train_test_split(
    X_balanced, y_encoded, balanced_comments, test_size=0.2, random_state=42, stratify=y_encoded
)

# -----------------------------------------
# Step 7: Train XGBoost Model
# -----------------------------------------
model = XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    learning_rate=0.1,
    max_depth=6,
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------------------
# Step 8: Predictions
# -----------------------------------------
y_pred = model.predict(X_test)
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

# -----------------------------------------
# Step 9: Evaluation Metrics
# -----------------------------------------
print("\nClassification Report:\n")
print(classification_report(y_test_labels, y_pred_labels))

accuracy = accuracy_score(y_test_labels, y_pred_labels)
print(f"Overall Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test_labels, y_pred_labels, labels=['Positive', 'Neutral', 'Negative'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Positive', 'Neutral', 'Negative'],
            yticklabels=['Positive', 'Neutral', 'Negative'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# -----------------------------------------
# Step 10: Word Clouds
# -----------------------------------------
plt.figure(figsize=(14, 5))
for i, sentiment in enumerate(['Positive', 'Neutral', 'Negative']):
    text = ' '.join(df[df['sentiment_label'] == sentiment]['comment_body'].astype(str))
    wordcloud = WordCloud(width=400, height=300, background_color='white',
                          colormap='viridis').generate(text)
    plt.subplot(1, 3, i + 1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{sentiment} Word Cloud')
plt.tight_layout()
plt.show()

# -----------------------------------------
# Step 11: Accuracy Visualization
# -----------------------------------------
plt.figure(figsize=(5, 4))
plt.bar(['Accuracy'], [accuracy], color='green')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Model Accuracy')
plt.show()

# -----------------------------------------
# Step 12: Save Predictions
# -----------------------------------------
df_results = pd.DataFrame({
    'comment': comments_test.values,
    'true_sentiment': y_test_labels,
    'predicted_sentiment': y_pred_labels
})

df_results.to_csv('us_tariffs_final_predictions.csv', index=False)
df_results[df_results['predicted_sentiment'] == 'Positive'].to_csv('positive_comments.csv', index=False)
df_results[df_results['predicted_sentiment'] == 'Negative'].to_csv('negative_comments.csv', index=False)

print("\nPredictions saved successfully!")
print("Positive comments saved to 'positive_comments.csv'")
print("Negative comments saved to 'negative_comments.csv'")

# -----------------------------------------
# Step 13: Feature Importance (Top Words)
# -----------------------------------------
importances = model.feature_importances_
feature_names = vectorizer.get_feature_names_out()

feat_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

top_features = feat_imp_df.head(20)
print("\nTop 20 important words/features influencing sentiment:\n")
print(top_features)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=top_features, palette='coolwarm')
plt.title('Top 20 Important Words Influencing Sentiment')
plt.xlabel('Importance Score')
plt.ylabel('Word')
plt.show()

# -----------------------------------------
# Step 14: Display Top 5 Positive & Negative Comments
# -----------------------------------------
print("\nTop 5 Positive Comments:")
top_positive = df_results[df_results['predicted_sentiment'] == 'Positive']['comment'].head(5)
for i, comment in enumerate(top_positive, 1):
    print(f"{i}. {comment}")

print("\nTop 5 Negative Comments:")
top_negative = df_results[df_results['predicted_sentiment'] == 'Negative']['comment'].head(5)
for i, comment in enumerate(top_negative, 1):
    print(f"{i}. {comment}")
