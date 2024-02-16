# Movie Reviews Sentiment Analysis
The objective of this project is to determine the underlying sentiment in a text, in this case, by classifying user reviews of movies as positive or negative. 
To achieve this, various natural language processing and machine learning techniques have been applied to a pre-labeled dataset from <a href="https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data">Kaggle</a>. 
The results obtained are presented below, and the main conclusions are discussed.

## Text cleaning
Below is the Python code used to extract the data from Kaggle and store it in the variable `data`:

```python
import pandas as pd

data=pd.read_csv("IMDB Dataset.csv")
```
We can then utilize the function `show_word_clouds` to visualize the most frequent words used in positive and negative reviews separately, as illustrated in `Figure 1`.

```python
from wordcloud import WordCloud

def show_word_clouds(data, stopwords=None):
    '''
    Input:
        data: Pandas DataFrame containing reviews and their sentiment evaluation.
        stopwords: Words that, being very common and/or uninformative, will not be considered when generating the word cloud.
    '''
    
    # Extract positive and negative reviews
    pos = ' '.join(map(str, data['review'][data['sentiment'] == 'positive']))
    neg = ' '.join(map(str, data['review'][data['sentiment'] == 'negative']))

    # Generate word clouds for positive and negative sentiments
    positive_cloud = WordCloud(width=800, height=800, 
                               background_color='black', 
                               stopwords=stopwords, 
                               min_font_size=10).generate(pos) 
    
    negative_cloud = WordCloud(width=800, height=800, 
                               background_color='black', 
                               stopwords=stopwords, 
                               min_font_size=10).generate(neg) 

    # Display the word clouds
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(positive_cloud)
    plt.title('Positive Sentiment')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(negative_cloud)
    plt.title('Negative Sentiment')
    plt.axis('off')

show_word_clouds(data)
```
### Figure 1: Word Clouds for Positive and Negative Reviews

<table>
  <tr>
    <td><img src="show_word_clouds.png" alt="Figure 1"></td>
  </tr>
</table>

At first glance, there doesn't appear to be a strong correlation between the words in the word cloud and the sentiment of the reviews. However, it's noticeable that the word cloud contains HTML elements such as 'br' tags, indicating the presence of HTML formatting within the reviews. Therefore, it's important to preprocess the data appropriately and remove HTML tags to ensure accurate analysis. For that reason the function `simple_preprocessor` was created, which removes HTML tags, non-word characters, and converts the text to lowercase. 

```python
import re

def simple_preprocessor(text):
    text = re.sub('<.*?>', '', text)      # Remove HTML tags
    text = re.sub('[\W]+', ' ', text)     # Remove non-word characters
    text = text.lower()                   # Convert text to lowercase
    return text

data_clean = data.copy()  # Copying to avoid modifying the original
data_clean["review"] = data["review"].apply(simple_preprocessor)  # Applying the preprocessing to each row
```
## Splitting Data into Training and Testing Sets

After preprocessing, now with the cleaned data, it will be split into the train and test sets for model training and testing.  In this configuration, the first 35,000 reviews will be used for training, while the remaining 15,000 will be used for testing.

```python
data_train=data_clean[:35000]
data_test=data_clean[35000:]
```
## Pipeline Configuration

Four distinct pipelines are built, each designed to explore different preprocessing techniques and feature representations before applying a logistic regression classifier. The pipelines are defined as follows:

**Pipeline 1:** Utilizes a basic CountVectorizer to convert text data into token counts, followed by standard scaling of features and logistic regression for classification.

**Pipeline 2:** Extends Pipeline 1 by incorporating stop words removal during tokenization with CountVectorizer to improve feature representation.

**Pipeline 3:** Integrates a Term Frequency-Inverse Document Frequency (TF-IDF) transformation alongside CountVectorizer to enhance feature representation for logistic regression.

**Pipeline 4:** Utilizes CountVectorizer with a specific n-gram range of (2, 2) to capture word pairs as features, combined with standard scaling and logistic regression.

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer

pipeline_1 = make_pipeline(CountVectorizer(preprocessor=simple_preprocessor), StandardScaler(with_mean=False), LogisticRegression())
pipeline_1.fit(data_train["review"], data_train["sentiment"])

pipeline_2 = make_pipeline(CountVectorizer(preprocessor=simple_preprocessor, stop_words=stop_words), StandardScaler(with_mean=False), LogisticRegression())
pipeline_2.fit(data_train["review"], data_train["sentiment"])

pipeline_3 = make_pipeline(CountVectorizer(preprocessor=simple_preprocessor), TfidfTransformer(), LogisticRegression())
pipeline_3.fit(data_train["review"], data_train["sentiment"])

pipeline_4 = make_pipeline(CountVectorizer(preprocessor=simple_preprocessor, ngram_range=(2, 2)), StandardScaler(with_mean=False), LogisticRegression())
pipeline_4.fit(data_train["review"], data_train["sentiment"])
```

## Pipeline performance 

The performance of each pipeline is evaluated using 5-fold cross-validation. The evaluation metrics employed include precision, recall, and accuracy, which are computed across all folds to provide a comprehensive assessment of classification performance.

**Precision:** Measures the proportion of correctly predicted positive cases relative to all cases predicted as positive.

**Recall:** Measures the proportion of correctly predicted positive cases relative to all actual positive cases.

**Accuracy:** Represents the proportion of correctly classified cases out of all cases.

The following code shows the evaluation procedure, the resulting metrics are presented in the  `Table 1 `.

```python

# Define the pipelines to be evaluated
pipelines = [pipeline_1, pipeline_2, pipeline_3, pipeline_4]

# Define the evaluation metrics
scoring = {'precision': make_scorer(precision_score, average='weighted'),
           'recall': make_scorer(recall_score, average='weighted'),
           'accuracy': make_scorer(accuracy_score)}

# Evaluate each pipeline using 5-fold cross-validation and the defined metrics
for i, pipeline in enumerate(pipelines):
    scores = cross_validate(pipeline, data_train["review"], data_train["sentiment"], cv=5, scoring=scoring)
    precision = scores['test_precision'].mean()
    recall = scores['test_recall'].mean()
    accuracy = scores['test_accuracy'].mean()
    print(f'Pipeline {i+1}: Precision={precision:.6f}, Recall={recall:.6f}, Accuracy={accuracy:.6f}')

```
### Table 1:  Evaluation Metrics for the Pipelines

| Pipeline | Precision | Recall | Accuracy |
|----------|-----------|--------|----------|
| 1        | 0.864530  | 0.864343 | 0.864343 |
| 2        | 0.860490  | 0.860286 | 0.860286 |
| 3        | 0.891912  | 0.891714 | 0.891714 |
| 4        | 0.892550  | 0.891371 | 0.891371 |

The results from `Table 1` indicate that pipelines 1 and 2, which combine CountVectorizer, StandardScaler, and LogisticRegression, exhibit similar values of precision, recall, and accuracy. However, these values are lower than those obtained by pipelines 3 and 4.

Specifically, pipeline 3, which includes TfidfTransformer in its combination of techniques, achieves the highest values of precision, recall, and accuracy. This suggests that weighting words according to their relevance in documents significantly enhances the model's performance.

On the other hand, pipeline 4 utilizes a CountVectorizer with an n-gram range of (2,2) and also achieves high values of precision, recall, and accuracy. The inclusion of two-word n-grams in the CountVectorizer may allow the model to more effectively capture semantic relationships between words in the texts.

In summary, the results demonstrate that the inclusion of TfidfTransformer or n-grams in text preprocessing can significantly improve the model's performance in terms of precision, recall, and accuracy.

## Model Optimization

Based on the values obtained in the evaluation of the previous 4 pipelines, four new pipelines are created.

**Pipeline 5:** utilizes a custom text analysis function (analyzer) to perform lemmatization of words in the input text.

**Pipeline 6:** employs a custom function (no contractions) to remove contractions from words in the input text.

**Pipeline 7:** is a combination of some of the previous pipelines to assess whether their joint application enhances the model's performance.

**Pipeline 8:** represents an enhancement of Pipeline 4, incorporating the best parameters found using GridSearchCV.

```python
# Custom function for lemmatizing words in text
def analyzer(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return lemmatized_words

# Pipeline 5
pipeline_5 = make_pipeline(CountVectorizer(analyzer=analyzer),
                            StandardScaler(with_mean=False),
                            LogisticRegression())
pipeline_5.fit(data_train["review"], data_train["sentiment"])

# Custom function to remove contractions
def no_contractions(df):
    return df.apply(lambda x: contractions.fix(x))

# Pipeline 6
pipeline_6 = make_pipeline(FunctionTransformer(no_contractions),
                            CountVectorizer(),
                            StandardScaler(with_mean=False),
                            LogisticRegression())
pipeline_6.fit(data_train["review"], data_train["sentiment"])

# Pipeline 7
pipeline_7 = make_pipeline(FunctionTransformer(no_contractions),
                            CountVectorizer(stop_words=stop_words),
                            TfidfTransformer(),
                            LogisticRegression())
pipeline_7.fit(data_train["review"], data_train["sentiment"])

# Pipeline 8
pipeline_8 = make_pipeline(CountVectorizer(ngram_range=(2, 2), max_features=15000),
                            StandardScaler(with_mean=False),
                            LogisticRegression(C=0.1))
pipeline_8.fit(data_train["review"], data_train["sentiment"])
```
The performance of the new pipelines is illustrated in `Table 2`.

### Table 2: 

| Pipeline | Precision | Recall | Accuracy |
|----------|-----------|--------|----------|
| 5        | 0.866065  | 0.865857 | 0.865857 |
| 6        | 0.864486  | 0.864314 | 0.864314 |
| 7        | 0.892817  | 0.892486 | 0.892486 |
| 8        | 0.839951  | 0.839914 | 0.839914 |

It can be observed that pipelines 5 and 6, which employ new techniques, exhibit similar performance to pipelines 1 and 2. However, pipeline 7 achieves better performance compared to the previous ones, suggesting that the combination of different transformations enhances the model's performance. On the other hand, it is noted that pipeline 8 shows a deterioration in its performance when tuning parameters, indicating possible overfitting when finding the best parameters, leading to reduced performance on new cross-validation folds. Since pipeline 7 has demonstrated the best performance among all considered options, it will be used as the final model. In order to further improve its performance, GridSearchCV has been utilized to find the best parameters, and a new pipeline generated as presented in the following code:





