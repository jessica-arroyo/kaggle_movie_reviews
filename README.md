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
def show_word_clouds(data, stopwords=None):
    
    '''
    Entrada:
        data: DataFrame de pandas con las críticas y su evaluación
        stopwords: palabras que por ser muy comunes y/o
                   poco informativas no se tomarán en cuenta al realizar la nube
    '''
    
    pos=' '.join(map(str,data['review'][data['sentiment']=='positive']))
    neg=' '.join(map(str,data['review'][data['sentiment']=='negative']))

    positive_cloud = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stopwords, 
                min_font_size = 10).generate(pos) 
    
    negative_cloud = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stopwords, 
                min_font_size = 10).generate(neg) 

    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.imshow(positive_cloud)
    plt.title('Sentimento positivo')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(negative_cloud)
    plt.title('Sentimento negativo')
    plt.axis('off')

show_word_clouds(data)
```
### Figure 1: Word Clouds for Positive and Negative Reviews

<table>
  <tr>
    <td><img src="show_word_clouds.png" alt="Figure 1"></td>
  </tr>
</table>

At first glance, there doesn't appear to be a strong correlation between the words in the word cloud and the sentiment of the reviews. However, it's noticeable that the word cloud contains HTML elements such as 'br' tags, indicating the presence of HTML formatting within the reviews. Therefore, it's important to preprocess the data appropriately and remove HTML tags to ensure accurate analysis.


