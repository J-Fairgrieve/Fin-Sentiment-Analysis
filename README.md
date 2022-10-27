# Financial News Sentiment Analysis

James Fairgrieve

 - Linkedin: [https://www.linkedin.com/in/jfairgrieve/](https://www.linkedin.com/in/jfairgrieve/)
 - Portfolio: [https://j-fairgrieve.github.io/](https://j-fairgrieve.github.io/)

## Contents
- About the Project
- Resources
- Financial News EDA
- Multinomial NB Model

## About the Project

This project uses data from Kaggle, kindly uploaded by [Ankur Sinha](https://www.kaggle.com/ankurzing). The dataset itself contains the sentiments for financial news headlines for retail investors. Sentiments are split into three categories:

 - Positive
 - Neutral
 - Negative

The dataset itself only contains two columns: the article title and the sentiment. There are two sections in total for this project:

 - EDA: Exploring the dataset as best as possible with the limited data available
 - Multinomial Naive Bayes Model: A model ideal for text-based data

## Resources
 - [Kaggle Dataset](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)

## Financial News EDA
###### You can find the full code for this section [here](https://github.com/J-Fairgrieve/Fin-Sentiment-Analysis/blob/main/1%20-%20Financial%20News%20EDA.ipynb).

#### 1.1 Exploring the Data
In total, there are 4,846 news articles in the dataset. The number of articles for each sentiment are as follows:

![Article Sentiment Counts](https://github.com/J-Fairgrieve/Fin-Sentiment-Analysis/blob/main/Images/Article%20Sentiment%20Counts.png?raw=true)

Instantly, we can determine that there may be a problem when it comes to creating models with the data. With just under 5,000 articles in the dataset there may not be enough data to create a reliable model. Also, the dataset is unbalanced. Almost 60% of the dataset is neutral data, so any models created may struggle with identifying positive/negative article titles.

This is probably to be expected, as financial news *should* be impartial and therefore neutral. Some articles reporting profit/loss will have sentiment, but otherwise these are articles reporting factual, numerical information.

Another feature we can analyse is the word count of each article title. The average article is 23 words long (median 21 words). Plotting a histogram can visually represent this:

![Article Word Count Histogram](https://github.com/J-Fairgrieve/Fin-Sentiment-Analysis/blob/main/Images/Article%20Word%20Count%20Length.png?raw=true)

As shown above, the word length has a positive skew. A higher proportion of articles have a shorter word length. There are potential outliers in the neutral data with a handful of articles reaching the 80 word mark! Looking at the averages for each sentiment further explains the similarity of the articles.

![Article Word Count Histogram](https://github.com/J-Fairgrieve/Fin-Sentiment-Analysis/blob/main/Images/Average%20Sentiment%20Word%20Count.png?raw=true)

Not included in this write up: I have also created word clouds in [the code](https://github.com/J-Fairgrieve/Fin-Sentiment-Analysis/blob/main/1%20-%20Financial%20News%20EDA.ipynb) but I have chosen not to include them as they are so similar. This, unfortunately, is another indicator that creating an effective model will be difficult. But are my assumptions correct?

## Multinomial NB Model
###### You can find the full code for this section [here](https://github.com/J-Fairgrieve/Fin-Sentiment-Analysis/blob/main/2%20-%20Multinomial%20Naive%20Bayes%20Model.ipynb).

#### 2.1 Converting Sentiment to Numerical Values
For this model, I will convert the article sentiment column to a numerical value. -1 will be negative, 0 neutral and 1 positive.

~~~
# Assign numerical value depending on Sentiment text
sentiment_num = []

for i in range(len(data)):
    if data["Sentiment"][i] == "neutral":
        sentiment_num.append(0)
    elif data["Sentiment"][i] == "positive":
        sentiment_num.append(1)
    else:
        sentiment_num.append(-1)

# Create a new DataFrame for the model
ModelData = pd.DataFrame()
ModelData["Sentiment"] = sentiment_num
ModelData["Text"] = data["Text"]
ModelData
~~~

#### 2.2 Preparing the Data for Modelling
The article text needs to be converted to a numerical value in order for the model to be able to operate. In this case, I have used TfidVectorizer on a sample of 3,500 articles from the dataset:

~~~
# Collect a sample of the data for X & y
X = ModelData.iloc[:3500,1]
y = ModelData.iloc[:3500,0]

# Run vectorizer so we can convert the text into numerical features
vectorizer = TfidfVectorizer(max_features = 50000 , lowercase=False , ngram_range=(1,2))
~~~

We can now create/transform the training and testing data:

~~~
# Transform the training data
vector_train = vectorizer.fit_transform(train_data)
vector_train = vector_train.toarray()

# Transform the test data
vector_test = vectorizer.transform(test_data).toarray()

# Create the dataframes for the test & train data
training_data = pd.DataFrame(vector_train , columns=vectorizer.get_feature_names_out())
testing_data = pd.DataFrame(vector_test , columns= vectorizer.get_feature_names_out())
~~~

#### 2.3 Multinomial NB Model
Now that we have the test and train data, we can create a model

~~~
# Prepare the Multinomial Naive Bayes model
clf = MultinomialNB()

# Fit the model
clf.fit(training_data.values, train_label.values)
y_pred  = clf.predict(testing_data.values)
~~~

##### F1 Scores for the Test Data
- Negative: 0.04
- Neutral: 0.80
- Positive: 0.36
- Accuracy: 0.66

##### F1 Scores for the Training Data
- Negative: 0.10
- Neutral: 0.87
- Positive: 0.73
- Accuracy: 0.79

The model, as expected, is poor for detecting news articles that aren't neutral. The model is 79% accurate on the training data but only 66% on the testing data. The best results are with the neutral articles (87% on training and 80% on test).

The full classification report, alongside some further single prediction testing, can be viewed in [the code](https://github.com/J-Fairgrieve/Fin-Sentiment-Analysis/blob/main/2%20-%20Multinomial%20Naive%20Bayes%20Model.ipynb).