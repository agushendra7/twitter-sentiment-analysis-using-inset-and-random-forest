# Twitter Sentiment Analysis Using InSet Indonesia Sentiment Lexicon and Random Forest Classifier

I conducted a comparison of word weighting using Count Vectorizer and TF-IDF, in classifying sentiment on tweets of motorcycle racing events at Mandalika Circuit. The dataset was taken from Twitter of tweets from February 04, 2022, to March 27, 2022. Then the dataset is cleaned to get a clean dataset that will be used in the next stage, using text preprocessing techniques. After that, sentiment labeling will be carried out, using `Inset Lexicon`. The last stage is data modeling. Data modeling is done using the `Random Forest Classifier` method to get the final result in the form of a classification model and the results of the comparison between `Count Vectorizer and TF-IDF` word weighting. 

## Requirements
There are some general library requirements for the project and some which are specific to individual methods. The general requirements are as follows.

- `NumPy` [https://numpy.org] - Fast and versatile, the NumPy vectorization, indexing, and broadcasting concepts are the de-facto standards of array computing today.
- `Pandas` [https://pandas.pydata.org] - Fast, powerful, flexible, and easy-to-use open-source data analysis and manipulation tool.
- `NLTK` [https://www.nltk.org] - Standard Python library with prebuilt functions and utilities for ease of use and implementation.
- `Scikit-learn` [https://scikit-learn.org] - Simple and efficient tools for predictive data analysis.

The library requirements specific to some methods are:
- `Swifter` [https://pypi.org/project/swifter] - A package which efficiently applies any function to a pandas dataframe or series in the fastest available manner.
- `Seaborn` [https://seaborn.pydata.org] - It provides a high-level interface for drawing attractive statistical graphics.
- `Matplotlib` [https://matplotlib.org] - Comprehensive library for creating static, animated, and interactive visualizations.

## Stage Explanation

### Preprocessing
Text pre-processing is carried out to ensure good data quality before being used during data analysis, and is able to change the data to be more structured. The stages carried out in text pre-processing include `Case Folding, Tokenizing, Normalization, Stemming, and Filtering`.
1. `Case Folding` => Case folding is a process where the program will read the text per row in the `Text` column, if there are characters that are uppercase or capital letters, they will be converted to lowercase or lowercase letters.
2. `Tokenizing` => In the tokenizing process, the text is separated into pieces called tokens, which are then analyzed. `Words`, `numbers`, `symbols`, `punctuation marks`, and other important entities can be considered as tokens. In NLP, tokens are defined as "words" although tokenizing can also be done in paragraphs or sentences. The tokenizing process also removes `symbols`, `numbers`, `spaces`, and `links`.
3. `Normalization` => The normalization process will change words that are not standard into standard words and also correct words that are typos, abbreviations, or "slang" by matching terms or words that come from the previous tokenizing process with the normalization dictionary. If the word is in the normalization dictionary, then the word will be changed to the correct word or the word that should be. In this process, I added a normalization dictionary in Excel (.xlsx), with the name `kamuskatabaku.xlsx`. The normalization dictionary is obtained from the source https://github.com/teguhary/Automated-labelling-Inset-Lexicon/blob/master/Data/kamus_kata_alay.xlsx.
4. `Stemming` => Stemming is the process of returning the root word of a word that has successfully gone through the normalization process. At this stage, a word will be converted into a base word until the word that has a prefix and suffix will be removed according to the base word. At this stage, I also use the `swifter` library to speed up the stemming process on the dataframe by running tasks in parallel. The processing speed can be twice or even faster if using `swifter`.
5. `Filtering` => Filtering is the process of removing words that have no meaning such as conjunctions. At this stage, I used English stopwords obtained from the NLTK library for filtering the data frame. I added the stopword list in the form of .txt with the name `stopwordbahasa.txt`.

### Labeling
Data labeling uses the `Inset Lexicon method`. I determined the sentiment labels by using the sentiment lexicon https://github.com/fajri91/InSet, which consists of `3609` positive sentiment words and `6609` negative sentiment words. Inset has a weighted value or polarity score on each word with a weight range between -5 to +5. In the implementation in Python, I entered the words contained in the inset dictionary manually, in .xlsx format. There are three sentiment labels that the author determines in this study, namely positive, negative, and neutral. Polarity Score above `(>) 0 is positive`, below `(<) 0 is negative`, and Polarity Score `0 is neutral`.
 
### Modeling
At this stage, there are four stages including `Data Division`, `Word Weighting`, `Data Classification`, and `Model Visualization`.

1. `Data Division` => The labeled tweet data will be divided into `training` data and `testing` data. The training data is used by the classification algorithm to form a classifier model, while the testing data is used to measure the extent to which the classifier succeeds in classifying correctly. The distribution of sentiment data is done randomly, with a ratio of `80%` for training data and `20%` for testing data.
2. `Word Weighting` => Datasets that have been divided into training data and testing data, then the training data will be weighted using the `Count Vectorizer and TF-IDF` methods.
3. `Data Classification` => The dataset will be classified into three categories positive, negative, and neutral. In this research, the data classification stage uses the `Random Forest` algorithm, `Random Forest` is a classifier consisting of a collection of structured tree classifiers where each tree throws voting units for the most popular class in input x. In other words, `Random Forest` consists of a set of decision trees, where the collection of decision trees is used to classify data into a class.
4. `Model Visualization` => In this evaluation, there are three categories or classes in the classification model so that the resulting `Confusion Matrix` has an ordo of 3x3, where the matrix table consists of actual and predicted data, through the `Confusion Matrix`, the average value of `accuracy, precision, and recall` is obtained. In model visualization using the library `Seaborn` and `Matplotlib`.

## Information about other files

* `dataset/tweet.xlsx`: The dataset used.
* `dataset/kamus_positive.xlsx`: List of positive words.
* `dataset/kamus_positive.xlsx`: List of negative words.
