# Sentiment-Analysis-using-NLP

The primary goal was to assess the effectiveness of these models in discerning harsh and hateful language within the OLID dataset (Zampieri et al., 2019), a repository containing diverse instances of offensive language.

## To Run the .ipynb file:
1. Upload the the file to yout colab
2. Run All the cells

**Read the comments to understand each line of the code.

### Implemented Method :

METHOD 1:
* preprocesses/clean it using the nltk library and using TF-IDF model to “vectorize” the text into a vector
* trains an ensemble model (bagging and boosting) with SVM as the base model on the preprocessed training data and saves the best model to disk.

METHOD 2:
* preprocesses/clean it using the nltk library and using TF-IDF model to “vectorize” the text into a vector
* Trains a Convolutional Neural Network (CNN) model on the given training dataset, validates it on the given validation dataset, saves the trained model

METHOD 3:
* Trains a Deep-learning model with pre-trained BERT model on the given training dataset, validates it on the given validation
   dataset, saves the trained model, and prints the classification report.
   The model consists of:

    Input layer: to tell the model which input format to expect, so that the model knows what to expect
    Distil Bert model: to encode the input data into a new sequence of vectors (that is the output of BERT). Only the first vector of this sequence will be used as an input for the rest of the classifier
    Dropout layer: for regularization
    Dense layer (with relu activation function, with 64 neurons): to solve the specific problem of classification
    Dense layer (with softmax activation function): for a probability distribution for each label

    Ref: Claude Feldges (2022) Text Classification with TF-IDF, LSTM, BERT: a comparison of performance [Source code].
	https://medium.com/@claude.feldges/text-classification-with-tf-idf-lstm-bert-a-quantitative-comparison-b8409b556cb3



### Notes :

* the Dataset is not a balanced dataset. the ratio of NOT:OFF class is 2:1. Which mean that offenssive class has less texts.
* We are asked not to keep the distribution of the class as it is and asked not balance.
* When we are training a model on unbalanced dataset, we observe that Accuracy is high and F1, precission, recall is lower.
* As the model is fed with sufficient amount of samples in the 'NOT' class, it is able to identify 100% of the text which are Not-offenssive accuratly.
* It is observed that TFID vectorizer with machine learning Eensemble model outperformes all the other moethods implemented.
