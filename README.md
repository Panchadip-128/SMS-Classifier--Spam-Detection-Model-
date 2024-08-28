# SMS-Classifier(-Spam-Detection-Model) : [Natural Language Processing]
The SMS Spam Detector project utilizes machine learning to create an effective model for detecting spam messages. It begins with preprocessing the SMS dataset, including handling encoding, removing duplicates, and text preprocessing. Exploratory data analysis provides insights into the dataset's characteristics.
Modeling involves comparing various algorithms like Naive Bayes, Logistic Regression, and SVM. The Voting Classifier, combining SVM, Multinomial Naive Bayes, and Extra Trees Classifier, emerged as the top performer, achieving an accuracy of 98.16% and precision of 98.17%.

![spam_message_model_demo](https://github.com/Panchadip-128/SMS-Classifier--Spam-Detection-Model-/assets/165953910/e9fa341e-b41d-4908-bfd3-d12a724cf7ed)


![ham_message_model_demo](https://github.com/Panchadip-128/SMS-Classifier--Spam-Detection-Model-/assets/165953910/76acd083-7023-4947-a7ab-b285c24165cb)



**SMS Spam Detector**:
------------------

Overview
This project aims to create a robust SMS spam detector using machine learning. Various classification algorithms were compared to identify the best-performing model for spam detection. The final model is then used to classify incoming SMS messages as either 'spam' or 'ham' (not spam).
Table of Contents
•	Overview
•	Features
•	Dataset
•	Data Cleaning and Pre-Processing
•	Exploratory Data Analysis (EDA)
•	Modeling
•	Evaluation
•	Best Performing Model
•	Installation
•	Usage
•	Results
•	Contributing
•	License

Features
•	Preprocessing of text data
•	Comparison of various machine learning algorithms
•	Evaluation metrics to compare models
•	Selection of the best-performing model
•	Deployment-ready code for SMS spam detection

Dataset
The dataset used in this project is the SMS Spam Collection Dataset, which consists of a collection of SMS messages labeled as 'spam' or 'ham'.
Data Cleaning and Pre-Processing
1.	Loading Data: The dataset is loaded and inspected for structure and content.
2.	Handling Encoding: Ensuring proper text encoding using the chardet library.
3.	Removing Unnecessary Columns: Dropping columns that are not relevant for analysis.
4.	Renaming Columns: Renaming columns for better readability.
5.	Label Encoding: Converting categorical labels ('ham' and 'spam') into numerical values.
6.	Removing Duplicates: Identifying and removing duplicate entries.
7.	Text Preprocessing:
•	Lowercasing text
•	Tokenization
•	Removing non-alphanumeric characters
•	Removing stop words and punctuation
•	Stemming words
Exploratory Data Analysis (EDA)
1.	Basic Statistics: Calculating basic statistics for text length, number of words, and number of sentences.
2.	Visualizations:
•	Pie chart showing the distribution of 'ham' and 'spam'
•	Histograms for the number of characters and words in messages
•	Pair plot for visualizing relationships between features
•	Correlation heatmap
Modeling
Various machine learning algorithms were compared, including:
•	Naive Bayes (Gaussian, Multinomial, Bernoulli)
•	Logistic Regression
•	Support Vector Machine (SVM)
•	Decision Tree
•	Random Forest
•	K-Nearest Neighbors (KNN)
•	Gradient Boosting
•	AdaBoost
•	Bagging Classifier
•	Extra Trees Classifier
•	XGBoost
A Voting Classifier and a Stacking Classifier were also employed for combining the strengths of multiple models.
Evaluation
Each model was evaluated using the following metrics:
•	Accuracy
•	Precision
•	Confusion Matrix
Best Performing Model
The Voting Classifier with SVM, Multinomial Naive Bayes, and Extra Trees Classifier achieved the highest performance with:
•	Accuracy: 98.16%
•	Precision: 99.17%


Installation
To run this project, we need to have Python installed on our machine and need to install the required packages using pip:

pip install -r requirements.txt 

Usage
To use the SMS spam detector, follow these steps:
1.	Clone the repository:
	git clone https://github.com/Panchadip-128/SMS-Classifier--Spam-Detection-Model-.git

2.      Install the required packages:
        pip install -r requirements.txt

3.     Run the main script to train the model and make predictions:
       python main.py

4.     To classify a new SMS message, use the following command:
       python classify.py "Your SMS message here"

Results
The best performing model, the Voting Classifier, achieved the following performance metrics on the test dataset:
•	Accuracy: 98.16%
•	Precision: 99.17%

Contributing
Contributions are welcome! If you have any suggestions or feel any scope of improvements, please create a pull request or open an issue. Thank You!



