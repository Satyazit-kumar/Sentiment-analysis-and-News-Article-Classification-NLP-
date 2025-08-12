# **IMDb Movie Review Sentiment Analysis**

#### Dataset : [Imdb](https://docs.google.com/spreadsheets/d/106x15uz8ccQ6Wvpc8-sYjXisBN8vewS435I7z3wd4sw/edit?gid=1889101679#gid=1889101679)

### **Problem Statement**
  
The primary objective of this project is to build a machine learning classification model that
can predict the sentiment of IMDb movie reviews. The dataset contains a collection of movie
reviews, and each review is labelled as either positive or negative.
Using text preprocessing, feature extraction techniques (such as TF-IDF), and various
classification algorithms, the project will aim to develop a model that can effectively classify
the sentiment of movie reviews. The model's performance will be evaluated using standard
classification metrics, such as accuracy, precision, recall, and F1-score.

### **Data Exploration and Preprocessing**
 
The dataset, named imdb_data.csv, comprises a collection of 50,000 movie reviews, each systematically labeled with its corresponding sentiment polarity.
•	Dataset Dimensions: The dataset contains 50000 rows and two columns namely review (Textual Content) and sentiment (Categorical Value)
•	Missing Values: There were no missing values found.
•	Sentiment Distribution: The distribution of sentiment classes shows that there is equal distribution of positive and negative sentiments. Which means it is a balanced data.
•	Review Length Analysis: The mean review length was approximately 1309 characters, with values ranging from a minimum of 7 to a maximum of 13704 characters.

  **Text Preprocessing**
  
  In this phase, I conducted the following preprocessing techniques.
  1.	Lowercasing: All textual content was converted to lowercase to ensure uniformity and prevent variations in capitalisation from being treated as distinct linguistic entities.
  2.	HTML Tag Removal: Removed HTML tags using regular expressions.
  3.	Punctuation Removal: Using re.sub() function from regular expression module to remove noise
  4.	Special Character and Number Removal: Used re.sub() function to remove non-alphabets.
  5.	Tokenization: Reviews were tokenized
  6.	Stop Word Removal: Common English stop words (e.g., "the", "a", "is") were removed.
  7.	Lemmatization: Using WordNetLematizer class, lemmatized all the tokenized words to its dictionary root form.
     
  ### **Feature Engineering**
  converted cleaned text into numerical features and add other textual features. I also combined intrinsic textual features with TF-IDF vectorization.

  #### Textual Features
  •	Word Count: The number of words present within each processed review.
  •	Character Count: The aggregate number of characters, excluding whitespace, contained within each processed review.
  •	Average Word Length: The arithmetic mean of word lengths within each processed review.
  
  #### TF-IDF Vectorization
  
  TF-IDF (Term Frequency-Inverse Document Frequency) It assigns a weighted significance to each term, reflecting its prevalence within a specific document relative to its rarity across the entire corpus.
  •	A TfidfVectorizer instance was initialized with specific parameters: max_features=5000 (constraining the feature set to the top 5000 most frequent terms), min_df=5 (excluding terms appearing in fewer than 5     documents), and max_df=0.8 (disregarding terms present in over 80% of documents, as these are often ubiquitous and less discriminative).
  •	The vectorizer was subsequently fitted and applied to the cleaned_review column, yielding a sparse TF-IDF matrix with dimensions of (50000, 5000).
  •	Ultimately, the TF-IDF features were conjoined horizontally with the word_count, char_count, and avg_word_length features. This concatenation resulted in a comprehensive feature matrix, X_features, with         overall dimensions of (50000, 5003).
  •	The sentiment variable was numerically encoded, with positive reviews assigned a value of 1 and negative reviews assigned a value of 0, serving as the designated target variable, y.
 Model Development and Evaluation
 
 ### Data Splitting
  Dataset was split into training and test set
•	Training Set (80%): Contains 40,000 samples.
•	Testing Set (20%): Consisting of 10,000 samples.

### **Model Training and Evaluation:**
Four machine learning models were trained: Logistic Regression, Multinomial Naive Bayes, Linear Support Vector Machine (SVM), and Random Forest. For each model, a standardized set of performance metrics was computed and reported:
•	Accuracy:  proportion of correctly classified instances relative to the total number of instances.
•	Precision: Measures the proportion of instances predicted as positive that were indeed true positives.
•	Recall: Assesses the proportion of actual positive instances that were correctly identified by the model.
•	F1-Score: Represents the harmonic mean of precision and recall, serving as a balanced indicator of a model's performance, particularly relevant in classification tasks with imbalanced classes 
•	Classification Report:  Breakdown of precision, recall, and F1-score for each individual class ('negative' and 'positive').
•	Confusion Matrix: Visual representation of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
 Performance Comparison
 
### **A comparative summary of the performance metrics across all models is presented in the table below:**

  

   **|Model**           **| Accuracy | Precision | Recall | F1-Score |**
  
|---------------------|----------|-----------|--------|----------|

 **| Logistic Regression | 88.60%   | 87.76%    | 89.92% | 88.83%   |**
 
 **| Naive Bayes         | 82.10%   | 82.46%    | 81.90% | 82.18%   |**
 
 **| Linear SVM          | 88.23%   | 87.50%    | 89.42% | 88.45%   |**
 
 **| Random Forest       | 84.91%   | 85.64%    | 84.16% | 84.90%   |**

    
### **Observations:**

•	Logistic Regression consistently gave superior performance, achieving the highest F1-Score (0.8883) and Accuracy (0.8860). Its precision and recall values for both positive and negative classes were observed to be well-balanced.
•	Linear SVM demonstrated a F1-Score of 0.8845 which is close to what Logistic Regression predicted
•	Random Forest also performed well. It lagged behind the other models by a slight margin.
•	Multinomial Naive Bayes registered the comparatively lowest performance among the evaluated models, with an F1-Score of 0.8218.

### **Conclusion**

Among all the models tested, Logistic Regression performed the best, giving the most accurate and balanced results. Linear SVM also gave strong results. 


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# News Article Classification
 
  #### Dataset: [data_news](https://docs.google.com/spreadsheets/d/1m4YMfqQxo_DcbtzGqbfZitvJmytbWUE8qjixhHmtadk/edit?gid=1552269726#gid=1552269726)

## Problem Statement:
The primary objective of this project is to build a classification model that can automatically
categorize news articles into different predefined categories. The model will be trained using
a labeled dataset of news articles and will output the most likely category (e.g., sports,
politics, or technology) for any given article.

### 1.Data Collection and Exploration
•	Dataset: 50,000 news articles evenly distributed across 10 categories
•	Categories: BUSINESS, ENTERTAINMENT, FOOD & DRINK, PARENTING, POLITICS, SPORTS, STYLE & BEAUTY, TRAVEL, WELLNESS, WORLD NEWS
•	Key Features:
•	Headline
•	Short description
•	Keywords
•	URL links
### 2. Data Preprocessing
1.	Text Combination: Merged headline and short description into single text feature
2.	Text Cleaning:
  •	Converted to lowercase
  •	Removed punctuation
  •	Eliminated stopwords
3.	Feature Extraction: TF-IDF vectorization (5,000 most frequent words)
4.	Train-Test Split: 80-20 split with stratified sampling


### 3. Model Training
Implemented three classification models:
  **1.	Logistic Regression**
  
  **2.	Multinomial Naive Bayes**
  
  **3.	Linear Support Vector Machine (SVM)**
  
  **4. Model Evaluation Results**

### Performance Metrics Summary:

**- Model	Accuracy	Precision (macro avg)	Recall (macro avg)	F1-Score (macro avg)**

**- Logistic Regression	0.7950	0.80	0.80	0.7953**

**- Naive Bayes	0.7813	0.78	0.78	0.7815**

**- Linear SVM	0.7891	0.79	0.79	0.7887**

### Best Performing Categories Across All Models:

•	SPORTS (F1-score: 0.86-0.89)

•	STYLE & BEAUTY (F1-score: 0.84-0.86)

•	FOOD & DRINK (F1-score: 0.83-0.85)
 
 ## Key Findings
  ####  1.	Model Performance:
o	Logistic Regression achieved the highest overall accuracy (79.50%) and F1-score (0.7953)

o	Linear SVM showed slightly better performance than Naive Bayes

o	All models performed consistently well on sports and lifestyle categories

o	Wellness and parenting categories were most challenging across all models

### Conclusion

All the three models performed well but the Logistic Regression emerged as the best-performing model among the three models that were tested. 

