# HW4: Document Categorization on Persika dataset
This project focuses on document classification using the Persika dataset, which is a Persian language dataset. The classification is based on term frequency concepts such as count vectorizer, latent semantic analysis, and TF-IDF. Various classifiers including gradient boosting, SVM, naive Bayes, KNN, and XGBoost are employed for this purpose.


## Methodology

1. Term Frequency Approaches:
    * Count Vectorizer: Basic term frequency method is employed.
    * Latent Semantic Analysis (LSA): Dimensionality reduction technique based on term frequency-inverse document frequency (TF-IDF).
    * TF-IDF: Term Frequency-Inverse Document Frequency method is used to weigh the importance of words in documents.
2. Classifier Algorithms:
    * Gradient Boosting
    * Support Vector Machine (SVM)
    * Naive Bayes
    * K-Nearest Neighbors (KNN)
    * XGBoost

## Optimization Techniques

  * Stopword Elimination: Stopwords are removed from the documents to improve classification accuracy.
  * Mid-Document Frequency (Mid_DF): A high value for mid-document frequency is utilized to enhance the results.
  * TF-IDF for Numerical Representation: TF-IDF is employed to convert text data into numerical values, leading to better classification results compared to other approaches.
