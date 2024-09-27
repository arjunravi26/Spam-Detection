## End-to-End Spam Detection Using Word2Vec and SVC

This project presents an end-to-end solution for message spam detection, leveraging Google's pretrained average Word2Vec model for vectorization and a Support Vector Classifier (SVC). The model is designed to achieve high accuracy in identifying spam messages. The key components of the project include:

- **Data Preprocessing**: Comprehensive text cleaning, tokenization, and handling of missing values to ensure data quality.
- **Feature Extraction**: Utilization of the average Word2Vec model for effective vectorization of text data.
- **Model Training**: Implementation of the SVC, trained on a labeled dataset with a defined train-test split ratio and extensive hyperparameter tuning to optimize performance.
- **Evaluation**: Rigorous assessment of the model's performance using metrics such as accuracy, precision, recall, and F1-score, along with cross-validation to ensure the model is not overfitting.
- **Accuracy Improvement**: The initial model achieved an accuracy of 70%. Through the integration of the pretrained Word2Vec model and extensive algorithm and hyperparameter tuning, the accuracy was enhanced by 10%.
- **Deployment**: Seamless integration with a Flask web application and deployment on AWS Elastic Beanstalk, ensuring scalable and reliable access to the model.

All these processes are encapsulated within a Scikit-learn pipeline, facilitating streamlined processing and reproducibility.
