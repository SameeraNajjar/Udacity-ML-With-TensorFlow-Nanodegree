# Finding Donors for CharityML

## Overview
This project leverages supervised learning techniques to predict which individuals in the Census dataset earn more than $50,000 per year. This prediction aids CharityML, a non-profit organization, in identifying potential donors. The project includes data exploration, preprocessing, model evaluation, optimization, and feature importance analysis.

## Project Structure

1. **Data Exploration**
   - Calculate the total number of records.
   - Determine the number of individuals with income >$50,000 and <=$50,000.
   - Calculate the percentage of individuals with income >$50,000.

2. **Data Preprocessing**
   - Apply one-hot encoding to categorical features.
   - Convert income data into binary labels.
   - Split the data into training and testing sets.

3. **Naive Predictor Benchmark**
   - Establish a baseline using a naive predictor (assumes all individuals earn >$50,000).
   - Calculate accuracy and F1 score to compare with actual models.

4. **Model Selection and Evaluation**
   - Implement and evaluate the following supervised learning models:
     - Decision Tree Classifier
     - Random Forest Classifier
     - AdaBoost Classifier
   - Assess each model's pros and cons based on accuracy, F1 score, and training time.

5. **Model Optimization**
   - Select the best-performing model based on accuracy and F1 score.
   - Use grid search to optimize hyperparameters.

6. **Feature Importance**
   - Extract feature importances from the model.
   - Rank the top five most important features for predicting income.
   - Compare model performance using only the top five features vs. the entire feature set.

7. **Final Model Evaluation**
   - Report accuracy and F1 score for both unoptimized and optimized models.
   - Compare final model performance with the naive predictor and earlier models.

## Installation

Ensure the following libraries are installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `jupyter notebook`

To install dependencies, run:
```bash
      pip install pandas numpy scikit-learn matplotlib seaborn 

## Running the Project

1. **Data Preparation**
   - Load the census dataset.
   - Perform one-hot encoding and data preprocessing.
   - Split the dataset into training and testing sets.

2. **Model Training and Evaluation**
   - Train the supervised learning models and evaluate their performance.
   - Compare results using accuracy, precision, recall, and F1 score.

3. **Model Optimization**
   - Perform grid search optimization on the chosen model.
   - Evaluate the performance of the optimized model.

4. **Feature Importance**
   - Analyze and compare the importance of features used by the model.
   - Experiment with reducing the feature set and assess the impact on model performance.

## Results

- The final optimized model demonstrates significantly improved performance compared to the naive predictor.
- Feature importance analysis shows which variables have the greatest impact on income prediction.

## Conclusion
This project demonstrates the application of supervised learning techniques to a real-world classification problem. By selecting and optimizing models, the project achieves improved predictive accuracy, making it a valuable tool for identifying potential donors.
