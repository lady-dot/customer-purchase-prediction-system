# Customer Purchase Prediction System

## 📌 Overview
Developed an end-to-end machine learning system to predict customer purchase behavior using historical data.

## 💼 Business Problem
Businesses often struggle to identify which customers are likely to convert. This leads to inefficient marketing spend and missed revenue opportunities.

## 🎯 Objective
Predict whether a customer will make a purchase to enable targeted marketing strategies.

## ⚙️ System Design
- Data preprocessing pipeline
- Feature engineering
- Model training and evaluation
- Prediction pipeline

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn

## 📊 Model Performance
1. Logistic Regression...
Accuracy: 0.840
Precision: 0.853
Recall: 0.762
F1 Score: 0.805
ROC-AUC: 0.902

2. Decision Tree...
Accuracy: 0.853
Precision: 0.836
Recall: 0.823
F1 Score: 0.829
ROC-AUC: 0.850

3. Random Forest...
Accuracy: 0.933
Precision: 0.944
Recall: 0.900
F1 Score: 0.921
ROC-AUC: 0.941

### Model Comparison Insights
- Random Forest achieved the highest ROC-AUC, indicating strong classification performance.
- Logistic Regression provides a good baseline but may underperform due to linear assumptions.
- Decision Tree shows signs of potential overfitting.

### Business Interpretation
- High recall is critical if the goal is to identify as many potential buyers as possible.
- High precision is important if marketing cost needs to be minimized.

The model choice depends on business priorities:
- Maximize conversions → prioritize recall
- Minimize cost → prioritize precision

## 🚀 Features
- Automated preprocessing pipeline
- Trained ML model for prediction
- Reusable prediction script

## ▶️ How to Run
To run this project, first clone the repository (`git clone https://github.com/your-username/customer-purchase-prediction-system.git`) and navigate into it (`cd customer-purchase-purchase-prediction-system`). # clone repo and enter folder Then, create a virtual environment (`python -m venv venv`) and activate it (`source venv/bin/activate` on Mac/Linux or `venv\Scripts\activate` on Windows). # setup isolated Python environment Install required dependencies using `pip install -r requirements.txt`. # install all Python packages To train the models and generate the baseline Random Forest, run the training script with `python src/train.py`. # train and save the model After training, you can make predictions with `python src/predict.py`, which loads the saved model and outputs predictions on new or test data. # load model and predict Optionally, you can explore data and results interactively in the Jupyter notebook by running `jupyter notebook` and opening `notebooks/exploration.ipynb`. # interactive exploration of analysis and plots.

### 📁 Project Structure

- `src/` → Core pipeline scripts
- `models/` → Saved trained models
- `notebooks/` → Exploration and analysis
- `reports/` → Visualizations and insights

## 📎 Future Improvements
- Deploy as API
- Add real-time prediction
