# Crime Prediction Tool — Streamlit UI with Trained ML Models

This project applies machine learning to predict the likelihood of homicides using structured data. It focuses on addressing the class imbalance issue often seen in crime data by comparing different models.

We provide a Streamlit-based web application that allows users to run predictions using three trained models:
- Random Forest (`best_model_rf.sav`)
- Neural Network (`best_model_nn.sav`)
- NEXT Model (`best_model_next.sav`), which combines improved Neural Network and XGBoost components

## Abstract

Machine learning models often face difficulty in detecting rare events such as homicides due to class imbalance in the dataset. This study compares three models: a basic Neural Network, a Random Forest, and the proposed NEXT model, which incorporates imbalance correction and combines predictions from Neural Network and XGBoost. The NEXT model demonstrated better detection of minority cases while maintaining comparable accuracy to the other models.

## Project Structure

CaseStudy/
├── Notebook_Pred2Town_XAI_v1.1.ipynb # Jupyter Notebook for training and evaluation
├── dataset_pred2town_bel.csv # Original dataset
├── X_train.csv, X_test.csv, y_test.csv # Processed data files
├── best_model_rf.sav # Trained Random Forest model
├── best_model_nn.sav # Trained Neural Network model
├── best_xgb_next.sav # Trained XGBoost component of NEXT model
├── best_nn_next.sav # Trained improved Neural Network component
├── best_model_next.sav # Final combined NEXT model
└── UI/
└── app.py # Streamlit user interface


## How to Run

1. Install the required packages:

 `requirements.txt` file:

pip install -r requirements.txt



2. Navigate to the UI folder:

cd /Users/heinzarni/CaseStudy/UI



3. Run the Streamlit application:

streamlit run app.py



## What the App Does

- Allows users to input features for prediction
- Offers three model options for comparison
- Demonstrates how imbalance correction improves minority class prediction
- Visualizes results from each model

## Notes

- All models are pre-trained and stored as `.sav` files
- The notebook (`Notebook_Pred2Town_XAI_v1.1.ipynb`) documents the training and evaluation process
- Dataset is based on real crime records with emphasis on homicide prediction

## License

This project is for academic and research use only.