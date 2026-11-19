# Customer Churn Prediction with an Artificial Neural Network

Predicts whether a bank customer is likely to leave (churn), using an artificial neural network (ANN) built with TensorFlow/Keras. The trained model runs inside a simple **Streamlit** web app: enter a customer's details and you get their churn probability.

![Project pipeline](images/project_pipeline.svg)

## Dataset

`Churn.csv` has 10,000 bank customers with these features: credit score, geography (France / Germany / Spain), gender, age, tenure, balance, number of products, has credit card, is active member, and estimated salary. The target is `Exited` (1 = customer left the bank). About 20% of customers churned.

## What's in this repo

| File | What it does |
|---|---|
| `1_churn_model_training.ipynb` | Preprocessing (label encoding, one-hot encoding, scaling), building and training the ANN with EarlyStopping and TensorBoard |
| `2_churn_prediction.ipynb` | Loads the saved model and predicts churn for one new customer |
| `3_salary_regression.ipynb` | Same ANN setup used for regression, predicting estimated salary |
| `4_hyperparameter_tuning.ipynb` | GridSearchCV with 3-fold cross-validation over neurons, layers and epochs |
| `app.py` | Streamlit web app for live churn prediction |
| `model.h5` | Trained churn classification model |
| `regression_model.h5` | Trained salary regression model |
| `*.pkl` | Saved encoders and scaler, so new inputs are transformed exactly like the training data |
| `requirements.txt` | Python packages needed |

## Model

- **Input:** 12 features (after encoding and scaling)
- **Hidden layers:** Dense(64, ReLU) → Dense(32, ReLU)
- **Output:** Dense(1, Sigmoid), giving the probability of churn
- **Loss:** Binary crossentropy | **Optimizer:** Adam (learning rate 0.01)
- **EarlyStopping** (patience 10, restores best weights) to prevent overfitting
- **Total parameters:** 2,945

## Results

| Experiment | Result |
|---|---|
| Churn classifier (64-32-1) | ~86% accuracy on held-out data (baseline of always predicting "stays" = 80%) |
| Hyperparameter tuning | Best: 1 hidden layer, 16 neurons, 100 epochs, ~85.6% cross-validated accuracy; a smaller network performs as well as the larger one |
| Salary regression | Test MAE ≈ 50,199, no better than predicting the median salary (≈ 49,676). Estimated salary is not related to the other features in this dataset, so it can't be predicted from them |

## How to run

```bash
git clone https://github.com/srivatchan2004/ann-churn-prediction.git
cd ann-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. Choose the customer details and it shows the churn probability. A probability above 0.5 means the customer is likely to churn.

To view training curves, run the training notebook, then:

```bash
tensorboard --logdir log/fit
```

## Tech stack

Python · TensorFlow / Keras · scikit-learn · scikeras · pandas · NumPy · Streamlit · TensorBoard

## Author

**Srivatchan D** · [LinkedIn](https://www.linkedin.com/in/srivatchan2004)
