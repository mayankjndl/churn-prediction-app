# Customer Churn Prediction App 🚀

A full end-to-end Machine Learning web application that predicts **customer churn risk** using a trained Decision Tree model, deployed live with **Streamlit Cloud**.

🔗 Live App: https://mayankjindal-customer-churn-predictor.streamlit.app/

---

## 📌 Overview

Customer churn refers to customers who stop using a product or service.  
This app helps estimate **how likely a customer is to churn** based on key features such as:

- Tenure
- Monthly charges
- Total charges
- Contract type
- Tech support availability
- Internet service type

The goal is to assist businesses in identifying high-risk customers and taking proactive retention actions.

---

## 🧠 Machine Learning Details

- **Problem Type:** Binary Classification  
- **Target Variable:** Churn (Yes / No)  
- **Final Model Used:** Decision Tree Classifier  
- **Reason for selection:** Highest recall among tested models (better at catching churners)

### Models Compared
| Model | Used |
|------|------|
| Logistic Regression | ❌ |
| K-Nearest Neighbors | ❌ |
| Decision Tree | ✅ Final |

---

## 🧪 Model Performance (on test set)

- Accuracy: ~80%  
- Recall (Churn class): ~63%  
- F1-score: ~0.63  

Focus was placed on **recall**, as missing churn customers is costlier than false positives.

---

## 🛠 Tech Stack

- Python 3.12  
- scikit-learn 1.6.1  
- pandas  
- numpy  
- joblib  
- Streamlit  

---

## 📂 Project Structure
```
churn-prediction-app/
│
├── app.py # Streamlit web app
├── model.pkl # Trained Decision Tree model
├── scaler.pkl # StandardScaler
├── imputer.pkl # SimpleImputer
├── requirements.txt # Dependency versions
└── README.md
```
---

## 🌐 Deployment

The app is deployed on **Streamlit Cloud** and runs in a controlled environment.

Key challenges solved during deployment:
- Python version mismatch
- Pickle incompatibility
- NumPy / scikit-learn conflicts
- Cloud runtime differences

Final solution uses:
Python 3.12
scikit-learn 1.6.1
numpy 2.0.2

---

## ▶️ Run Locally
```
git clone https://github.com/mayankjndl/churn-prediction-app
cd churn-prediction-app

pip install -r requirements.txt
streamlit run app.py
Then open: http://localhost:8501
```
## 📈 Example Output

The app returns:

- Churn probability (percentage)
- Risk category:
  - Low risk
  - High risk
- Example:
  - Low Churn Risk (16.49%)

## 🔐 Model Persistence Note

Models are stored using joblib.
Pickle files must always be loaded using the same library versions they were trained with.

## 👤 Author

Mayank Jindal

This project demonstrates:

- End-to-end ML workflow
- Feature engineering
- Model evaluation
- Real-world deployment
- Debugging production environments
