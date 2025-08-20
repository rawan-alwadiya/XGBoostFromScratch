# **XGBoostFromScratch: XGBoost from Scratch on Make Moons Dataset**

**XGBoostFromScratch** is a machine learning project focused on implementing, evaluating, and deploying the **XGBoost algorithm entirely from scratch** without relying on scikit-learn’s built-in models.  
It demonstrates how gradient boosting fundamentals can be implemented from first principles, including **data exploration, model implementation, evaluation with custom-built metrics, and deployment with Streamlit**.

---

## **Demo**

- 🎥 [View LinkedIn Demo Post](https://www.linkedin.com/posts/rawan-alwadeya-17948a305_xgboostfromscratch-building-xgboost-from-activity-7363757736313966592-0Rsa?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE3YzG0BAZw48kimDDr_guvq8zXgSjDgk_I)  
- 🌐 [Try the App Live on Streamlit](https://xgboostfromscratch-fqmnjkfdavi4rdgybk6yx6.streamlit.app/)

![App Demo](https://github.com/rawan-alwadiya/XGBoostFromScratch/blob/main/Happy%20Moon.png)

---

## **Project Overview**

The workflow includes:  
- **Data exploration and visualization**  
- **Custom implementation of XGBoost (from scratch)**  
- **Evaluation using custom-built metrics**  
- **Deployment of the model via a Streamlit web application**

---

## **Objective**

Develop and deploy a gradient boosting model built from first principles to classify samples in the **Make Moons dataset**, showcasing **mastery of algorithmic foundations, end-to-end pipeline implementation, and interactive deployment**.

---

## **Dataset**

- **Source**: [Make Moons Dataset (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)  
- **Samples**: 3000  
- **Features**: 2 numerical features  
- **Target**: Binary classification (two interleaving moon-shaped classes)  

For the **Streamlit app**, feature names and target classes were **renamed for fun**:  
- **X1 → Moon Position (horizontal position in the sky 🌌)**  
- **X2 → Moon Glow (brightness or intensity ✨)**  
- **Target → Happy 🌝 vs Sad 🌚 moon**

---

## **Project Workflow**

- **Exploration & Visualization (EDA)**: Verified the moon-shaped non-linear separability  
- **Modeling**: Implemented **XGBoost from scratch**, covering:  
  - Gradient boosting framework  
  - Tree-based weak learners  
  - Regularization
- **Evaluation Metrics (from scratch)**:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Confusion Matrix  
  - Classification Report  
- **Deployment**: Interactive **Streamlit app** where users can move sliders for moon features and get real-time predictions.

---

## **Performance Results**

**XGBoost (from scratch)** achieved:  
- **Accuracy**: 0.92  
- **Precision**: 0.93  
- **Recall**: 0.92  
- **F1-score**: 0.92  

---

## **Project Links**

- **GitHub Repository**: [XGBoostFromScratch](https://github.com/rawan-alwadiya/XGBoostFromScratch)  
- **Kaggle Notebook**: [View on Kaggle](https://www.kaggle.com/code/rawanalwadeya/xgboostfromscratch-xgboost-from-scratch?scriptVersionId=256970091)  
- **Live Streamlit App**: [Try it Now](https://xgboostfromscratch-fqmnjkfdavi4rdgybk6yx6.streamlit.app/)  
- **LinkedIn Demo Post**: [Watch Here](https://www.linkedin.com/posts/rawan-alwadeya-17948a305_xgboostfromscratch-building-xgboost-from-activity-7363757736313966592-0Rsa?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE3YzG0BAZw48kimDDr_guvq8zXgSjDgk_I)

---

## **Tech Stack**

**Languages & Libraries**:  
- Python, Pandas, NumPy  
- Matplotlib, Seaborn  
- Streamlit (Deployment)  

**Techniques**:  
- Gradient Boosting from scratch  
- Decision Tree learners from scratch  
- First & second-order gradient updates  
- Custom Evaluation Metrics  
- Streamlit Deployment  
