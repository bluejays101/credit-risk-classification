# Credit Risk Classification Report

## 📊 Overview of the Analysis

The purpose of this analysis was to evaluate the credit risk of loan applicants using machine learning. Specifically, I aimed to classify loans as either "healthy" (`0`) or "high-risk" (`1`) using features from a dataset of historical lending decisions.  The dataset, `lending_data.csv`, was located in the **Resources** folder provided with the starter code. It includes financial information about loan applicants, which we used to train and evaluate a classification model for credit risk prediction.

The dataset included financial attributes such as:
- `loan_size`
- `interest_rate`
- `borrower_income`
- `debt_to_income`
- `num_of_accounts`
- `derogatory_marks`
- `total_debt`

The target variable was `loan_status`, where:
- `0` = healthy loan
- `1` = high-risk loan

This followed a standard machine learning workflow:
1. **Preprocessing**: Split the data into features (`X`) and labels (`y`).
2. **Training and Testing Split**: Used an 80/20 train-test split to ensure the model's ability to generalize.
3. **Model Building**: Trained a Logistic Regression model (`sklearn.linear_model.LogisticRegression`) using the training data.
4. **Evaluation**: Made predictions on the test data, then evaluated performance using:
   - Confusion Matrix
   - Accuracy Score
   - Classification Report (Precision, Recall, F1-Score)

---

## ✅ Results

**Logistic Regression Model:**
- **Accuracy Score**: `0.925`
- **Precision**:
  - Class 0 (Healthy Loans): `0.94`
  - Class 1 (High-Risk Loans): `0.84`
- **Recall**:
  - Class 0 (Healthy Loans): `0.97`
  - Class 1 (High-Risk Loans): `0.71`

---

## 🧠 Summary

The Logistic Regression model demonstrates strong performance overall, especially in identifying healthy loans (class `0`) with a high recall rate of `0.97`. However, it is less effective at correctly identifying high-risk loans (class `1`), with a recall of only `0.71`. This implies that nearly 29% of actual high-risk loans were misclassified as healthy, which may pose a risk to lenders.

Despite this shortcoming, the model's:
- High precision for class `1` (0.84),
- Strong overall accuracy (92.5%),
- Simplicity and interpretability,

This makes it a reasonable baseline model. However, **if minimizing risk is critical**, especially avoiding false negatives for high-risk loans, then further model tuning or alternative algorithms (e.g., Random Forest, Gradient Boosting) should be considered to improve recall for class `1`.

**Recommendation**: Use the Logistic Regression model as a starting point, but enhance it with techniques that better capture high-risk loan patterns.
