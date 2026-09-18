# Student Performance Prediction

An end-to-end machine learning project for predicting student academic performance from socio-academic factors. The project covers data validation, exploratory data analysis, outlier treatment, feature preparation, regression modelling, and model evaluation.

## Project Objective

The objective is to build a reliable regression pipeline that can estimate a student's performance score from the available input features and compare the behaviour of regularized and non-regularized linear models.

## Project Workflow

1. Loaded and inspected the student-performance dataset.
2. Checked the data types, missing values, duplicate records, and descriptive statistics.
3. Explored feature distributions and relationships with the target variable.
4. Identified and handled outliers where appropriate.
5. Prepared the predictor variables and target variable for modelling.
6. Split the data into training and testing sets.
7. Trained Linear, Ridge, and Lasso regression models.
8. Evaluated and compared the models using R-squared and RMSE.

## Models Used

| Model | Purpose |
|---|---|
| Linear Regression | Established an interpretable baseline relationship between the features and student performance. |
| Ridge Regression | Applied L2 regularization to reduce sensitivity to correlated predictors and limit overfitting. |
| Lasso Regression | Applied L1 regularization to shrink less-informative feature coefficients. |

## Evaluation Metrics

- **R-squared (`R²`)** measures the proportion of variation in student performance explained by the model. A higher value indicates a better fit.
- **Root Mean Squared Error (`RMSE`)** measures the typical prediction error in the target variable's units. A lower value indicates better predictive performance.

The exact model scores and comparison are available in the project notebook.

## Technologies Used

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook

## Repository Structure

```text
Student-Performance-Prediction/
├── Student_Performance_Prediction.ipynb
├── README.md
└── data/                              # Dataset, if stored separately
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/vijyush/Student_Performance_Prediction.git
cd Student_Performance_Prediction
```

If your repository has a different name, replace the URL and folder name accordingly.

### 2. Install the required packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Launch the notebook

```bash
jupyter notebook Student_Performance_Prediction.ipynb
```

Run the notebook cells sequentially to reproduce the data preparation, visualizations, model training, and evaluation.

## Key Skills Demonstrated

- Data cleaning and validation
- Exploratory data analysis
- Outlier identification and treatment
- Feature preparation
- Train-test splitting
- Regression modelling
- Regularization with Ridge and Lasso
- Model comparison using `R²` and RMSE
- Interpretation and communication of analytical results

## Possible Improvements

- Use cross-validation for more robust model comparison.
- Tune the Ridge and Lasso regularization parameters.
- Add residual analysis and prediction-error visualizations.
- Compare linear models with tree-based regression methods.
- Package the preprocessing and model steps in a scikit-learn `Pipeline`.

## Author

**Burra Vijyusha**  
B.Tech, Metallurgical Engineering and Materials Science  
Indian Institute of Technology Indore

