# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.metrics import d2_tweedie_score
from scipy.linalg import pinv
import autograd.numpy as np
from autograd import grad
import autograd.numpy as anp 

# %%
# Load the dataset
df = pd.read_csv('data/hour.csv')

# Display summary statistics to get an overview of the data
print("\nBasic statistics of the dataset:")
print(df.describe())

# Display the data types of each column
print("\nData types of each column:")
print(df.dtypes)

# Display the columns of the dataset to understand the features
print("\nColumns in the dataset:")
print(df.columns)


# %%
# Display the first few rows of the dataset to inspect the structure
print("First 5 rows of the dataset:")
print(df.head())

# Display summary statistics to get an overview of the data
print("\nBasic statistics of the dataset:")
print(df.describe())

# Checking for any missing values in the dataset
print("\nMissing values check:")
print(df.isnull().sum())

# Display the columns of the dataset to understand the features
print("\nColumns in the dataset:")
print(df.columns)

# Describe and summarize the data in terms of number of data points, dimensions, and used data types
print("\nNumber of data points (rows):", df.shape[0])
print("Number of dimensions (columns):", df.shape[1])

# Display the data types of each column
print("\nData types of each column:")
print(df.dtypes)


# %% [markdown]
# Q3.1c

# %%
# Define the target variable
target = 'cnt'  # This is the column representing the number of bikes used per hour.

# Define all columns except the target for plotting
features = df.columns.drop(['cnt'])

# Create a grid of plots for each feature against the target
plt.figure(figsize=(20, 15))  # Adjust the figure size to accommodate all subplots

# Iterate through each feature to create a subplot
for i, feature in enumerate(features, 1):
    plt.subplot(4, 4, i)  # Create a 4x4 grid of subplots
    plt.scatter(df[feature], df[target], alpha=0.5)  # Use scatter plot for numerical features
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f'{target} vs {feature}')

# Adjust the layout to prevent overlapping
plt.tight_layout()
plt.show()


# %% [markdown]
# Q3.1d

# %%


# Load the dataset
df = pd.read_csv('data/hour.csv')

# Include all columns but filter out non-numeric ones for correlation calculation
numeric_df = df.select_dtypes(include=[np.number])

# Compute the correlation matrix for numeric columns only
corr_matrix = numeric_df.corr()

# Plot the correlation matrix as a heatmap with correlation values inside the boxes
plt.figure(figsize=(12, 10))
heatmap = plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(heatmap)
plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
plt.yticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Correlation Matrix of Bike Sharing Data')

# Display correlation values inside the boxes
for (i, j), val in np.ndenumerate(corr_matrix):
    plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)

plt.tight_layout()
plt.show()

# Identify the most positively, negatively, and least correlated features with 'cnt'
correlations_with_cnt = corr_matrix['cnt'].sort_values()
print("\nFeatures sorted by correlation with 'cnt':")
print(correlations_with_cnt)

most_positive = correlations_with_cnt.idxmax()
most_negative = correlations_with_cnt.idxmin()
least_correlated = correlations_with_cnt.abs().idxmin()

print(f"\nMost positively correlated with 'cnt': {most_positive} (correlation: {correlations_with_cnt[most_positive]})")
print(f"Most negatively correlated with 'cnt': {most_negative} (correlation: {correlations_with_cnt[most_negative]})")
print(f"Least correlated with 'cnt': {least_correlated} (correlation: {correlations_with_cnt[least_correlated]})")


# %% [markdown]
# Q3.1e
# 

# %%

# Load the dataset
df = pd.read_csv('data/hour.csv')

# Drop the specified columns
columns_to_drop = ['instant', 'atemp', 'registered', 'casual', 'dteday']
df = df.drop(columns=columns_to_drop, axis=1)

# Display the updated DataFrame structure to verify the changes
print("Updated DataFrame after dropping specified columns:")
print(df.head())
print("\nRemaining columns:")
print(df.columns)


# %% [markdown]
# Q3.1 f
# 

# %%

# Load the dataset
df = pd.read_csv('data/hour.csv')

# Drop the specified columns from part (e)
columns_to_drop = ['instant', 'atemp', 'registered', 'casual', 'dteday']
df = df.drop(columns=columns_to_drop, axis=1)

# Shuffle the DataFrame with a fixed random state for reproducibility
df = shuffle(df, random_state=0)

# Split the shuffled data into training and test sets
train_df = df[:10000]  # First 10,000 rows as the training set
test_df = df[10000:]   # Remaining rows as the test set

# Display the sizes of the resulting sets to verify the split
print(f"Training set size: {train_df.shape[0]} rows")
print(f"Test set size: {test_df.shape[0]} rows")


# %% [markdown]
# Q3.2a

# %%


# Prepare input features (X) and target variable (y)
X_train = train_df.drop(columns=['cnt']).values  # Drop the target variable 'cnt'
y_train = train_df['cnt'].values  # Target variable

# Add a column of ones to X for the intercept term (bias)
X_train = np.c_[np.ones(X_train.shape[0]), X_train]

# Closed-form solution of OLS: beta = (X^T * X)^-1 * X^T * y
beta = np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ y_train)

# Display the coefficients
print("OLS coefficients (beta):")
print(beta)


# %% [markdown]
# Q3.2b

# %%


# Make predictions on the training data
y_train_pred = X_train @ beta

# Calculate the R^2 score using sklearn
r2_train = r2_score(y_train, y_train_pred)

# print(f"R^2 score for the training set: {r2_train}")

def ols_regression(X, y):
    # Closed-form solution (X^T * X)^-1 * X^T * y
    return np.linalg.inv(X.T @ X) @ (X.T @ y)

# Preparing data

X_test = test_df.drop('cnt', axis=1).values
y_test = test_df['cnt'].values

# Add intercept term (bias) to X_train and X_test

X_test = np.c_[np.ones(X_test.shape[0]), X_test]


y_pred_test = X_test @ beta


print(f"R^2 score for the testing set is", r2_score(y_test, y_pred_test))


# %% [markdown]
# Q3.2c

# %%
# One-hot encode the categorical columns in the training set
train_df_encoded = pd.get_dummies(train_df, columns=['season', 'mnth', 'hr', 'weekday', 'weathersit'])

# Ensure the test set has the same columns as the training set after encoding
test_df_encoded = pd.get_dummies(test_df, columns=['season', 'mnth', 'hr', 'weekday', 'weathersit'])
test_df_encoded = test_df_encoded.reindex(columns=train_df_encoded.columns, fill_value=0)

# Display the shapes to verify consistency
print(f"Encoded training set shape: {train_df_encoded.shape}")
print(f"Encoded test set shape: {test_df_encoded.shape}")


# %% [markdown]
# Q3.2d

# %%
import numpy as np
import pandas as pd
from scipy.linalg import pinv
from sklearn.metrics import r2_score

def ols_regression(X, y):
    # Use the pseudo-inverse instead of the inverse for stability
    return pinv(X.T @ X) @ (X.T @ y)

# One-hot encode the categorical columns in the training set
train_df_encoded = pd.get_dummies(train_df, columns=['season', 'mnth', 'hr', 'weekday', 'weathersit'])

# One-hot encode the categorical columns in the test set and ensure the same columns as the training set
test_df_encoded = pd.get_dummies(test_df, columns=['season', 'mnth', 'hr', 'weekday', 'weathersit'])
test_df_encoded = test_df_encoded.reindex(columns=train_df_encoded.columns, fill_value=0)

# Preparing the training and test data for regression
X_train_encoded = train_df_encoded.drop('cnt', axis=1).values
y_train_encoded = train_df_encoded['cnt'].values
X_test_encoded = test_df_encoded.drop('cnt', axis=1).values
y_test_encoded = test_df_encoded['cnt'].values

# Add intercept term (bias) to X_train_encoded and X_test_encoded using np.concatenate
X_train_encoded = np.concatenate([np.ones((X_train_encoded.shape[0], 1)), X_train_encoded], axis=1)
X_test_encoded = np.concatenate([np.ones((X_test_encoded.shape[0], 1)), X_test_encoded], axis=1)

# Ensure that all values are float64
X_train_encoded = X_train_encoded.astype(float)
y_train_encoded = y_train_encoded.astype(float)
X_test_encoded = X_test_encoded.astype(float)
y_test_encoded = y_test_encoded.astype(float)

# Refit the OLS model using the encoded data
beta_encoded = ols_regression(X_train_encoded, y_train_encoded)

# Make predictions for both training and testing sets using the encoded model
y_train_pred_encoded = X_train_encoded @ beta_encoded
y_test_pred_encoded = X_test_encoded @ beta_encoded

# Calculate R^2 score for the encoded model on training and testing sets
r2_train_encoded = r2_score(y_train_encoded, y_train_pred_encoded)
r2_test_encoded = r2_score(y_test_encoded, y_test_pred_encoded)

print(f"Updated R^2 score for the training set after encoding: {r2_train_encoded}")
print(f"Updated R^2 score for the testing set after encoding: {r2_test_encoded}")


# %% [markdown]
# Q3.2e

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv

def compute_weights(X, x_query, tau):
    weights = np.exp(-np.sum((X - x_query) ** 2, axis=1) / (2 * tau ** 2))
    return np.diag(weights)

def locally_weighted_regression(X_train, y_train, x_query, tau):
    # Step 1: Compute the weights for the query point
    A = compute_weights(X_train, x_query, tau)
    
    # Step 2: Solve for w_star using the normal equation for weighted least squares
    XTX = X_train.T @ A @ X_train
    XTy = X_train.T @ A @ y_train
    w_star = pinv(XTX) @ XTy
    
    # Step 3: Make the prediction for the query point
    y_pred = x_query @ w_star
    return y_pred

# Generate some synthetic data for demonstration
np.random.seed(0)
X_train = np.random.rand(100, 2)  # 100 examples, 2 features
y_train = np.sin(X_train[:, 0] * 10) + np.random.randn(100) * 0.1  # Non-linear target

# Define the query point and tau
x_query = np.array([0.5, 0.5])  # Example query point
tau = 0.1

# Make a prediction for the query point
y_pred = locally_weighted_regression(X_train, y_train, x_query, tau)

# To visualize the effect of LWR, we could plot predictions over a range of points
x_queries = np.linspace(0, 1, 100)
predictions = [locally_weighted_regression(X_train, y_train, np.array([x, 0.5]), tau) for x in x_queries]


# %% [markdown]
# Q3f
# 

# %%

# Using the same one-hot encoded training and test sets from the earlier question.
X_train_encoded = np.c_[np.ones(train_df_encoded.shape[0]), train_df_encoded.drop('cnt', axis=1).values]
y_train_encoded = train_df_encoded['cnt'].values

X_test_encoded = np.c_[np.ones(test_df_encoded.shape[0]), test_df_encoded.drop('cnt', axis=1).values]
y_test_encoded = test_df_encoded['cnt'].values

# Take the first 200 samples for the reduced test set
X_test_reduced = X_test_encoded[:200]
y_test_reduced = y_test_encoded[:200]

# Ensure that all values are float64
X_train_encoded = X_train_encoded.astype(float)
y_train_encoded = y_train_encoded.astype(float)
X_test_reduced = X_test_reduced.astype(float)
y_test_reduced = y_test_reduced.astype(float)

# Define the compute_weights function
def compute_weights(X, x_query, tau):
    weights = np.exp(-np.sum((X - x_query) ** 2, axis=1) / (2 * tau ** 2))
    return np.diag(weights)

# Define the locally_weighted_regression function
def locally_weighted_regression(X_train, y_train, x_query, tau):
    A = compute_weights(X_train, x_query, tau)
    XTX = X_train.T @ A @ X_train
    XTy = X_train.T @ A @ y_train
    w_star = pinv(XTX) @ XTy
    y_pred = x_query @ w_star
    return y_pred

# Set the value of tau for LWR
tau = 1

# Make predictions on the reduced test set using LWR
y_test_pred = np.array([locally_weighted_regression(X_train_encoded, y_train_encoded, X_test_reduced[i], tau) for i in range(X_test_reduced.shape[0])])

# Calculate the R^2 score manually
def r2_score_manual(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

# Compute R^2 for the reduced test set
r2_test = r2_score_manual(y_test_reduced, y_test_pred)
print(f"R^2 score for Locally Weighted Regression with τ={tau}: {r2_test}")

# Discussion of the behavior:
print("\nDiscussion:")
print("As τ approaches 0, the model behaves like k-nearest neighbors (k=1) because the influence of nearby points becomes dominant, resulting in predictions very close to the closest training point.")
print("As τ increases, the influence of all points becomes more uniform, and the model behaves like ordinary least squares (OLS) regression, where all training points contribute equally.")

# %% [markdown]
# Q3.2g

# %%
# Plot a histogram of the target variable 'cnt'
plt.figure(figsize=(10, 6))
plt.hist(df['cnt'], bins=30, color='blue', edgecolor='black')
plt.title('Histogram of Target Variable (cnt)')
plt.xlabel('Count of Bikes')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# %% [markdown]
# Q3.2h+i

# %%
# 3.2(h)(i) Implement the Poisson regression algorithm
import autograd.numpy as np  # Import autograd's version of numpy
from autograd import grad
from sklearn.metrics import d2_tweedie_score

# Function for feature scaling (standardization)
def standardize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

# Ensure that the encoded data has matching dimensions and is of type float64
X_train_encoded = X_train_encoded.astype(float)
X_test_encoded = X_test_encoded.astype(float)

# Manually standardize the features of training and testing sets
X_train_standardized = standardize_features(X_train_encoded[:, 1:])  # Remove intercept column for scaling
X_test_standardized = standardize_features(X_test_encoded[:, 1:])

# Add intercept term back to the standardized data
X_train_standardized = np.c_[np.ones(X_train_standardized.shape[0]), X_train_standardized]
X_test_standardized = np.c_[np.ones(X_test_standardized.shape[0]), X_test_standardized]

# Define the Poisson loss function with gradient clipping
def poisson_loss(w, X, y):
    linear_combination = X @ w
    linear_combination = anp.clip(linear_combination, -10, 10)  # Clip to prevent overflow
    predictions = anp.exp(linear_combination)
    return -anp.sum(y * linear_combination - predictions)

# Compute gradient using autograd
grad_poisson_loss = grad(poisson_loss)

# Define the gradient descent function with gradient clipping
def gradient_descent_poisson(X, y, lr=0.01, n_iters=1000, clip_value=5.0):
    np.random.seed(0)
    w = np.random.randn(X.shape[1]) * 0.01  # Initialize weights

    for i in range(n_iters):
        grad_w = grad_poisson_loss(w, X, y)

        # Clip gradients to prevent large updates
        grad_w = np.clip(grad_w, -clip_value, clip_value)

        # Update weights with the learning rate
        w -= lr * grad_w

        # Print loss every 100 iterations to monitor progress
        if i % 100 == 0:
            loss = poisson_loss(w, X, y)

    return w

# Train the model using the adjusted gradient descent
w_trained = gradient_descent_poisson(X_train_standardized, y_train_encoded, lr=0.01, n_iters=1000)

def predict_poisson(X, w):
    linear_combination = X @ w
    linear_combination = np.clip(linear_combination, -10, 10)  # Clip for stability
    return anp.exp(linear_combination)

# Predict on the standardized test set
y_test_pred = predict_poisson(X_test_standardized, w_trained)

# Calculate R^2 score manually for Poisson regression
def r2_score_manual(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

# Train the model using the adjusted gradient descent
w_trained = gradient_descent_poisson(X_train_standardized, y_train_encoded, lr=0.01, n_iters=1000)

# Prediction function with clipping for stability
def predict_poisson(X, w):
    linear_combination = X @ w
    linear_combination = np.clip(linear_combination, -10, 10)  # Clip for stability
    return anp.exp(linear_combination)

# Predict on the standardized test set
y_test_pred = predict_poisson(X_test_standardized, w_trained)

# Define the regularized negative log-likelihood for Poisson regression
def poisson_neg_log_likelihood(w, X, y, lambda_reg=10):
    
    # Predicted value (lambda) for Poisson regression, with clipping to avoid overflow
    pred = np.exp(np.clip(X @ w, -100, 100))  # Use autograd.numpy's exp and clip
    
    # Regularized negative log-likelihood with small regularization term
    return -np.sum(y * np.log(pred + 1e-10) - pred) + lambda_reg * np.sum(w ** 2)

# Gradient of the negative log-likelihood
poisson_gradient = grad(poisson_neg_log_likelihood)

# Gradient descent for Poisson regression with additional debugging and gradient clipping
def gradient_descent_poisson(X, y, learning_rate=0.0000029, iterations=1000, lambda_reg=10, clip_value=1000):
    w = np.zeros(X.shape[1])  # Initialize weights
    for i in range(iterations):
        gradient = poisson_gradient(w, X, y, lambda_reg)  # Regularized gradient
        
        # Clip gradients to prevent explosion
        gradient = np.clip(gradient, -clip_value, clip_value)

        w -= learning_rate * gradient  # Update weights

    return w

# Fit Poisson regression using gradient descent
w_poisson = gradient_descent_poisson(X_train_encoded, y_train_encoded)

# Make predictions
y_pred_poisson = np.exp(np.clip(X_test_encoded @ w_poisson, -100, 100))

# Calculate D² score using the Tweedie deviance metric (for full test set)
D_score_poisson = d2_tweedie_score(y_test_encoded, y_pred_poisson, power=1)

# Calculate R² score for Poisson regression (for full test set)
r2_poisson = r2_score(y_test_encoded, y_pred_poisson)

print(f"\nD² Score (Poisson Regression, full test set): {D_score_poisson}")
print(f"R² Score (Poisson Regression, full test set): {r2_poisson}")

# %% [markdown]
# Q3.2j

# %%
# Display the weights for Linear Regression
print("Final weights for Linear Regression:")
print(beta_encoded)

# Display the weights for Poisson Regression
print("\nFinal weights for Poisson Regression:")
print(w_trained)

# Find the most and least significant features for Linear Regression
most_significant_idx_linear = np.argmax(np.abs(beta_encoded[1:]))  # Skip intercept
least_significant_idx_linear = np.argmin(np.abs(beta_encoded[1:]))

print(f"\nMost significant feature for Linear Regression: {train_df_encoded.columns[most_significant_idx_linear]}")
print(f"Least significant feature for Linear Regression: {train_df_encoded.columns[least_significant_idx_linear]}")

# Find the most and least significant features for Poisson Regression
most_significant_idx_poisson = np.argmax(np.abs(w_trained[1:]))  # Skip intercept
least_significant_idx_poisson = np.argmin(np.abs(w_trained[1:]))

print(f"\nMost significant feature for Poisson Regression: {train_df_encoded.columns[most_significant_idx_poisson]}")
print(f"Least significant feature for Poisson Regression: {train_df_encoded.columns[least_significant_idx_poisson]}")



