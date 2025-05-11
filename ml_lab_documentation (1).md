# Machine Learning Lab Documentation

This document provides comprehensive explanations for each ML algorithm implementation in your lab. Each section covers code functionality, theoretical notes, and key implementation details to help you understand and memorize the code for your final exam.

## Table of Contents
1. [KNN - K-Nearest Neighbors](#1-knn---k-nearest-neighbors)
2. [Naive Bayes](#2-naive-bayes)
3. [Decision Tree](#3-decision-tree)
4. [SVM - Support Vector Machine](#4-svm---support-vector-machine)
5. [K-Means Clustering](#5-k-means-clustering)
6. [Agglomerative Clustering](#6-agglomerative-clustering)
7. [K-Medoid Clustering](#7-k-medoid-clustering)
8. [Summary of Dataset Exploration Techniques](#8-summary-of-dataset-exploration-techniques)

## 1. KNN - K-Nearest Neighbors

```python
# Import necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris # Load the built-in Iris dataset

# Load dataset and explore basic properties
dataset = load_iris()
print(dataset.feature_names) # Show feature names (sepal length/width, petal length/width)
print(dataset.target) # Show target values (0, 1, 2 representing different iris species)
np.unique(dataset.target) # Check unique values in target (0, 1, 2)
print(dataset.data.shape) # Print dataset dimensions (150 samples, 4 features)

# Split data into training and testing sets (80% train, 20% test)
from sklearn.model_selection import train_test_split
X = dataset.data # Feature matrix
y = dataset.target # Target vector
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)

# Check shapes of training and testing sets
print(X_train.shape) # 120 samples, 4 features
print(X_test.shape) # 30 samples, 4 features
print(y_train.shape) # 120 target values
print(y_test.shape) # 30 target values

# Create and train KNN classifier with k=5 neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5) # Create model with 5 neighbors
knn.fit(X_train, y_train) # Train the model

# Predict on test set and evaluate accuracy
y_pred = knn.predict(X_test) # Predict on test data
print(y_pred) # Print predictions

# Calculate accuracy
from sklearn import metrics
ac = metrics.accuracy_score(y_test, y_pred) # Compare predictions with true values
print(ac) # Print accuracy score

# Make prediction on new sample
knn.predict([[6, 3.4, 4.5, 1.6]]) # Predict class for a new flower

# Visualize the dataset with a 2D scatter plot (using first two features)
from matplotlib.colors import ListedColormap
colormap = ListedColormap(['b', 'r', 'g']) # Define colors for each class
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap) # Plot first two features colored by class
plt.show()
```

### KNN Theory and Implementation Notes:

1. **Algorithm Choice**: KNN is chosen as a starting point because:
   - It's simple and intuitive (classifies based on majority class among k nearest neighbors)
   - Works well with the Iris dataset which has clear class boundaries
   - Doesn't make assumptions about the underlying data distribution

2. **n_neighbors=5**: The choice of k=5 is a common starting point that:
   - Balances between overfitting (k too small) and over-smoothing (k too large)
   - Provides an odd number to avoid ties in binary classification

3. **Data Preprocessing**: No explicit scaling is done because:
   - The Iris dataset features have similar scales
   - For other datasets, feature scaling would be crucial for KNN since it uses distance metrics

4. **Implementation Structure**:
   - Dataset loading → Exploration → Train/test split → Model training → Prediction → Evaluation
   - This pattern is standard across most supervised learning implementations

5. **Visualization**: Only uses 2 features (sepal length/width) to create a 2D plot, even though the model uses all 4 features

## 2. Naive Bayes

```python
# Example 1: Categorical Naive Bayes (all categorical features)
import pandas as pd

# Load dataset with categorical features
data_example_1 = pd.read_csv('data-all-categorical.data')
data_example_1.head() # View first few rows
data_example_1.dtypes # Check data types

# Explore unique values in each column
print(data_example_1['Outlook'].unique()) # ['Sunny', 'Overcast', 'Rain']
print(data_example_1['Temperature'].unique()) # ['Hot', 'Mild', 'Cool']
print(data_example_1['Humidity'].unique()) # ['High', 'Normal']
print(data_example_1['Wind'].unique()) # ['Weak', 'Strong']
print(data_example_1['PlayTennis'].unique()) # ['No', 'Yes']

# Encode categorical features to numerical
data_example_1_encoded = data_example_1.replace({
    'Outlook': {'Sunny': 0, 'Overcast': 1, 'Rain': 2},
    'Temperature': {'Hot': 0, 'Mild': 1, 'Cool': 2},
    'Humidity': {'High': 0, 'Normal': 1},
    'Wind': {'Weak': 0, 'Strong': 1},
    'PlayTennis': {'No': 0, 'Yes': 1}
})

# Prepare features (X) and target (y)
X = data_example_1_encoded.drop(columns=['Day', 'PlayTennis'])
y = data_example_1_encoded['PlayTennis']

# Create and train Categorical Naive Bayes model
from sklearn.naive_bayes import CategoricalNB
cat_nb = CategoricalNB() # Can use alpha parameter for smoothing (default=1.0)
cat_nb.fit(X, y) # Train the model

# Predict and evaluate
y_pred = cat_nb.predict(X)
from sklearn.metrics import accuracy_score
accuracy_score(y, y_pred) # Calculate accuracy

# Example 2: Gaussian Naive Bayes (numerical features)
data_example_2 = pd.read_csv('data-with-numerical.data')
data_example_2.head() # View first few rows
data_example_2.dtypes # Check data types

# Check unique values for categorical columns
print(data_example_2['Outlook'].unique())
print(data_example_2['Wind'].unique())
print(data_example_2['PlayTennis'].unique())

# Encode categorical features
data_example_2_encoded = data_example_2.replace({
    'Outlook': {'Sunny': 0, 'Overcast': 1, 'Rain': 2},
    'Wind': {'Weak': 0, 'Strong': 1},
    'PlayTennis': {'No': 0, 'Yes': 1}
})

# Create features (X) and target (y) - using only numerical features
X = data_example_2_encoded[['Temperature', 'Humidity']]
y = data_example_2_encoded['PlayTennis']

# Create and train Gaussian Naive Bayes model (for numerical features)
from sklearn.naive_bayes import GaussianNB
gaussian_nb = GaussianNB() # Assumes normal distribution for numerical features
gaussian_nb.fit(X, y) # Train the model

# Predict and evaluate
y_pred = gaussian_nb.predict(X)
accuracy_score(y, y_pred) # Calculate accuracy

# Example 3: Mixed Naive Bayes (both categorical and numerical features)
X = data_example_2_encoded.drop(columns=['Day', 'PlayTennis'])
y = data_example_2_encoded['PlayTennis']

# Use a custom implementation for mixed features
from mixed_naive_bayes import MixedNB
# Specify which features are categorical (by index)
mixed_nb = MixedNB(categorical_features=[0, 3]) # Outlook (index 0) and Wind (index 3)
mixed_nb.fit(X, y) # Train the model

# Predict and evaluate
y_pred = mixed_nb.predict(X)
accuracy_score(y, y_pred) # Calculate accuracy
```

### Naive Bayes Theory and Implementation Notes:

1. **Algorithm Choice**: Naive Bayes is used here because:
   - It works well with categorical data (weather dataset)
   - Performs well even with small training datasets
   - Computationally efficient and simple to implement
   - Good for multi-class classification problems

2. **Multiple NB Variants**: The code demonstrates three different NB implementations:
   - **CategoricalNB**: For purely categorical features
   - **GaussianNB**: For numerical features, assuming normal distribution
   - **MixedNB**: Custom implementation for mixed data types

3. **Alpha Parameter (Smoothing)**:
   - Default value is 1.0 (Laplace smoothing)
   - Prevents zero probabilities when a feature value doesn't appear with a class in training
   - Higher values provide more smoothing (more regularization)

4. **Feature Independence**:
   - The "naive" assumption is that features are conditionally independent given class
   - This simplifies computation but is rarely true in practice
   - Despite this unrealistic assumption, NB often works well in practice

5. **Implementation Structure**:
   - Data loading → Encoding categorical variables → Feature/target preparation → Model specific training → Evaluation
   - Note how the same dataset is used with different feature subsets for different NB variants

## 3. Decision Tree

```python
import pandas as pd

# Load and explore dataset
df = pd.read_csv("salaries.csv")
df.head() # View first few rows

# Prepare features (X) and target (y)
inputs = df.drop('salary_more_then_100k', axis='columns') # Features
target = df['salary_more_then_100k'] # Target

# Encode categorical features
from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder() # For 'company' column
le_job = LabelEncoder() # For 'job' column
le_degree = LabelEncoder() # For 'degree' column

# Transform each categorical column
inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

# Remove original categorical columns
inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')

# Create and train Decision Tree model
from sklearn import tree
model = tree.DecisionTreeClassifier() # Create model with default parameters
model.fit(inputs_n, target) # Train the model

# Evaluate model accuracy on training data
model.score(inputs_n, target)

# Make predictions with new data
model.predict([[2, 1, 0]]) # Example: company_n=2, job_n=1, degree_n=0
model.predict([[2, 1, 1]]) # Example: company_n=2, job_n=1, degree_n=1
```

### Decision Tree Theory and Implementation Notes:

1. **Algorithm Choice**: Decision Tree is used because:
   - It handles both numerical and categorical data naturally
   - Creates human-interpretable rules
   - Requires minimal data preprocessing
   - Can capture non-linear relationships

2. **No Train/Test Split**: Unlike previous examples, this code:
   - Doesn't split data into train and test sets
   - Evaluates model on training data (risk of overfitting assessment)
   - This is not ideal for real applications but serves educational purposes

3. **Feature Encoding**:
   - Uses LabelEncoder to convert categorical text values to numbers
   - Creates separate encoder for each categorical column
   - This maintains the ability to inverse transform later if needed

4. **Default Parameters**:
   - Uses default parameters for DecisionTreeClassifier which include:
     - criteria='gini' (Gini impurity for splits)
     - max_depth=None (fully grown tree)
     - min_samples_split=2 (minimum samples to split an internal node)
   - No pruning is applied which could lead to overfitting

5. **Implementation Structure**:
   - Data loading → Categorical encoding → Model creation → Training → Evaluation → Prediction

## 4. SVM - Support Vector Machine

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load and explore dataset
df = pd.read_csv('Social_Network_Ads.csv')
df.head() # View first few rows
df.shape # Check dataset dimensions

# Prepare features (X) and target (y)
X = df.iloc[:, [2, 3]] # Use only 'Age' and 'EstimatedSalary' features
Y = df.iloc[:, 4] # Target variable (Purchased or not)

# Split data into training and testing sets (75% train, 25% test)
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=0)

print("Training data: ", X_Train.shape)
print("Testing data: ", X_Test.shape)

# Scale features (important for SVM)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train) # Fit and transform training data
X_Test = sc_X.transform(X_Test) # Transform test data using same scaling

# Linear SVM
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0) # Create linear SVM
classifier.fit(X_Train, Y_Train) # Train the model

# Predict and evaluate
Y_Pred = classifier.predict(X_Test)
from sklearn import metrics
print('Accuracy Score with linear kernel:')
print(metrics.accuracy_score(Y_Test, Y_Pred))

# RBF kernel SVM (default gamma)
classifier = SVC(kernel='rbf') # Create SVM with RBF kernel
classifier.fit(X_Train, Y_Train) # Train the model

# Predict and evaluate
Y_Pred = classifier.predict(X_Test)
print('Accuracy Score with default rbf kernel:')
print(metrics.accuracy_score(Y_Test, Y_Pred))

# Optimized RBF kernel SVM
classifier = SVC(kernel='rbf', gamma=15, C=7, random_state=0) # Create SVM with tuned parameters
classifier.fit(X_Train, Y_Train) # Train the model

# Predict and evaluate
Y_Pred = classifier.predict(X_Test)
print('Accuracy Score On Test Data with optimized rbf kernel:')
print(metrics.accuracy_score(Y_Test, Y_Pred))

# Polynomial kernel SVM
svc = SVC(kernel='poly', degree=4) # Create SVM with polynomial kernel
svc.fit(X_Train, Y_Train) # Train the model

# Predict and evaluate
y_pred = svc.predict(X_Test)
print('Accuracy Score with poly kernel and degree:')
print(metrics.accuracy_score(Y_Test, Y_Pred))

# Visualize training data
plt.scatter(X_Train[:, 0], X_Train[:, 1], c=Y_Train)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Training Data')
plt.show()

# Visualize test data
plt.scatter(X_Test[:, 0], X_Test[:, 1], c=Y_Test)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Test Data')
plt.show()

# Create and train linear SVM for visualization
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_Train, Y_Train)

# Predict and visualize decision boundary
Y_Pred = classifier.predict(X_Test)
plt.scatter(X_Test[:, 0], X_Test[:, 1], c=Y_Test)

# Create the hyperplane
w = classifier.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5, 2.5)
yy = a * xx - (classifier.intercept_[0]) / w[1]

# Plot the hyperplane
plt.plot(xx, yy)
plt.axis("off")
plt.show()
```

### SVM Theory and Implementation Notes:

1. **Algorithm Choice**: SVM is chosen because:
   - It works well for binary classification problems
   - Effective in high-dimensional spaces (though only 2 features used here)
   - Memory efficient as it uses a subset of training points (support vectors)
   - Versatile through different kernel functions

2. **Feature Scaling**:
   - StandardScaler is applied to normalize features
   - This is crucial for SVM as it's sensitive to feature scales
   - Uses same scaler for both train and test data (fit on train, transform on test)

3. **Multiple Kernel Experiments**:
   - **Linear kernel**: Simple decision boundary (hyperplane)
   - **RBF kernel**: Non-linear decision boundary using radial basis function
   - **RBF with tuned parameters**: gamma=15, C=7 for better performance
   - **Polynomial kernel**: Non-linear boundary with polynomial function

4. **Hyperparameter Tuning**:
   - **C parameter**: Controls trade-off between smooth decision boundary and classifying training points correctly (higher C = less regularization)
   - **gamma parameter**: Defines influence radius of each training example (higher gamma = more complex, tighter boundary)
   - **degree parameter**: Controls flexibility of polynomial kernel

5. **Implementation Structure**:
   - Data loading → Feature selection → Train/test split → Feature scaling → Multiple model variations with different kernels → Evaluation of each → Visualization

6. **Visualization Technique**:
   - Calculates decision boundary using weight coefficients and intercept
   - Only possible for linear kernel (linear decision boundary)
   - Visualizes the margin between classes

## 5. K-Means Clustering

```python
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Load and explore dataset
df = pd.read_csv("income.csv")
df.head() # View first few rows

# Visualize raw data
plt.scatter(df.Age, df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')

# Create and fit KMeans model with 3 clusters
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])

# Add cluster labels to dataframe
df['cluster'] = y_predicted
df.head()

# Display cluster centers
km.cluster_centers_

# Scale features to [0,1] range for better clustering
scaler = MinMaxScaler()
df['Income($)'] = scaler.fit_transform(df[['Income($)']])
df['Age'] = scaler.fit_transform(df[['Age']])
df.head()

# Visualize scaled data
plt.scatter(df.Age, df['Income($)'])

# Re-run KMeans with scaled data
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])

# Add cluster labels to dataframe
df['cluster'] = y_predicted
df.head()

# Display cluster centers for scaled data
km.cluster_centers_

# Visualize clusters with different colors
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

plt.scatter(df1.Age, df1['Income($)'], color='green')
plt.scatter(df2.Age, df2['Income($)'], color='red')
plt.scatter(df3.Age, df3['Income($)'], color='black')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
            color='purple', marker='*', label='centroid')
plt.legend()

# Determine optimal number of clusters using elbow method
import warnings
warnings.filterwarnings("ignore")  # Ignore warnings

sse = [] # Sum of squared errors
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age', 'Income($)']])
    sse.append(km.inertia_) # inertia is sum of squared distances to closest centroid

# Plot elbow curve
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)
```

### K-Means Clustering Theory and Implementation Notes:

1. **Algorithm Choice**: K-Means is used because:
   - It's simple and efficient for clustering
   - Works well with spherical clusters
   - Easy to implement and interpret
   - Scales well to large datasets

2. **Feature Scaling**:
   - The code first tries without scaling, then applies MinMaxScaler
   - Scaling is critical for K-Means as it uses Euclidean distance
   - Without scaling, features with larger values would dominate
   - MinMaxScaler transforms features to [0,1] range

3. **Running K-Means Twice**:
   - First run: On raw data (suboptimal)
   - Second run: On scaled data (better cluster separation)
   - This demonstrates the importance of preprocessing

4. **Cluster Interpretation**:
   - The code separates data points by cluster assignment for visualization
   - Centroids are plotted as special markers (purple stars)
   - This helps identify the center of each cluster

5. **Elbow Method**:
   - Systematic approach to find optimal number of clusters
   - Plots SSE (inertia) against number of clusters (K)
   - The "elbow" point (where adding more clusters gives diminishing returns) suggests optimal K
   - This is crucial because K-Means requires predefined number of clusters

6. **Implementation Structure**:
   - Data loading → Initial clustering → Feature scaling → Re-clustering → Visualization → Cluster optimization

## 6. Agglomerative Clustering

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Load breast cancer dataset
BreastData = load_breast_cancer()
x = BreastData.data
y = BreastData.data

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=44)

# Create and fit Agglomerative Clustering model
AggClusteringModel = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage='ward')
y_pred_train = AggClusteringModel.fit_predict(x_train)
y_pred_test = AggClusteringModel.fit_predict(x_test)

# Create dendrogram for training data (first 30 samples)
dendrogram = sch.dendrogram(sch.linkage(x_train[:30, :], method='ward'))
plt.title('Training Set')
plt.xlabel('x values')
plt.ylabel('Distances')
plt.show()

# Create dendrogram for test data (first 30 samples)
dendrogram = sch.dendrogram(sch.linkage(x_test[:30, :], method='ward'))
plt.title('Training Set')
plt.xlabel('x values')
plt.ylabel('Distances')
plt.show()

# Visualize clusters in training set
plt.scatter(x_train[y_pred_train == 0, 0], x_train[y_pred_train == 0, 1], s=10, c='red', label='Cluster 1')
plt.scatter(x_train[y_pred_train == 1, 0], x_train[y_pred_train == 1, 1], s=10, c='blue', label='Cluster 2')
plt.scatter(x_train[y_pred_train == 2, 0], x_train[y_pred_train == 2, 1], s=10, c='green', label='Cluster 3')
plt.scatter(x_train[y_pred_train == 3, 0], x_train[y_pred_train == 3, 1], s=10, c='cyan', label='Cluster 4')
plt.scatter(x_train[y_pred_train == 4, 0], x_train[y_pred_train == 4, 1], s=10, c='black', label='Cluster 5')
plt.title('Training set')
plt.xlabel('x value')
plt.ylabel('y value')
plt.legend()
plt.show()

# Visualize clusters in test set
plt.scatter(x_test[y_pred_test == 1, 0], x_test[y_pred_test == 1, 1], s=10, c='blue', label='Cluster 2')
plt.scatter(x_test[y_pred_test == 2, 0], x_test[y_pred_test == 2, 1], s=10, c='green', label='Cluster 3')
plt.scatter(x_test[y_pred_test == 3, 0], x_test[y_pred_test == 3, 1], s=10, c='cyan', label='Cluster 4')
plt.scatter(x_test[y_pred_test == 4, 0], x_test[y_pred_test == 4, 1], s=10, c='black', label='Cluster 5')
plt.title('Training set')
plt.xlabel('x value')
plt.ylabel('y value')
plt.legend()
plt.show()
```

### Agglomerative Clustering Theory and Implementation Notes:

1. **Algorithm Choice**: Agglomerative Clustering is used because:
   - It's a hierarchical clustering approach (bottom-up)
   - Doesn't require predetermined number of clusters (though we set it to 2 here)
   - Creates dendrograms to visualize clustering hierarchy
   - Good for discovering hierarchical relationships in data

2. **Parameters**:
   - **n_clusters=2**: Creates two final clusters
   - **affinity="euclidean"**: Uses Euclidean distance metric
   - **linkage='ward'**: Ward's method minimizes variance within clusters
     - Other options include 'single', 'complete', 'average' linkage

3. **Dendrograms**:
   - Created for both training and test sets (first 30 samples only due to visualization limitations)
   - Shows hierarchical merging of clusters
   - Height represents distance between merged clusters
   - Helps determine optimal number of clusters

4. **Implementation Structure**:
   - Dataset loading → Train/test split → Model creation → Fitting on both train and test → Dendrogram visualization → Cluster visualization

5. **Breast Cancer Dataset**:
   - High-dimensional medical dataset (many features)
   - Binary classification problem (not actually used for classification here)
   - Note: The code has a potential issue using the same data for both x and y

6. **Visualization Limitation**:
   - The scatter plots show only the first two dimensions
   - This is a common limitation when visualizing high-dimensional data
   - Real clustering uses all dimensions, but visualization is limited

## 7. K-Medoid Clustering

```python
from sklearn_extra.cluster import KMedoids
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load and explore dataset
df = pd.read_csv("income.csv")
print(df.columns) # Display column names

# Select features
features = df[['Age', 'Income($)']]

# Scale features to [0,1] range
scaler = MinMaxScaler()
df[['Age', 'Income($)']] = scaler.fit_transform(df[['Age', 'Income($)']])

# Set number of clusters
num_clusters = 3

# Create and fit K-Medoids model
kmedoids = KMedoids(n_clusters=num_clusters, random_state=42)
clusters = kmedoids.fit_predict(df[['Age', 'Income($)']])

# Add cluster labels to dataframe
df['cluster'] = clusters

# Display medoid centers
print(kmedoids.cluster_centers_, '\n')
print(kmedoids.cluster_centers_[:, 0])

# Visualize clusters and medoids
plt.scatter(df['Age'], df['Income($)'], c=df['cluster'], cmap='rainbow')
plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], 
            s=200, c='black', marker='X', label='Medoids')
plt.xlabel('العمر') # Age in Arabic
plt.ylabel('الدخل ($)') # Income in Arabic
plt.title('تجميع البيانات باستخدام K-Medoids') # Data clustering using K-Medoids in Arabic
plt.legend()
plt.show()
```

### K-Medoid Clustering Theory and Implementation Notes:

1. **Algorithm Choice**: K-Medoid is used because:
   - It's more robust to outliers than K-Means
   - Uses actual data points as cluster centers (medoids) unlike K-Means (centroids)
   - Better for categorical data or when mean isn't meaningful
   - Works with any distance metric (not just Euclidean)

2. **Implementation Differences from K-Means**:
   - Uses `KMedoids` from `sklearn_extra.cluster` (not in core sklearn)
   - Centers are actual data points (medoids)
   - More computationally expensive than K-Means
   - Often more interpretable as cluster centers are real observations

3. **Feature Scaling**:
   - Similar to K-Means, uses MinMaxScaler for feature normalization
   - This is important for distance-based algorithms like K-Medoids

4. **No Elbow Method**:
   - Unlike the K-Means example, there's no elbow method implementation
   - Directly sets num_clusters=3 without optimization
   - In practice, similar methods could be applied (e.g., silhouette score)

5. **Visualization**:
   - Uses rainbow colormap to distinguish clusters
   - Medoids marked with large black X markers
   - Labels in Arabic (showing internationalization)

6. **Implementation Structure**:
   - Data loading → Feature selection → Feature scaling → Model creation and fitting → Visualization

## 8. Summary of Dataset Exploration Techniques

Throughout your code, you've used several consistent methods to explore and understand datasets before applying ML algorithms:

1. **Basic Dataset Inspection**:
   - `df.head()`: View first few rows of the dataset
   - `df.shape`: Check dimensions (rows and columns)
   - `df.dtypes`: Check data types of each column
   - `print(dataset.feature_names)`: Display feature names for sklearn datasets
   - `print(dataset.data.shape)`: Check dimensions of sklearn datasets

2. **Target Variable Exploration**:
   - `np.unique(dataset.target)`: Find unique classes in target variable
   - `print(dataset.target)`: Display all target values

3. **Feature Exploration**:
   - `print(df['column_name'].unique())`: Check unique values in a feature
   - `plt.scatter(X[:, 0], X[:, 1], c=y)`: Visualize feature relationships and class distribution

4. **Data Splitting Verification**:
   - `print(X_train.shape)`: Confirm training set size
   - `print(X_test.shape)`: Confirm test set size
   - `print(y_train.shape)`: Verify target training set size
   - `print(y_test.shape)`: Verify target test set size

5. **Model Output Inspection**:
   - `print(y_pred)`: Display model predictions
   - `len(y_pred)`: Confirm number of predictions matches test set
   - `print(metrics.accuracy_score(y_test, y_pred))`: Calculate and display accuracy

6. **Clustering Analysis Tools**:
   - `km.cluster_centers_`: View cluster centers coordinates
   - `plt.scatter(..., c=df['cluster'])`: Visualize cluster assignments
   - Dendrograms for hierarchical clustering visualization
   - Elbow method for determining optimal number of clusters

These exploratory techniques are essential for:
- Understanding data distributions and relationships
- Verifying data preprocessing steps
- Ensuring models receive properly formatted inputs
- Interpreting model outputs and evaluating performance
- Making informed decisions about algorithm parameters