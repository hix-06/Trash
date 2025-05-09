# Machine Learning Lab Documentation

## Table of Contents
1. [K-Nearest Neighbors (KNN)](#1-k-nearest-neighbors-knn)
2. [Naive Bayes](#2-naive-bayes)
3. [Decision Tree](#3-decision-tree)
4. [Support Vector Machine (SVM)](#4-support-vector-machine-svm)
5. [K-Means Clustering](#5-k-means-clustering)
6. [Agglomerative Clustering](#6-agglomerative-clustering)
7. [K-Medoids Clustering](#7-k-medoids-clustering)

---

## 1. K-Nearest Neighbors (KNN)

### Code Structure:
```python
# Import necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from matplotlib.colors import ListedColormap

# Load the dataset
dataset = load_iris()  # Using the built-in iris dataset
X = dataset.data       # Features - contains 4 measurements for each iris sample
y = dataset.target     # Target labels - contains 3 classes (0,1,2) for different iris species

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # Create model with k=5 neighbors
knn.fit(X_train, y_train)                  # Train the model using training data

# Make predictions and evaluate
y_pred = knn.predict(X_test)              # Predict test data labels
ac = metrics.accuracy_score(y_test, y_pred)  # Calculate accuracy by comparing to true labels
print(ac)

# Use the model to predict a new flower species
prediction = knn.predict([[6, 3.4, 4.5, 1.6]])  # Pass measurements of unknown flower

# Visualize the data (first two features only for 2D plotting)
colormap = ListedColormap(['b', 'r', 'g'])  # Define colors for 3 classes
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap)  # Plot sepal length vs sepal width
plt.show()
```

### Theoretical Notes:

1. **Why KNN for this problem?**
   - **Why KNN?** KNN is chosen because it's non-parametric (makes no assumptions about data distribution), which is ideal for the Iris dataset where class boundaries are relatively clear but not necessarily linear
   - **Why Iris dataset?** This dataset is perfect for KNN since it has clear clusters with minimal overlap between classes, allowing KNN to perform well even with simple Euclidean distance measures
   - **Why use all four features for prediction but only two for visualization?** All features contribute to classification accuracy, but we're limited to plotting in 2D, so we select the first two features (sepal measurements) which already show reasonable class separation

2. **Why these parameter choices?**
   - **Why k=5?** This value balances between overfitting (too small k) and underfitting (too large k). With k=1, the model would be too sensitive to noise, while large k values would blur the class boundaries too much
   - **Why test_size=0.20?** This 80/20 split provides sufficient training data while keeping enough samples for testing. For Iris dataset with only 150 samples, we need a reasonable number of test samples (30) to evaluate performance
   - **Why random_state=4?** This ensures reproducibility of results - the exact same train/test split will be created each time the code runs

3. **Why this visualization approach?**
   - **Why use a colormap?** Color mapping helps distinguish between the three iris classes visually, making it easier to see how well the classes are separated in feature space
   - **Why not visualize the decision boundaries?** While possible, the simple 2D plot already shows class separation well. In practice, you would plot decision boundaries to better understand how KNN classifies regions of the feature space

4. **How KNN works and why it's suitable:**
   - For a new data point, KNN examines the k nearest training points (using Euclidean distance by default)
   - It assigns the most common class among those k neighbors
   - **Why this works:** The algorithm assumes that similar instances likely belong to the same class - a reasonable assumption for many real-world problems, especially biomedical data like Iris

---

## 2. Naive Bayes

### Code Structure:
```python
import pandas as pd
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.metrics import accuracy_score
from mixed_naive_bayes import MixedNB  # Custom library for mixed data

# Part 1: Categorical Naive Bayes
# Load categorical data - PlayTennis dataset with all categorical features
data_example_1 = pd.read_csv('data-all-categorical.data')

# Encode categorical features to numerical values - required for most ML algorithms
data_example_1_encoded = data_example_1.replace({
    'Outlook': {'Sunny': 0, 'Overcast': 1, 'Rain': 2},
    'Temperature': {'Hot': 0, 'Mild': 1, 'Cool': 2},
    'Humidity': {'High': 0, 'Normal': 1},
    'Wind': {'Weak': 0, 'Strong': 1},
    'PlayTennis': {'No': 0, 'Yes': 1}  # Binary target variable
})

# Prepare features and target
X = data_example_1_encoded.drop(columns=['Day', 'PlayTennis'])  # Features
y = data_example_1_encoded['PlayTennis']  # Target

# Train CategoricalNB model - designed for categorical features
cat_nb = CategoricalNB()  # Alpha parameter (default=1.0) controls Laplace smoothing
cat_nb.fit(X, y)

# Evaluate categorical model
y_pred = cat_nb.predict(X)
accuracy_score(y, y_pred)  # Calculate accuracy

# Part 2: Gaussian Naive Bayes
# Load data with numerical features - temperature and humidity are continuous
data_example_2 = pd.read_csv('data-with-numerical.data')

# Encode only the categorical features
data_example_2_encoded = data_example_2.replace({
    'Outlook': {'Sunny': 0, 'Overcast': 1, 'Rain': 2},
    'Wind': {'Weak': 0, 'Strong': 1},
    'PlayTennis': {'No': 0, 'Yes': 1}
})

# Use only numerical features for GaussianNB
X = data_example_2_encoded[['Temperature', 'Humidity']]  # Select only numerical features
y = data_example_2_encoded['PlayTennis']

# Train GaussianNB model - designed for continuous features
gaussian_nb = GaussianNB()  # Assumes features follow normal distribution
gaussian_nb.fit(X, y)

# Evaluate Gaussian model
y_pred = gaussian_nb.predict(X)
accuracy_score(y, y_pred)

# Part 3: Mixed Naive Bayes
# Use all features (mix of categorical and numerical)
X = data_example_2_encoded.drop(columns=['Day', 'PlayTennis'])
y = data_example_2_encoded['PlayTennis']

# Train Mixed Naive Bayes model (specify which features are categorical)
mixed_nb = MixedNB(categorical_features=[0, 3])  # Indices of Outlook and Wind
mixed_nb.fit(X, y)

# Evaluate mixed model
y_pred = mixed_nb.predict(X)
accuracy_score(y, y_pred)
```

### Theoretical Notes:

1. **Why Naive Bayes for this problem?**
   - **Why Naive Bayes?** Naive Bayes is chosen because it's computationally efficient, works well with small datasets, and is excellent for categorical data like weather conditions in the PlayTennis example
   - **Why "naive"?** The algorithm makes a "naive" assumption that features are independent, which surprisingly works well even when this assumption is violated (as in weather data, where humidity and outlook are clearly related)
   - **Why test on same data used for training?** This is done to demonstrate basic implementation. In practice, you would use separate test data to avoid overfitting assessment. Here we're just checking if the model can learn the training patterns

2. **Why multiple Naive Bayes implementations?**
   - **Why CategoricalNB?** Specifically designed for categorical features, it models the data using multinomial distributions, ideal for the first dataset with all categorical features
   - **Why GaussianNB?** Used for continuous numerical features that follow (approximately) normal distribution, making it appropriate for temperature and humidity
   - **Why MixedNB?** Real-world datasets often contain both categorical and numerical features. The custom implementation handles mixed data types by applying appropriate probability models to each type
   - **Why specify categorical_features=[0, 3]?** This tells the model which columns are categorical (Outlook at index 0 and Wind at index 3), so it can apply the correct probability distribution to each feature type

3. **Why parameter choices?**
   - **Why default alpha=1.0 in CategoricalNB?** This is Laplace smoothing - prevents zero probabilities when a feature value doesn't appear with a particular class in training data. Value 1.0 adds one pseudo-observation of each feature value with each class
   - **Why no explicit parameters for GaussianNB?** It automatically estimates mean and variance of features from training data, which are sufficient parameters for normal distribution

4. **How Naive Bayes works and why it's effective:**
   - Based on Bayes theorem: P(class|features) ∝ P(features|class) × P(class)
   - Calculates probability of each class given the feature values
   - **Why it works despite "naive" assumption:** Even when features are dependent, NB often performs well for classification tasks because it doesn't need exact probability estimates, just the relative ranking of class probabilities to make correct decisions
   - **Why it's fast:** Requires only a single pass through the data to calculate necessary probabilities, making it computationally efficient

---

## 3. Decision Tree

### Code Structure:
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# Load the dataset - salary prediction based on company, job, degree
df = pd.read_csv("salaries.csv")

# Prepare features and target
inputs = df.drop('salary_more_then_100k', axis='columns')  # Features
target = df['salary_more_then_100k']  # Binary target - high salary or not

# Encode categorical features - trees need numerical input
le_company = LabelEncoder()  # Create encoder for company names
le_job = LabelEncoder()      # Create encoder for job titles
le_degree = LabelEncoder()   # Create encoder for degree types

# Transform each categorical feature to numerical values
inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

# Remove original categorical columns after encoding
inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')

# Create and train decision tree model
model = tree.DecisionTreeClassifier()  # Using default parameters
model.fit(inputs_n, target)  # Train the model

# Evaluate the model
accuracy = model.score(inputs_n, target)  # Calculate accuracy on training data

# Make predictions on new data
model.predict([[2, 1, 0]])  # Example: company_id=2, job_id=1, degree_id=0
model.predict([[2, 1, 1]])  # Example: company_id=2, job_id=1, degree_id=1
```

### Theoretical Notes:

1. **Why Decision Tree for this problem?**
   - **Why Decision Tree?** It's chosen because the salary prediction problem involves multiple categorical inputs with potentially complex relationships. Decision trees can capture non-linear decision rules based on different combinations of company, job, and degree
   - **Why no preprocessing beyond encoding?** Unlike many algorithms, decision trees don't require feature scaling because they use splitting rules rather than distance metrics
   - **Why use LabelEncoder?** Decision trees require numerical inputs, and LabelEncoder efficiently converts categorical text values to integers. Each unique category gets assigned a unique integer

2. **Why default parameters?**
   - **Why no max_depth restriction?** By default, the tree grows until all leaves are pure or contain minimum samples. For this small dataset, overfitting risk is low, so pruning may not be necessary
   - **Why no min_samples_split specification?** The default value (2) allows the tree to create very specific rules, which is acceptable for this demonstration but might need tuning in production
   - **Why Gini impurity by default?** Gini measures the probability of incorrect classification and tends to isolate the most frequent class in its own branch, which is sensible for salary prediction

3. **Why evaluate on training data?**
   - **Why not use test/train split?** This is a simplified example. In practice, you should always evaluate on held-out test data to avoid overestimating performance
   - **Why might perfect accuracy be suspicious?** Perfect accuracy on training data often indicates overfitting, especially with decision trees which can memorize the training data

4. **How Decision Trees work and why they're appropriate:**
   - They recursively split the dataset based on features that best separate the classes
   - At each node, the algorithm selects the feature/threshold that results in the most homogeneous subgroups
   - **Why this is effective:** The splitting process can capture complex decision boundaries and feature interactions
   - **Why trees are interpretable:** Each path from root to leaf represents a clear decision rule (e.g., "if company=Google AND degree=Masters THEN salary>100k")
   - **Why trees automatically handle feature selection:** The algorithm naturally prioritizes informative features and ignores less useful ones

---

## 4. Support Vector Machine (SVM)

### Code Structure:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics

# Load the dataset - social network ad purchase prediction
df = pd.read_csv('Social_Network_Ads.csv')

# Prepare features and target
X = df.iloc[:, [2, 3]]  # Age and Estimated Salary
Y = df.iloc[:, 4]       # Purchased (0/1)

# Split data into training and testing sets
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature scaling - essential for SVM performance
sc_X = StandardScaler()  # Create scaler object
X_Train = sc_X.fit_transform(X_Train)  # Fit scaler on training data & transform
X_Test = sc_X.transform(X_Test)  # Transform test data with training scaler

# 1. Linear kernel SVM
classifier = SVC(kernel='linear', random_state=0)  # Create linear SVM
classifier.fit(X_Train, Y_Train)  # Train the model
Y_Pred = classifier.predict(X_Test)  # Make predictions
print('Accuracy Score with linear kernel:', metrics.accuracy_score(Y_Test, Y_Pred))

# 2. RBF kernel SVM with default parameters
classifier = SVC(kernel='rbf')  # Radial Basis Function kernel
classifier.fit(X_Train, Y_Train)
Y_Pred = classifier.predict(X_Test)
print('Accuracy Score with default rbf kernel:', metrics.accuracy_score(Y_Test, Y_Pred))

# 3. RBF kernel SVM with tuned parameters
classifier = SVC(kernel='rbf', gamma=15, C=7, random_state=0)  # Fine-tuned parameters
classifier.fit(X_Train, Y_Train)
Y_Pred = classifier.predict(X_Test)
print('Accuracy Score with tuned rbf kernel:', metrics.accuracy_score(Y_Test, Y_Pred))

# 4. Polynomial kernel SVM
svc = SVC(kernel='poly', degree=4)  # Polynomial kernel with degree 4
svc.fit(X_Train, Y_Train)
Y_Pred = svc.predict(X_Test)
print('Accuracy Score with poly kernel:', metrics.accuracy_score(Y_Test, Y_Pred))

# Visualization 1: Training data distribution
plt.scatter(X_Train[:, 0], X_Train[:, 1], c=Y_Train)  # Color points by class
plt.xlabel('Age (scaled)')
plt.ylabel('Estimated Salary (scaled)')
plt.title('Training Data')
plt.show()

# Visualization 2: Test data distribution
plt.scatter(X_Test[:, 0], X_Test[:, 1], c=Y_Test)
plt.xlabel('Age (scaled)')
plt.ylabel('Estimated Salary (scaled)')
plt.title('Test Data')
plt.show()

# Visualization 3: Decision boundary (for linear kernel)
classifier = SVC(kernel='linear', random_state=0)  # Recreate linear SVM
classifier.fit(X_Train, Y_Train)  # Retrain
Y_Pred = classifier.predict(X_Test)

# Plot test data points
plt.scatter(X_Test[:, 0], X_Test[:, 1], c=Y_Test)

# Calculate and plot the hyperplane
w = classifier.coef_[0]  # Get weights of the hyperplane
a = -w[0] / w[1]  # Calculate slope
xx = np.linspace(-2.5, 2.5)  # Range of x values to plot
yy = a * xx - (classifier.intercept_[0]) / w[1]  # y = mx + b

# Plot the hyperplane (decision boundary)
plt.plot(xx, yy)
plt.axis("off")
plt.show()
```

### Theoretical Notes:

1. **Why SVM for this problem?**
   - **Why SVM?** SVM is chosen for this binary classification problem because it finds an optimal boundary between classes and handles non-linear relationships through kernels
   - **Why this dataset?** The Social Network Ads dataset involves predicting purchases based on age and salary - these demographic features often have complex relationships with purchasing behavior that SVM can capture well
   - **Why use only Age and Salary columns?** These two features are likely the most predictive for purchase behavior and allow easy visualization in 2D space

2. **Why multiple kernel functions?**
   - **Why try different kernels?** Different kernels transform the feature space in different ways. Testing multiple kernels helps find the one that best captures the underlying pattern
   - **Why linear kernel?** Tests if classes are linearly separable (can be divided by a straight line). It's the simplest model and serves as a baseline
   - **Why RBF kernel?** Radial Basis Function can capture non-linear relationships by mapping to infinite dimensions. It often works well when classes aren't linearly separable
   - **Why polynomial kernel?** Creates polynomial combinations of features, allowing curved decision boundaries. Degree 4 allows complex curves while avoiding extreme overfitting
   - **Why compare accuracy between kernels?** This helps identify which kernel function best captures the underlying pattern in the data

3. **Why feature scaling?**
   - **Why is scaling critical for SVM?** SVMs calculate distances between points, so features with larger scales would dominate smaller ones. Scaling ensures all features contribute equally
   - **Why StandardScaler?** It standardizes features to zero mean and unit variance, which is ideal for SVM which assumes data is centered around origin
   - **Why fit_transform on training but only transform on test?** To prevent data leakage - the test data should be scaled using parameters from training data only

4. **Why parameter tuning?**
   - **Why gamma=15, C=7 for RBF?** These values were likely determined through experimentation:
     - **gamma** controls the influence radius of each support vector (higher = more complex model)
     - **C** is the regularization parameter (higher = less regularization, more complex model)
   - **Why degree=4 for polynomial?** This creates a quartic polynomial, allowing moderately complex curves without extreme overfitting
   - **Why random_state=0?** For reproducibility - ensures the same random initializations each time the code runs

5. **Why these visualization approaches?**
   - **Why visualize both training and test data?** To check if they have similar distributions - large differences might indicate sampling bias
   - **Why plot the decision boundary?** To show how the model separates classes and understand its decision-making logic
   - **Why only show the linear kernel's boundary?** Linear boundaries are easy to visualize as a single line. Non-linear boundaries from RBF or polynomial kernels would require more complex plotting code
   - **Why extract w and calculate a line equation?** For linear SVM, the decision boundary is a hyperplane defined by w·x + b = 0. We extract these parameters to plot the line
   - **Why np.linspace(-2.5, 2.5)?** This creates a range of x values to plot the line - chosen to cover the range of scaled feature values

6. **How SVM works and why these approaches are appropriate:**
   - SVM finds the optimal hyperplane that maximizes the margin between classes
   - Kernels transform data into higher dimensions where linear separation becomes possible
   - **Why this is effective:** Maximum margin ensures best generalization to new data
   - **Why support vectors matter:** Only points near the decision boundary (support vectors) affect its position - making SVM robust to outliers far from the boundary
   - **Why multiple visualization methods help understanding:** Seeing both data distribution and decision boundaries provides insight into how the model works

---

## 5. K-Means Clustering

### Code Structure:
```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import warnings

# Load the dataset - age and income data
df = pd.read_csv("income.csv")

# Visualize raw data
plt.scatter(df.Age, df['Income($)'])  # Simple scatter plot of age vs income
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.show()

# Initial clustering without scaling
km = KMeans(n_clusters=3)  # Create model with 3 clusters (arbitrary initial choice)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])  # Fit model and get cluster assignments
df['cluster'] = y_predicted  # Add cluster labels to dataframe

# Feature scaling - important for K-means performance
scaler = MinMaxScaler()  # Will scale features to 0-1 range
df['Income($)'] = scaler.fit_transform(df[['Income($)']])  # Scale income
df['Age'] = scaler.fit_transform(df[['Age']])  # Scale age

# Clustering after scaling
km = KMeans(n_clusters=3)  # Create model with 3 clusters again
y_predicted = km.fit_predict(df[['Age', 'Income($)']])  # Get cluster assignments
df['cluster'] = y_predicted  # Update cluster labels

# Visualize clusters
df1 = df[df.cluster == 0]  # Filter data for cluster 0
df2 = df[df.cluster == 1]  # Filter data for cluster 1
df3 = df[df.cluster == 2]  # Filter data for cluster 2

# Plot each cluster with different colors
plt.scatter(df1.Age, df1['Income($)'], color='green')
plt.scatter(df2.Age, df2['Income($)'], color='red')
plt.scatter(df3.Age, df3['Income($)'], color='black')
# Add centroids to the plot
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
           color='purple', marker='*', label='centroid')
plt.legend()
plt.show()

# Find optimal number of clusters using elbow method
warnings.filterwarnings("ignore")  # Ignore warnings about convergence
sse = []  # Sum of squared errors (inertia)
k_rng = range(1, 10)  # Test from 1 to 9 clusters
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age', 'Income($)']])
    sse.append(km.inertia_)  # inertia = sum of squared distances to closest centroid

# Plot elbow curve
plt.xlabel('K (number of clusters)')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)
plt.show()
```

### Theoretical Notes:

1. **Why K-Means for this problem?**
   - **Why K-Means?** It's chosen because we want to identify natural groupings in population based on age and income - a classic clustering problem
   - **Why clustering instead of classification?** We don't have predefined labels, so unsupervised learning is appropriate
   - **Why this dataset?** Age and income are continuous variables that often form natural demographic segments, making them suitable for clustering

2. **Why feature scaling?**
   - **Why is scaling crucial for K-means?** K-means uses Euclidean distance measures - without scaling, income (values in thousands) would completely dominate age (values in tens)
   - **Why MinMaxScaler instead of StandardScaler?** MinMaxScaler preserves the shape of the distribution while bringing all values to 0-1 range, making it easier to interpret
   - **Why cluster before and after scaling?** To demonstrate how dramatically scaling affects clustering results - providing a visual justification for this preprocessing step

3. **Why initial parameter choices?**
   - **Why n_clusters=3?** This is an initial guess - clustering often starts with an arbitrary number of clusters before optimization
   - **Why not specify other parameters?** K-means has sensible defaults: 
     - random initialization of centroids
     - maximum 300 iterations
     - 10 different random initializations to find best solution

4. **Why the elbow method?**
   - **Why not just pick a fixed number of clusters?** The optimal number isn't known in advance and should be determined from the data
   - **Why plot SSE vs. k?** As k increases, within-cluster variance always decreases, but with diminishing returns
   - **Why look for an "elbow"?** The point where adding more clusters gives minimal reduction in error indicates a good trade-off between model complexity and fit
   - **Why ignore warnings?** K-means occasionally has convergence issues, but these warnings don't usually affect the overall pattern in the elbow plot

5. **Why visualize clusters and centroids?**
   - **Why different colors for each cluster?** Makes it easy to visually distinguish different segments
   - **Why display centroids?** Shows the "average" point in each cluster - useful for understanding what each cluster represents
   - **Why asterisk markers for centroids?** Makes them stand out from data points, showing their special role

6. **How K-Means works and why it's appropriate:**
   - Randomly initialize k centroids
   - Assign each data point to the nearest centroid
   - Recalculate centroids as the mean of all points in each cluster
   - Repeat until convergence
   - **Why this works:** It minimizes within-cluster variance, creating coherent groups
   - **Why it's efficient:** Simple algorithm with linear complexity in the number of samples
   - **Why it's intuitive:** The concept of grouping by similarity is natural for human understanding

---

## 6. Agglomerative Clustering

### Code Structure:
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Load the dataset - breast cancer data
BreastData = load_breast_cancer()  # Built-in dataset with cancer diagnostic features
x = BreastData.data  # Features - various cell measurements
y = BreastData.data  # Using the same data as "target" for demonstration

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=44)

# Create and fit Agglomerative Clustering model
AggClusteringModel = AgglomerativeClustering(
    n_clusters=2,      # Expecting 2 clusters (benign/malignant)
    affinity="euclidean",  # Distance measure
    linkage='ward'     # Method to calculate distance between clusters
)
y_pred_train = AggClusteringModel.fit_predict(x_train)  # Get cluster assignments for training data
y_pred_test = AggClusteringModel.fit_predict(x_test)    # Get cluster assignments for test data

# Create dendrogram for a subset of training data
dendrogram = sch.dendrogram(sch.linkage(x_train[: 30, :], method='ward'))
plt.title('Training Set Dendrogram')
plt.xlabel('Sample indices')
plt.ylabel('Distances')
plt.show()

# Create dendrogram for a subset of test data
dendrogram = sch.dendrogram(sch.linkage(x_test[: 30, :], method='ward'))
plt.title('Test Set Dendrogram')
plt.xlabel('Sample indices')
plt.ylabel('Distances')
plt.show()

# Visualize training set clusters (using first 2 dimensions)
plt.scatter(x_train[y_pred_train == 0, 0], x_train[y_pred_train == 0, 1], 
           s=10, c='red', label='Cluster 1')
plt.scatter(x_train[y_pred_train == 1, 0], x_train[y_pred_train == 1, 1], 
           s=10, c='blue', label='Cluster 2')
plt.title('Training set clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Visualize test set clusters (using first 2 dimensions)
plt.scatter(x_test[y_pred_test == 1, 0], x_test[y_pred_test == 1, 1], 
           s=10, c='blue', label='Cluster 2')
plt.scatter(x_test[y_pred_test == 2, 0], x_test[y_pred_test == 2, 1], 
           s=10, c='green', label='Cluster 3')
plt.scatter(x_test[y_pred_test == 3, 0], x_test[y_pred_test == 3, 1], 
           s=10, c='cyan', label='Cluster 4')
plt.scatter(x_test[y_pred_test == 4, 0], x_test[y_pred_test == 4, 1], 
           s=10, c='black', label='Cluster 5')
plt.title('Test set clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```

### Theoretical Notes:

1. **Why Agglomerative Clustering for this problem?**
   - **Why hierarchical clustering?** Unlike K-means, agglomerative clustering doesn't require specifying the number of clusters in advance - it builds a tree of clusters that can be cut at any level
   - **Why agglomerative (bottom-up) rather than divisive (top-down)?** Bottom-up approaches often perform better in practice, starting with individual points and merging similar ones
   - **Why for breast cancer data?** Medical data often has natural hierarchical relationships - subtle variations within broader categories of disease states

2. **Why these parameter choices?**
   - **Why n_clusters=2?** Breast cancer diagnosis is binary (benign/malignant), so we expect two natural clusters
   - **Why euclidean distance?** It's a standard distance metric that works well for continuous features like cell measurements
   - **Why ward linkage?** Ward minimizes the variance within clusters, creating compact clusters of similar size - appropriate for medical diagnostic data where we want clearly defined groups
   - **Why random_state=44?** Ensures reproducibility of the train/test split

3. **Why create dendrograms?**
   - **Why visualize the hierarchy?** Dendrograms show how clusters merge and at what distances, revealing the hierarchical