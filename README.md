# Coursera-Machine-Learning
Andrew Ng Machine Learning Course on Coursera

__Platform: Matlab__

## File list

Machine Learning - Coursera.md: personal notes from the course

octave-matlab.md: some commands in matlab, as reference.



## Exercise list and details

### ex1:  Linear regression

ex1.mlx

- uni-variable and multi-variables linear regression cost function
- gradient descent
- feature normalization
- Normal equation

ex1_companion.mlx

- using the Statistics and Machine Learning Toolbox
- `readtable` function for `table` data type
- `fitlm` for linear regression, and `predict`
- "Regression Learner App"

__Lecture Notes__

- Lecture 2
- Lecture 4

### ex2:  Logistic classification

ex2.mlx

- cost function (including sigmoid function) [sigmoid.m; costFunction.m]
- `fminunc` function
- plot decision boundary (have not understand the section on more than 2 features)
- regularization cost function

ex2_companion.mlx

- `fitglm` for generalized linear model
- Classification Learner App
- `fitclinear` with regularization type 'ridge'  

__Lecture Notes__

- Lecture 6
- Lecture 7

### ex3: Multiclass classification with neural networks

ex3.mlx (feedforward propagation)

- representation of image with grayscale intensity
- multi-class using logistic function (with regularization) to generate (train) multiple classifiers (input -> output), and prediction
- neural network: forward propagation prediction using trained weights. 
- (learn the displayData.m)

ex3_companion.mlx

- `fitcecoc` for one-vs-all classification
- train neural network structure //The file used in this section have problem, cannot use. 

__Lecture Note__

- Lecture 8

### ex4: Neural Networks Learning

ex4.mlx

- unroll parameters and `reshape` during calling cost function
- NN forward propagation
- NN cost function and regularization
- NN back propagation for gradient of cost function

ex4_companion.mlx

- Deep Learning Tool box (no installation permit)

**Lecture Note**

- Lecture 9

### ex5: Diagnostic

ex5.mlx

- regularization of linear regression cost function and gradient
- learning curve -> for each training set number, the training error evaluated for the corresponding samples, and the validation error evaluated for the entire cross validation set. 
- constructing polynomial datasets
- select lambda (regularization term) using the cross validation set
- computing test error
- (unsolved)(optional) random iteration for learning curve

ex5_companion.mlx

- 

**Lecture Note**

- Lecture 10

### ex6: Support Vector Machines (and spam classification)

ex6.mlx

- Gaussian kernel
- selecting C and sigma based on cross validation set
- processEmail.m: tidying up email, compare and find the feature word(`strcmp`)
- (optional) use trained model to predict
- (optional) build own database



ex6_companion.mlx



**Lecture Note**

- Lecture 12



### ex7: K-mean and Principal Component Analysis (PCA)

ex7.mlx

**K-means**

- find closest centroid
- move centroid
- random initialization
- image compression using k-means
  - read image in matlab using `imread`
  - image data processing into each pixel represent an example
  - clustering into expected number of color K as centroids
  - you can replace each pixel location with the mean of the centroid assigned to it.

**PCA**

- computing the covariance matrix and its Singular value decomposition (including eigenvector space)
- projection onto the reduce dimension k. 



ex7_companion.mlx



**Lecture Notes**

- Lecture 13
- Lecture 14