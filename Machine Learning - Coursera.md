# Machine Learning - Coursera

Supervised learning, the answer is known

Regression problem - continuous value

Classification problem

Unsupervised learning, ask the algorithm to find a pattern/structure in a set of given data.

## Linear regression

## The cost function

describe the total difference between the predicted fitting value and the actual data. the function of the parameters.

## Gradient descent

"batch" gradient descend. used to find the minimum cost function

- start with a guess for theta0 and theta1
- keep changing until local minimum
- "batch" look at the entire training set
- simultaneous update

**Intuitions** 

1. Feature scaling: (data-means)/(range/standard-deviation)
2. alpha: learning rate

  if too small, too many steps

  if too large, can overshoot or diverge

  as we approach minimum, the descent will be a smaller step (less steep slope), no need to change the learning rate

1. Features and polynomial regression: use existing parameter to construct additional "features", higher order features

## Normal equation

involve solving matrix inverse, not suitable for too many features n. 

Feature scaling is not needed

## Classification problem

eg. spam and not spam etc

threshold classifier

usually applying linear regression is not good for classification problem.

### Logistic Regression and Hypothesis representation

binary classification problem - y only take two values.

1. Sigmoid function

$$
h_{\theta} = g(\theta^T X)\\
z = \theta^T X\\
g(z) = \frac{1}{(1 + e^{-z})}
$$

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/1WFqZHntEead-BJkoDOYOw_2413fbec8ff9fa1f19aaf78265b8a33b_Logistic_function.png?expiry=1586995200000&hmac=o6OBhQjBNSrKNlSfjQKy8RtxL_I1fF67ldHejcqVlVo)



2. Output is the probability of y=1, with the given condition of x parameters
   $$
   h_{\theta} = P (y=1|x; \theta) = 1- P (y=0|x; \theta)
   $$

3. Decision boundary 
   z = 0 is the decision boundary for h-theta boundary at 0.5:  derived from the plot of sigmoid function.
   decision boundary is a property of the hypothesis, not the training set. once thetas are decided, the decision boundary is given. 
4. non-linear decision boundaries
   similar to polynomial regression, use higher order terms.