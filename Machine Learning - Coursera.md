# Machine Learning - Coursera

> Lecture 1

Supervised learning, the answer is known

Regression problem - continuous value

Classification problem

Unsupervised learning, ask the algorithm to find a pattern/structure in a set of given data.

## Part I.  Linear regression

>  Lecture 2
>
>  Lecture 4 

### A. Hypothesis

$$
\begin{align}
h_\theta(x) & = \theta_0 + \theta_1x_1 + \cdots + \theta_nx_n\\
            & = \begin{bmatrix} \theta_0 & \theta_1 & \cdots & \theta_n \end{bmatrix} \begin {bmatrix} x_0 \\ x_1 \\ \vdots \\ x_n \end {bmatrix}\\
            & = \theta^Tx
\end {align}
$$

$x_0$ is the intercept term, filled with 1. 

### B. The Cost function

Describe the total difference between the predicted fitting value and the actual data. the function of the parameters.
$$
J(θ)=\frac{1}{2m}∑_{i=1}^m(h_θ(x^{(i)})−y^{(i)})^2
$$

### C. Parameter learning - Gradient descent  

To minimize cost function.
$$
θ_j:=θ_j−α\frac{∂}{∂θ_j}J(θ) \\
θ_j:=θ_j−\frac{α}{m}\sum_{i=1}^{m} (h_\theta(x^{(i)}-y^{(i)})x_{j}^{(i)}\}
$$

"batch" gradient descend. used to find the minimum cost function

- start with a guess for $\theta_0$ and $\theta_1$
- keep changing until local minimum
- "batch" look at the entire training set
- simultaneous update

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/SMSIxKGUEeav5QpTGIv-Pg_ad3404010579ac16068105cfdc8e950a_Screenshot-2016-11-03-00.05.06.png?expiry=1587081600000&hmac=xK1C004DWNMO2pk4pf_ZclPjK2erruOK9rgLRwlFlz8)

__Practices__ 

a. Feature scaling:
$$
\begin{align*}
& x^{(i)}_j = \frac{x^{(i)}_j-\mu_j}{\sigma_j}\\
& \mu_j: \text{mean value of the feature training set}\\
& \sigma_j: \text {range (max-min) or standard deviation of the training set for feature j}
\end{align*}
$$
b. $\alpha$ : learning rate

- if too small, too many steps

- if too large, can overshoot or diverge

- as we approach minimum, the descent will be a smaller step (less steep slope), no need to change the learning rate

c. Polynomial Regression

Features and polynomial regression: use existing parameter to construct additional "features", higher order features

### C. Normal equation

__Design Matrix X__
$$
X = 
\begin{bmatrix} (x^{(1)})^T\\
(x^{(2)})^T\\
\vdots \\
(x^{(m)})^T\\
\end{bmatrix}\\
\text {X is a } m\times(n+1) \text{ matrix} \\
\text {m sets of samples, with n features}
$$
__Normal Equation__
$$
\theta = (X^TX)^{-1}X^Ty
$$

- Derived from setting the derivative of the cost function to 0.

- involve solving matrix inverse, not suitable for too many features n. 

- Feature scaling is not needed

| Gradient Descent      | Normal Equation                     |
| --------------------- | ----------------------------------- |
| Need to choose alpha  |                                     |
| iterations            |                                     |
| $O(kn^2)$             | $O (n^3)$ need to calculate inverse |
| work well for large n | slow if very large                  |
__Non-invertible cause__

- redundant features: closely related features
- too many features, delete (regularization)

## Part *A. Matlab Programming

> Lecture 5

## Part *B: Vectorization

(Mentor) General tips:

- When an equation includes the summation of the product of two vectors, or a vector and a matrix, that's a candidate for a vector math operation.
- Use dimensional analysis. If you know what size the arguments are, and what size you need the result to be, you can easily sort out what order of operands and transpositions are needed.
- You should (probably) never need more than one transposition.

(Personal) Useful Practice:

- element-wise operation
- partial access of the matrix

## Part II. Classification problem

> Lecture 6 

eg. spam and not spam etc.

Threshold classifier: usually applying linear regression is not good for classification problem.

### A. Logistic Regression and Hypothesis 

Binary classification problem - y only take two values.

1. Sigmoid function

$$
\begin{align}
& h_{\theta} = g(\theta^T X) \\
& z = \theta^T X \\
& g(z) = \frac{1}{(1 + e^{-z})}\\
\end{align}
$$

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/1WFqZHntEead-BJkoDOYOw_2413fbec8ff9fa1f19aaf78265b8a33b_Logistic_function.png?expiry=1586995200000&hmac=o6OBhQjBNSrKNlSfjQKy8RtxL_I1fF67ldHejcqVlVo)



2. Output is the probability of y=1, with the given condition of x parameters
   $$
   h_{\theta} = P (y=1|x; \theta) = 1- P (y=0|x; \theta)
   $$

3. Decision boundary 
   __z = 0__ is the decision boundary for $h_\theta$ boundary at 0.5:  derived from the plot of sigmoid function.
   decision boundary is a property of the hypothesis, not the training set. once thetas are decided, the decision boundary is given. 
4. Non-linear decision boundaries
   similar to polynomial regression, use higher order terms.

### B. Logistic Regression Model

1. Cost function
   $$
   J(\theta) = \frac{1}{m}\sum_{i=1}^{m}{Cost(h_\theta, y)}\\
   Cost (h_\theta, y) = 
   \begin{cases}
   -\log(h_\theta(x))	&\text{if y=1}\\
   -\log(1-h_\theta(x)) &\text{if y=0}
   \end{cases}
   $$
   $Cost (h_\theta, y) = \frac{1}{2}(h_\theta - y)^2$ This cost function will be "non-convex" that deteriorate gradient descent
   
   ![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Q9sX8nnxEeamDApmnD43Fw_1cb67ecfac77b134606532f5caf98ee4_Logistic_regression_cost_function_positive_class.png?expiry=1587081600000&hmac=mmwT1MMTMwgDfmyPPHjIYyXihAqdb674PnbEbjlGkog) ![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Ut7vvXnxEead-BJkoDOYOw_f719f2858d78dd66d80c5ec0d8e6b3fa_Logistic_regression_cost_function_negative_class.png?expiry=1586995200000&hmac=FV0FKGHkF8jVgjDqvCZAZ1Nxt6Nn0T_JPgSV1MrYjis)

>  

$$
Cost(h_θ(x),y)=0 \quad \text if \; h_θ(x)=y\\
Cost(h_θ(x),y)→∞ \quad \text if \; y=0 \; and\;h_θ(x)→1\\
Cost(h_θ(x),y)→∞ \quad \text if \; y=1 \;and \;h_θ(x)→0
$$

2. Simplified cost function and gradient descent
   $$
   J(\theta) = \frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h_\theta(x^{(i)})+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))]
   $$
   *Vectorized*
   $$
   h=g(Xθ)\\
   J(θ)=\frac{1}{m}⋅(−y^T\log(h)−(1−y)^T\log(1−h))
   $$
   **Gradient Descent**
   $$
   \begin{align*}
   & \text{Target} \quad min_\theta J(\theta)\\
   & Repeat \; \{θ_j:=θ_j−α\frac{∂}{∂θ_j}J(θ)\}\\
   & Repeat \; \{θ_j:=θ_j−\frac{α}{m}\sum_{i=1}^{m} (h_\theta(x^{(i)})-y^{(i)})x_{j}^{(i)}\}
   \end{align*}
   $$
   *Vectorized*
   $$
   \theta:=\theta - \frac{\alpha}{m}X^T(g(X\theta)-\vec{y})
   $$
   
3. Advanced optimization for minimizing cost (other than gradient descent)

   Optimization algorithms

   - Conjugate gradient
   - BFGS
   - L-BFGS

   Codes

   ```matlab
   
   % we need to write cost function for computing the cost function and the gradients of the cost functions %
   function [jVal, gradient] = costFunction (theta)
   % =============================================
   % then, advanced optimization function can be called upon to solve for local minimum:
   % fminunc: finds a local minimum of a function of several variables.
   
   options = optimset('GradObj', 'on', 'MaxIter', 100);
   initialTheta = zeros(2,1);
   [optTheta, functionVal, exitFlag]=fminunc(@costFunction, initialTheta, options);
   ```

| Advantage                                                    | Disadvantage                |
| ------------------------------------------------------------ | --------------------------- |
| No need to choose \alpha <br />faster than gradient descent (line search) | more complex implementation |

### Multiclass Classification

example: tagging 

__One-vs-all:__

use one boundary to separate each set with the rest, which means we need to train k classifiers for k classes

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/cqmPjanSEeawbAp5ByfpEg_299fcfbd527b6b5a7440825628339c54_Screenshot-2016-11-13-10.52.29.png?expiry=1586995200000&hmac=ArJmYwQI3G3QprBrGUrf6qd14YvMd3z9OBQQUPXl6MA)

Equation
$$
y∈\{0,1...n\}\\
h^{(0)}_θ(x)=P(y=0|x;θ)\\
h^{(1)}_θ(x)=P(y=1|x;θ)\\
⋯\\
h^{(n)}_θ(x)=P(y=n|x;θ)\\
prediction=max_i(h^{(i)}_θ(x))
$$

## Part III: Over-fitting and regularization

> Lecture 7

### A. Over-fitting 

under-fitting (high bias) vs. just right vs. over-fitting (high variance)

- Reduce number of features
  - model selection algorithm
  - manual selection

- Regularization
  - reduce magnitude/ values of $\theta_j$ while keeping all features
  - works well with a lot of features

### B. Regularization

__1. Cost Function__

Smaller parameters means some of the terms of the parameters are less effective -> simpler hypothesis (with lower order equation) -> less prone to over-fitting
$$
\min_\theta \frac{1}{2m} \sum^m_{i=1} {(h_\theta(x^{(i)}-y^{(i)})^2+\lambda \sum^n_{j=1} {\theta^2_j}}
$$
**Regularization term** with __regularization parameter $\lambda $__: compromise between fitting well and keeping the parameter small.

- *$\theta_0$ should not be regularized: subscript in the regularization term starts from 1. 

- Choice of $\lambda$: but if $\lambda$ is very large it might result in a very small value for all the parameter $\theta$ s, cause under-fitting.

__2. Regularized Linear Regression__

Gradient descent
$$
\begin {align}
Repeat \{ \\
        & θ_0:=θ_0−α\frac{1}{m} ∑_{i=1}^{m}{(h_θ(x^{(i)})−y^{(i)})x^{(i)}_0} \\
        & θ_j:=θ_j−α [(\frac{1}{m} ∑_{i=1}^m(h_θ(x^{(i)})−y^{(i)})x^{(i)}_j)+\frac{λ}{m}θ_j]         \quad j∈\{1,2...n\}\\
  \}
\end {align}
$$
Normal Equation
$$
θ=(X^TX+λ⋅L)^{−1}X^Ty \\
where \; L= \begin {bmatrix}
0  & \; &\;     &\;  \\
\; & 1  &\;     &\;  \\
\; & \; &\ddots &\;  \\
\; & \; &\;     &1
\end {bmatrix}
$$
if $\lambda > 0$, the equation also solve the $X^TX$ __non-invertible__ problem.

__3. Regularized Logistic Regression__

Cost function
$$
J(\theta) = \frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h_\theta(x^{(i)})+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))] + \frac{\lambda}{2m}\sum^n_{j=1}\theta^2_j
$$
Gradient descent
$$
\begin {align}
Repeat \{ \\ 
& θ_0:=θ_0−α\frac{1}{m} ∑_{i=1}^{m}{(h_θ(x^{(i)})−y^{(i)})x^{(i)}_0} \\        
& θ_j:=θ_j−α [(\frac{1}{m} ∑_{i=1}^m(h_θ(x^{(i)})−y^{(i)})x^{(i)}_j)+\frac{λ}{m}θ_j]         \quad j∈\{1,2...n\}\\  
\}\end {align}
$$

 