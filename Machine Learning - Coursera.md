# Machine Learning - Coursera (2011)

> Lecture 1

**Supervised learning,** the answer is known

Regression problem - continuous value

Classification problem

**Unsupervised learning**, ask the algorithm to find a pattern/structure in a set of given data.

# Supervised Learning

## Part I.  Linear regression

>  [[Lecture2-One Variable Linear Regression.pdf]]
>
> [[Lecture4-Multiple Variable Linear Regression.pdf]] 

> ex1

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

### D. Normal equation

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
| Need iterations       |                                     |
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

> ex2: logistic regression
>
> ex3: one vs.all application figure recognition

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

![image-20201026121446066](G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201026121446066.png)

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
   
   ![image-20201026123142472](G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201026123142472.png)

![image-20201026123306383](G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201026123306383.png)
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

### C. Multiclass Classification

example: tagging 

__One-vs-all:__

use one boundary to separate each set with the rest, which means we need to train k classifiers for k classes

Equation
$$
y∈\{0,1...n\}\\
h^{(0)}_θ(x)=P(y=0|x;θ)\\
h^{(1)}_θ(x)=P(y=1|x;θ)\\
⋯\\
h^{(n)}_θ(x)=P(y=n|x;θ)\\
prediction=max_i(h^{(i)}_θ(x))
$$

Train a logistic regression classifier got each class i to predict the probability that y=i.

To make prediction for new input x, generate the hypothesis of x for all classifiers with the theta, pick the class that maximize the hypothesis.

## Part C: Over-fitting and regularization

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

 ## Part III: Non-linear Hypothesis and Neural Networks

> Lecture 8
>
> Lecture 9

> ex3
>
> ex4: application figure recognition

### A. Neural Network Model Representation and forward propagation

__1. Neuron model.__

>  eg. neuron with sigmoid (logistic) activation function

$$
\begin{bmatrix} x_0 \\ x_1\\ x_2\\ x_3 \end{bmatrix} \to \begin{bmatrix} a_1^{(2)} \\ a_2^{(2)}\\ a_3^{(2)}\end{bmatrix} \to h_\theta(x)
$$



input layer-> hidden layer(s) -> output layer

$x_0$: bias unit

$a_j^{(i)}$ = "activation" of unit i in layer j

$\Theta^{(j)}$ = matrix of "weights" (or "parameter") controlling function mapping from layer j to layer j+1, 

- dimension: If network has $s_j$ units in layer *j* and $s_{j+1}$ units in layer *j*+1, then $Θ^{(j)}$ will be of dimension $ s_{j+1} \times (s_j+1) $
- index of element (target-source)

__2. Vectorized implementation__

Forward propagation (General)
$$
\begin{align}
&\text{Compute layer j}\\
& z^{(j)} = \Theta^{(j-1)}a^{(j-1)} \quad \text{weighted sum}\\
& a^{(j)} = g(z^{(j)})\\
\\
&\text{Compute next (output) layer}\\
& Add \quad a_0^{(j)} = 1\\
& z^{(j+1)} = \Theta^{(j)}a^{(j)}\\
& h_\Theta(x) = a^{(j+1)} = g(z^{(j+1)})
\end{align}
$$
__3. Neural Architectures__

> How the neurons are connected

__4. Example: Non-linear classification example: XOR/XNOR__

__Intuition I__

![image-20201026150514648](G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201026150514648.png)

![image-20201026150545567](G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201026150545567.png)

__Intuition II__

> XOR: true only if a or b true
>
> XNOR == NOT( a XOR b)
>
> ![image-20201026151600583](G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201026151600583.png)

__5. Multiclass Classification: One-vs-all__

![image-20201026151733598](G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201026151733598.png)

multiple output units, represent y as vectors.
$$
y^{(i)} = 
\begin{bmatrix} 1\\ 0\\ 0\\ 0\\ \end{bmatrix}
\begin{bmatrix} 0\\ 1\\ 0\\ 0\\ \end{bmatrix}
\begin{bmatrix} 0\\ 0\\ 1\\ 0\\ \end{bmatrix}
\begin{bmatrix} 0\\ 0\\ 0\\ 1\\ \end{bmatrix}
$$

### B. NN cost function and back propagation

__1. Cost function__
$$
h_\Theta(x) \in R^K  \quad (h_\Theta(x))_i=i^{th} output \\
\begin{align}
J(\Theta) 
&= \frac{1}{m}\sum_{(i=1)}^m\sum_{(k=1)}^K [y_k^{(i)}\log{(h_\Theta(x^{(i)}))}_k
+(1-y_k^{(i)})\log{(1-(h_\Theta(x^{(i)}))_k)}]\\
& +\frac{\lambda}{2m}\sum_{(l=1)}^{L-1}\sum_{(i=1)}^{s_l}\sum_{(j=1)}^{s_{l+1}}(\Theta^{(l)}_{ij})^2
\end{align}
$$

- L = total number of layers

- $s_l$ = number of units in layer l (exclude bias unit)

- K = number of output units/classes
- the double sum simply adds up the logistic regression costs calculated for each cell in the output layer
- the triple sum simply adds up the squares of all the individual Θs in the entire network.
- the i in the triple sum does **not** refer to training example i. 

__2. Backpropagation Algorithm__
$$
\delta_j^{(4)} = a_j^{(4)}-y_j\\
\delta_j^{(3)} =(\Theta^{(3)})^T\delta^{(4)}.*g'(z^{(3)})\\
\delta_j^{(2)} =(\Theta^{(2)})^T\delta^{(3)}.*g'(z^{(2)})\\
$$
$\delta_j^{(l)}$ = "error" of node j in layer l. = $\frac{\partial}{\partial z_j^{l}}cost(i)$

The partial derivative of $ J(\Theta)$ is needed for minimum cost function.
$$
\frac{∂}{∂Θ^{(l)}_{i,j}}J(Θ)
$$
Given training set ${(x^{(1)},y^{(1)})⋯(x^{(m)},y^{(m)})}$

- Set $\Delta^{(l)}_{i,j}$:= 0 for all (l,i,j), (hence you end up having a matrix full of zeros)

For training example t =1 to m:

1. Set $a^{(1)} := x^{(t)}$

2. Perform __forward propagation__ to compute $a^{(l)}$ for l=2,3,…,L

   **-> back propagate**

3. Using $y^{(t)}$, compute $\delta^{(L)} = a^{(L)} - y^{(t)}$

4. Compute $\delta^{(L-1)}, \delta^{(L-2)},\dots,\delta^{(2)}$

   > The delta values of layer l are calculated by multiplying the delta values in the next layer with the theta matrix of layer l.

   $$
   δ^{(l)}=((Θ^{(l)})^Tδ^{(l+1)}) .∗ a^{(l)} .∗ (1−a^{(l)})\\
   \text{where } g'(z^{(l)}) = a^{(l)}.*(1-a^{(l)})\\
   $$

5. $Δ_{i,j}^l:=Δ_{i,j}^l+a_{j}^l\delta_{i}^{(l+1)}$ or with vectorization, $\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T$
$$
D^{(l)}_{i,j}:=\frac{1}{m}(Δ^{(l)}_{i,j}+λΘ^{(l)}_{i,j}), \; if \;j≠0\\
   D^{(l)}_{i,j}:=\frac{1}{m}(Δ^{(l)}_{i,j}), \; if \; j=0
$$

> The capital-delta matrix D is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. 

$$
\frac{∂}{∂Θ^{(l)}_{i,j}}J(Θ) = D_{i,j}^{(l)}
$$
__Intuitions__

back of forward propagation

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/qc309rdcEea4MxKdJPaTxA_324034f1a3c3a3be8e7c6cfca90d3445_fixx.png?expiry=1587513600000&hmac=NSYzEDGr7QqumIljqQNvBP2gO_6B-NgOvifQN4AORTQ)

### C. Implementation of Back Propagation

__1. Unrolling (matrices and vector representation)__ 

"unroll" matrices into one big vector

```matlab
thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
deltaVector = [ D1(:); D2(:); D3(:) ]
```

`reshape` to get back to matrices

```matlab
Theta1 = reshape(thetaVector(1:110),10,11)
Theta2 = reshape(thetaVector(111:220),10,11)
Theta3 = reshape(thetaVector(221:231),1,11)
%  reshape(X,M,N) or reshape(X,[M,N]) returns the M-by-N matrix whose elements are taken columnwise from X.
```

![image-20201026160328184](G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201026160328184.png)

__2. Gradient Checking__
$$
\frac{∂}{∂Θ}J(Θ)≈\frac{J(Θ+ϵ)−J(Θ−ϵ)}{2ϵ}  \\
\frac{∂}{∂Θ_j}J(Θ)≈\frac{J(\Theta_1,\dots,Θ_j+ϵ,\dots,Θ_n)−J(\Theta_1,\dots,Θ_j-ϵ,\dots,Θ_n)}{2ϵ}
$$

- this compares the calculated cost function gradient with an infinitesimally approximated gradient
- check if the derivative function is outputting correct derivative values.
- two sided is slightly more accurate than one sided
- $\epsilon$ recommended to use $~10^{-4}$
- should see a relative difference that is less than 1e-9

codes:

```matlab
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```

- Once you have verified **once** that your backpropagation algorithm is correct, you don't need to compute gradApprox again. The code to compute gradApprox can be very slow (calculating all costs).

__3. Random initialization__

> Initializing all theta weights to zero does not work with neural networks. When we back propagate, all nodes will update to the same value repeatedly. problem of symmetric weights.

 __Symmetry breaking__

```matlab
If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

- `rand(10,11)` random 10 *11 matrix. rand(x,y) is just a function in octave that will initialize a matrix of random real numbers between 0 and 1.
- *One effective strategy for choosing  is to base it on the number of units in the network. A good choice of $\epsilon_{init}$ is  $\epsilon_{init}=\frac{\sqrt{6}}{\sqrt{L_{in}+L_{out}}}$where  $L_{in}=s_l, L_{out}=s_{l+1}+1$ are the number of units in the layers adjacent to $\Theta^{(l)}$ .

__4.Putting it together__

__a. Pick a network architecture__

- input number: dimension feature
- output: number of class
- reasonable default: (1) 1 hidden layer (2) if more than 1, have same number of units in different layers

__b. Training __

(1) random initialize weights

(2) forward prop to get $h_\theta(x^{(i)})$ for any $x^{(i)}$

(3) code to compute cost function $J(\Theta)$

(4) backprop to compute partial derivatives $\frac{\partial}{\partial\Theta^{(l)}_{jk}}$

(5) gradient checking, then disable gradient check code

(6) minimize $J(\Theta)$ (non-convex), using gradient descent or advanced optimization

 ```
for i = 1:m,
   Perform forward propagation and backpropagation using example (x(i),y(i))
   (Get activations a(l) and delta terms d(l) for l = 2,...,L
 ```

### D. Programming exercise note

**back propagation vectored implementation**

https://www.coursera.org/learn/machine-learning/discussions/all/threads/a8Kce_WxEeS16yIACyoj1Q

Let:

m = the number of training examples

n = the number of training features, including the initial bias unit.

h = the number of units in the hidden layer - NOT including the bias unit

r = the number of output classifications

\-------------------------------

1: Perform forward propagation, see the separate tutorial if necessary.

2: $\delta_3$ or d3 is the difference between a3 and the y_matrix. The dimensions are the same as both, (m x r).

3: z2 comes from the forward propagation process - it's the product of a1 and Theta1, prior to applying the sigmoid() function. Dimensions are (m x n) \cdot⋅ (n x h) --> (m x h). In step 4, you're going to need the sigmoid gradient of z2. From ex4.pdf section 2.1, we know that if u = sigmoid(z2), then sigmoidGradient(z2) = u .* (1-u).

4: $\delta_2$ or d2 is tricky. It uses the (:,2:end) columns of Theta2. d2 is the product of d3 and Theta2 (without the first column), then multiplied element-wise by the sigmoid gradient of z2. The size is (m x r) \cdot⋅ (r x h) --> (m x h). The size is the same as z2.

Note: Excluding the first column of Theta2 is because the hidden layer bias unit has no connection to the input layer - so we do not use backpropagation for it. See Figure 3 in ex4.pdf for a diagram showing this.

5: $\Delta_1$ or Delta1 is the product of d2 and a1. The size is (h x m) \cdot⋅ (m x n) --> (h x n)

6: $\Delta_2$ or Delta2 is the product of d3 and a2. The size is (r x m) \cdot⋅ (m x [h+1]) --> (r x [h+1])

7: Theta1_grad and Theta2_grad are the same size as their respective Deltas, just scaled by 1/m.

## Part D: Choosing and evaluation effectively

> ex5

### A. Evaluating a learning algorithm

What to try if it doesn't work

- get more training examples
- try smaller sets of features
- try getting additional features
- try adding polynomial features
- try increase or decrease lambda

**1. Machine learning diagnostic**

the implementation takes time

rule out a or suggest an action to take.

**2. Evaluating a Hypothesis**

1. split the dataset into training set (70%) and test set (30%)

2. learn parameter theta (minimize training error)

3. compute test set error (save definition of the cost function, only with test error)

   - Linear regression $J_{test}(θ)=\frac{1}{2m_{test}}∑_{i=1}^m(h_θ(x_{test}^{(i)})−y_{test}^{(i)})^2$

   - For classification ~ Misclassification error (aka 0/1 misclassification error):
     $$
     err(h_{\Theta}(x),y) = 
     \begin{cases}
     1	&{if \: h_{\Theta}(x) \geq 0.5 \: and \: y=0 \: or \:  h_{\Theta}(x) < 0.5 \: and \: y=1 \: }\\
     0   &\text{otherwise}
     \end{cases}
     $$
     The average test error
     $$
     \text{Test Error} = \frac{1}{m_{test}}\sum_{i=1}^{m_{test}} err(h_{\Theta}(x_{test}^{(i)}),y_{test}^{(i)})
     $$

**3. Model Selection and Train/Validation/Test sets**

Training 60%/ cross validation 20%/ test set 20%

1. Optimize the parameters in Θ using the **training set** for each polynomial degree. The set used for learning.
2. Find the polynomial degree d with the least error using the **cross validation set**.  Used to determine the regularization parameter.
3. Estimate the generalization error using the **test set** with $J_{test}(\Theta^{(d)})$, (d = theta from polynomial with lower error);

### B. Bias (underfit) and variance (overfit) problem

**1. Diagnostic**

- We need to distinguish whether **bias** or **variance** is the problem contributing to bad predictions.
- High bias is underfitting and high variance is overfitting. Ideally, we need to find a golden mean between these two.

![image-20201026185811541](G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201026185811541.png)

**High bias (underfitting)**: both $J_{train}(\Theta)$and $J_{CV}(\Theta)$ will be high. Also, $J_{CV}(\Theta) \approx J_{train}(\Theta)$

**High variance (overfitting)**: $J_{train}(\Theta)$ will be low and $J_{CV}(\Theta)$ will be much greater than $ J_{train}(\Theta)$

**2. Regularization** (lambda)

choosing lambda

1. Create a list of lambdas (i.e. λ∈{0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12,10.24});
2. Create a set of models with different degrees or any other variants.
3. Iterate through the λs and for each λ go through all the models to learn some Θ.
4. Compute the cross validation error using the learned Θ (computed with λ) on the $J_{CV}(\Theta)$**without** regularization or λ = 0. (and the training error without regularization) 
5. Select the best combo that produces the lowest error on the cross validation set.
6. Using the best combo Θ and λ, apply it on $J_{test}(\Theta)$ to see if it has a good generalization of the problem.

![image-20201026190834204](G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201026190834204.png)

**3. Learning curve**

plot error as a function of m training set sizes.

for each training set size, the training error evaluated for the corresponding samples, and the validation error evaluated for the entire cross validation set. 

**Experiencing high bias:**

**Low training set size**: causes $J_{train}(\Theta)$ to be low and $J_{CV}(\Theta)$ to be high.

**Large training set size**: causes both $J_{train}(\Theta)$ and $J_{CV}(\Theta)$ to be high with $J_{train}(\Theta)≈J_{CV}(\Theta)$

> If a learning algorithm is suffering from **high bias**, getting more training data will not **(by itself)** help much.

![image-20201026191337879](G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201026191337879.png)

**Experiencing high variance:**

**Low training set size**: $J_{train}(\Theta)$ will be low and $J_{CV}(\Theta)$ will be high.

**Large training set size**: $J_{train}(\Theta)$ increases with training set size and $J_{CV}(\Theta)$ continues to decrease without leveling off. Also,$ J_{train}(\Theta) < J_{CV}(\Theta)$but the difference between them remains significant.

> If a learning algorithm is suffering from **high variance**, getting more training data is likely to help.

![image-20201026191427506](C:\Users\echoz\AppData\Roaming\Typora\typora-user-images\image-20201026191427506.png)

**4. Debugging a learning algorithm**

| **Fix high variance**        | **Fix high bias**                                            |
| ---------------------------- | ------------------------------------------------------------ |
| get more training examples   |                                                              |
| try smaller sets of features | try getting additional features<br />try adding polynomial features |
| try increase lambda          | try decrease lambda                                          |

**Size of NN**

| size (no. parameters; no. layers) | pro                                               | characteristics                         |
| --------------------------------- | ------------------------------------------------- | --------------------------------------- |
| small                             | computationally inexpensive                       | fewer parameters; prone to underfitting |
| large                             | expensive<br />use regularization for overfitting | more parameters; prone to overfitting   |

**Model Complexity Effects:**

- Lower-order polynomials (low model complexity) have high bias and low variance. In this case, the model fits poorly consistently.
- Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. These have low bias on the training data, but very high variance.

## Part E: ML System Designing and possible issues

### Prioritizing and Error analysis 

- start with simple algorithm
- plot learning curve to determine what is needed
- error analysis: **manually** examine the examples (*in cross validation sets*) that your algorithm made errors on.

**importance of numerical evaluation**: 

It is very important to get error results as a **single, numerical value**  to assess your algorithm's performance through comparison.

### Handling skewed class

**Skewed class:** 

- much more examples in one class than the other,

- resulting a high accuracy if predict on the large data set, but may not suggest that it is a good model. 

**Precision/Recall** (two error metrics) (of cross validation set)

tp: true positive

fp: false positive

tn: true negative

fn: false negative

Precision: 

> true positive out of all the predicted positives

$$
\frac{tp}{tp+fp}
$$

Recall:

> predicted correct positive out of all the actual positive

$$
\frac{tp}{tp+fn}
$$





y = 1 for rare class detection. 

| Precision                                                    | **Recall**                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| high value desired                                           | high value desired                                           |
| true positives/ predicted positives = true positives/ (true positive+ false positives) | true positives/ actual positives = true positives/ (true positive+ false negative) |

**Trading off between precision and recall**

*adjusting the threshold*

toward higher threshold => classifier with higher precision, lower recall.

towards lower threshold => classifier with higher recall, lower precision.

**F_1 score**

Choosing threshold according to one metric from precision and recall
$$
F_1 \: score: 2\frac{PR}{P+R}
$$

### Large Data Sets

1. assume feature x has sufficient information to predict y accurately. (eg. human expert judgement)

2. train with a large number of parameters

## Part IV: Support Vector Machines

> Lecture 12

> ex6: application on spam classification

### A. Hypothesis and cost function

Hypothesis:
$$
h_\theta(x)=\begin{cases}
1	&\text{if }\theta^{T}x\geq 0\\
0   &\text{otherwise}
\end{cases}
$$
cost function
$$
min_{\theta}C\sum_{i=1}^m[y^{(i)}cost_1(\theta^Tx^{(i)})+(1-y^{(i)})cost_0(\theta^Tx^{(i)})]+\frac{1}{2}\sum_{i=1}^n\theta^2_j
$$

**Large Margin Intuition and Decision Boundary**

![image-20201026200314685](G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201026200314685.png)

if y=1, we want $ \theta^Tx \geq 1$  (not just >= 0)

if y=0, we want $ \theta^Tx \leq 1$  (not just <= 0)

* The SVM would choose a better decision boundary because that one has larger margin (distance between the sample edge), thus giving it better robustness.  Separating the positive and negative example with largest margin possible. 

![image-20201026200509356](G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201026200509356.png)

### B. Kernels

choosing the landmarks l to **define the kernels**

The new features created as following similarity kernel function  **(Gaussian Kernels here)**
$$
f_1 = similarity (x, l^{(1)}) = \exp(-\frac{||x-l^{(1)}||^2}{2\sigma^2}) = \exp(-\frac{\sum^n_{j=1}(x_j-l_j^{(1)})^2}{2\sigma^2})
$$
when x close to l, f1 = 1; when x far from l, f1 =0.

The Gaussian kernel is also parameterized by a bandwidth parameter, $\sigma$ , which determines how fast the similarity metric decreases (to 0) as the examples are further apart. The effect of sigma: if sigma squared is large, then as you move away from l_1, the value of the feature falls away much more slowly.

![image-20201026201032255](G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201026201032255.png)

**Choose the landmarks** at exactly the location of my training examples, end up with m landmarks. 

number of features n is same as the number of the training examples m, because of the way of definition.

#### SVM parameters

!  use the cross validation set Xval, yval to determine the best  and  parameter to use

1. $C= \frac{1}{\lambda}$

- large C: lower bias, high variance (small lambda)
- small C: higher bias, low variance

2. $\sigma^2$

- large $\sigma^2$: features fi vary more smoothly. -> higher bias, lower variance.
- small $\sigma^2$: features fi vary less smoothly. -> lower bias, higher variance.

### C. Using an SVM

**SVM software package**

need to specify

- choice of parameter C

- choice of kernel (similarity function)

  - eg. no kernel ("linear kernel")

  - Gaussian kernel (need to choose \sigma^2)

    choose when you have a pretty large training set with nonlinear decision boundary

    need to perform feature scaling before using the Gaussian kernel (avoid being dominated by the value itself)
    
  - Not all similarity functions make valid kernels
  
  - off the shelf: polynomial kernels, string kernel, chi-square kernel, histogram intersection kernel. 

### D. Logistic regression vs. SVMs

n = number of features; m = number of training examples

If n is larger, use logistic or SVM without kernel

if n is small, m is intermediate, use SVM with Gaussian kernel

if n is small, m is large, create/add more features, the use logistic regression or SVM without  kernel.   



# Unsupervised Learning

finding pattern in a set of data

## Part V: Clustering : K_Means Algorithm

> Lecture 13

> ex.7 application in image compression

1. Inputs

   - K(number of clusters)
   - Training set (without labels), drop x0=1 as convention, therefore n samples

2. Algorithm

   ```matlab
   Randomly initialize K cluster centroids
   
   Repeat 
   {
   
   %% A. (cluster assignment step)
   
      for i = 1 to m
   
          c(i) = index (from 1 to K) of cluster centroid mu_k closest to x(i)
          => c(i)=j for the min ||x(i) - mu(j)||^2
   
   %% B. (move centroid step)
   
      for k = 1 to K
   
           mu_k = average (mean) of points assigned to cluster k
   }
   ```

   

3. non-separated clusters

### A. K-means optimization objective

$c^{(i)}$ = index of cluster (1,2,...K) to which example $x^{(i)}$ is currently assigned

$\mu_k$ = cluster centroid k

$\mu_c^{(i)}$ = cluster centroid of cluster to which example $x^{(i)}$ has been assigned

**cost function:**
$$
J(c^{(1)}, \dots,c^{(m)},\mu_1,\dots,\mu_K) = \frac{1}{m}\sum^m_{i=1}||x^{(i)}-\mu_c^{(i)}||^2 \\
$$
**optimization objective: minimize cost function** 
$$
min_{c^{(1)}, \dots,c^{(m)},\mu_1,\dots,\mu_K}J(c^{(1)}, \dots,c^{(m)},\mu_1,\dots,\mu_K)
$$

### B. Random Initialization

Should have K<m, pick K training examples and set them as the initial centroids

To avoid sticking in local minima, try multiple random initialization is a possible approach -> pick clustering that gave the lowest cost function (distortion). 

### C. Choosing the number of clusters

common is manually from inspection of the samples. 

- Elbow method

  number of cluster (x) against cost function J, choose the value at the "elbow" seems reasonable

  ![image-20201027192321447](G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201027192321447.png)

- evaluate K-means based on how it performs for the later/downstream purpose

## Part VI: Dimensionality Reduction: PCA (Principal component Analysis)

> Lecture 14

### A. Motivations

**1.  Data Compression**

We apply dimensionality reduction to a dataset of m examples $\{x^{(1)}, x^{(2)}, \dots, x^{(m)}\}, where x^{(i)}\in\mathbb{R}^n$, and we will get a lower dimensional dataset $\{z^{(1)}, z^{(2)},\dots, z^{(m)}\} $ of m examples where $z^{(i)} \in \mathbb{R}^k$for some value of k and $k\leq n$ 

**2. Data Visualization**

Reduce to 2D or 3D so that a plotting is possible

### B. Principal Component Analysis (PCA) Algorithm

> PCA is not linear regression (minimum projection "path" but not minimum cost (difference between y and hypothesis))

#### 1. Problem formulation

Reduce from n-dimension to k-dimension: find k vectors u1, u2, ..., uk ( linear subspace spanned by this set of k vectors) onto which to project the data, so as to minimize the projection error. 

#### 2. PCA Algorithm

**Preprocessing (feature scaling/mean normalization)** 

a. mean normalization (each feature will then have exactly zero mean)
$$
\mu_j = \frac{1}{m}\sum^m_{j=1}x_j^{(i)}
$$
   replace each $x_j^{(i)}$ with $x_j-\mu_j$

b. feature scaling: scale features to have comparable range of values.

**Algorithm**

a. Compute "covariance matrix"
$$
\Sigma = \frac{1}{m}\sum^n_i{(x^{(i)})(x^{(i)})^T}
$$
​    Vectorized implementation

```matlab
Sigma = (1/m)*X'*X;
```

b. Compute "eigenvectors" of matrix $\Sigma$

```matlab
[U,S,V] = svd(Sigma);
```

- svd: Singular value decomposition, U will contain the principal components and S will contain a diagonal matrix
- Sigma is n\*n matrix (from definition ${(x^{(i)})(x^{(i)})^T}$ is a n*n matrix)
- U matrix is also n*n matrix, the columns are the vectors => if we want to reduce to k, we just use the first k columns of the U matrix

c. take the first k columns of the U matrix to form U_reduce for projection

```matalb
U_reduce = U(:, 1:k);
```

d. compute Z

Project each example in X onto the top K components in U and get the new projected dataset z.

```matlab
z = U_reduce' * x
```

### C. Applying PCA

#### 1. Reconstruction from Compressed Representation

```matlab
X_approx = U_reduce * z;
```

#### 2. Choose the number of principal components k

**a. "99% of variance is retained"**

typically, choose k to be smallest value so that
$$
\frac{\frac{1}{m}\sum^m_{i=1}{||x^{(i)} - x^{(i)}_{approx}||^2}}{\frac{1}{m}\sum^m_{i=1}{||x^{(i)} ||^2}} \leq 0.01
$$

> "99% of variance is retained"

- numerator: average squared projection error(, which the PCA aims to minimize)

- denominator: total variation in the data 
  $$
  \frac{1}{m}\sum^m_{i=1}{||x^{(i)} ||^2}
  $$
  

**b. Procedure**

I. try PCA from k=1, and check the fraction

II. `[U, S, V] = svd(Sigma`), the target fraction can be computed from the S matrix

for given k, check
$$
1-\frac{\sum^k_{i=1}S_{ii}}{\sum^n_{i=1}S_{ii}} \leq 0.01
$$

#### 3. Advice on application

**Example: Supervised learning speedup**

 (x1,y1),  (x2,y2), ...,  (xm,ym),

Step 1: Extract input (without output)

Step 2: PCA

Step 3: New training set: (z1,y1),  (z2,y2), ...,  (zm,ym), (with much less features)

Note: Mapping (U_reduce) x(i) -> z(i) should be defined by running PCA only on training set. Then, this mapping can be applied as well to cross validation and test sets

**Application**

- compression
  - Reduce memory/disk needed to store data
  - speed up learning algorithm (choose k by % of variance retain)
- Visualization -> usually k=2 or 3

**Bad use, misuse: **

- to prevent overfitting

  fewer feature thus less likely to overfit

  might be OK, but isn't a good way, use regularization instead

  because PCA ignore y values, thus is like to throw away valuable information

- PCA is sometimes used when it shouldn't be 

  Instead of making plan to use PCA in advance, before you implement  PCA, first do with the original raw data xi, and only if that doesn't do what you want, then implement PCA and consider using zi.




## Part VII: Anomaly Detection (Density Estimation)

### A. Motivation: Anomaly Detection

example: engine features

new sample, is it close to the dense of the rest of examples

p(x_test) < e -> flag anomaly

Examples

- fraud detection: 
  - x(i) = features of user i activities
  - model p(x) from data
  - identify unusual user through checking p(x) < e
- Manufacturing
  - x(i) = features of machine i

### *Gaussian (Normal) Distribution 

Gaussian distribution with mean $\mu$, variance $\sigma^2$, standard deviation $\sigma$ 
$$
p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma}\exp{(-\frac{(x-\mu)^2}{2\sigma^2})}
$$

$$
\mu = \frac{1}{m}\sum^m_{i=1}{x^{(i)} }\\

\sigma^2=\frac{1}{m}\sum^m_{i=1}({x^{(i)}-\mu)^2}
$$

### B. Algorithm

1. choose features $x_i$ that might be indicative of anomalous examples

2. fit parameters $\mu$ and $\sigma^2$ from i to n features
   $$
   \mu_j = \frac{1}{m}\sum^m_{i=1}{x_j^{(i)} }\\
   \sigma_j^2=\frac{1}{m}\sum^m_{i=1}({x^{(i)}_j-\mu_j)^2}
   $$

3. Given new examples x, compute $p(x)$:
   $$
   p(x) = \prod_{j=1}^np(x_j;\mu_j,\sigma^2_j)=\prod^n_{j=1}\frac{1}{\sqrt{2\pi}\sigma_j}\exp{(-\frac{(x_j-\mu_j)^2}{2\sigma^2_j})}
   $$
   Anomaly if $p(x)<\varepsilon$ 

### C. Development and Evaluation

Real number evaluation

1. Sample sets

   Training set: majority normal

   CV, Test:  normal mixed with anomaly (labeled data)

2. Evaluation

   The y will be **skewed class**, for the evaluation metrics

- true positive, false positive, false negative, true negative
- precision/recall
- F_1- score

3. Threshold choice

   Can also use cross validation sets to choose $\varepsilon$

### D. Anomaly Detection vs. Supervised Learning

| Anomaly Detection                                            | Supervised Learning                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Very small number of positive examples; large number of negative examples | large number of positive and negative examples               |
| many different "types" of anomalies.  Hard for any algorithm to learn from positive examples what the anomalies look like; future anomalies may look nothing like existing | Enough positive examples for algorithm to get a sense of what positive examples are, as well for future ones. |
| fraud detection; manufacturing; monitoring machines in a data center | spam; weather prediction; cancer classification              |

### E. Choosing features to use

**Feature data**

1. plot a histogram and investigate the distribution

   `hist(x)` in matlab. 

2. non-gaussian features -> use different transformation to make it more gaussian

**Error analysis**

Want p(x) large for normal examples and small for anomalous examples

- most common problem:  is when p(x) is comparable for both normal and anomalous examples. => look at the anomalous example and find new features that could distinguish

**Choose features** by investigating the operations and maybe create new features from the relation

### F. Multivariate Gaussian Distribution 

instead of model p(x1), p(x2), ... , etc. separately

Model p(x) all in one go.

Parameters $\mu \in \real^n$ , $\Sigma \in \real^{n\times n}$
$$
p(x;\mu,\Sigma) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\exp{(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))}
$$
Parameter fitting
$$
\mu = \frac{1}{m}\sum^m_{i=1}{x^{(i)} }\\

\Sigma=\frac{1}{m}\sum^m_{i=1}{(x^{(i)}-\mu)(x^{(i)}-\mu)^T}
$$

- Variating the covariance matrix will change the shape of density plot 
- variation of the mu will change the location of the peak of the distribution

<img src="G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201027200731129.png" alt="image-20201027200731129" style="zoom:50%;" /> <img src="G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201027200946226.png" alt="image-20201027200946226" style="zoom:50%;" />

**Apply to anomaly detection**

1. fit model p(x) by fitting $\mu$ and $\Sigma$

2. Given a new example x, compute p(x)

   Flag anomaly if $p(x) < \epsilon$

**Compare to original**

| Original                                                     | Multivariate                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| manually create feature when x1 x2 take unusual combinations of values, x3=x1/x2 | automatically capture correlations between features          |
| Computationally cheaper (scales better to large n)           | computationally more expensive                               |
| OK even if m (training set size) is small                    | must have m>n or else the covariant matrix is non-invertible; also with redundant feature (should be linearly independent), the covariant matric may also be non-invertible |

## Part VIII: Recommender Systems

> Lecture 16

**Example: movie rating prediction**

$n_u$ = no. of users

$n_m$ = no. of movies

$r(i, j) = 1$ if user j has rate the movie i

$y^{(i,j)}$ = rating given by user j to movies i, (defined only if $r(i,j)=1$)

### A. Content based recommendations (feature based)

a. features defined as the degree of content related to a certain type

$r(i, j) = 1$ if user j has rate the movie i

$y^{(i,j)}$ = rating given by user j to movies i, (defined only if $r(i,j)=1$)

$\theta^{(j)}$ = parameter vector for user j

$x^{(i)}$ = feature vector for movie i

For user j, movie i, predicted rating $(\theta^{(j)})^T(x^{(i)})$

$m^{(j)}$ = no. of movies rated by user j

**Optimization objective (cost function)**

given $x^{(1)}, \dots, x^{(n_m)}$ to learn $\theta^{(1)}, \theta^{(2)}, \dots, \theta^{(n_u)}$ for user $1, 2, ..., n_u$
$$
min_{\theta^{(1)}, \dots, \theta^{(n_u)},}\frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j=1)}((\theta^{(j)})^T(x^{(i)})-y^{(i-j)})^2+\frac{\lambda}{2}\sum^{n_u}_{j=1}\sum_{k=1}^n(\theta^{(j)}_k)^2
$$

**Optimization algorithm**

gradient descent update

### B. Collaborative Filtering

#### 1. feature learning

given $\theta^{(1)}, \theta^{(1)}, \dots, \theta^{(n_u)}$, to learn $x^{(1)}, \dots, x^{(n_m)}$
$$
min_{x^{(1)}, \dots, x^{(n_m)}}\frac{1}{2}\sum_{i=1}^{n_m}\sum_{j:r((i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum^{n_m}_{i=1}\sum^{n}_{k=1}(x_k^{(i)})^2
$$

#### 2.Collaborative filtering optimization objective

minimizing $\theta^{(1)}, \theta^{(2)}, \dots, \theta^{(n_u)}$ and  $x^{(1)}, \dots, x^{(n_m)}$ simultaneously

Guess theta->x->theta->x
$$
J(x^{(1)}, \dots, x^{(n_m)}, \theta^{(1)}, \dots, \theta^{(n_u)}) = \sum_{(i,j):r((i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum^{n_m}_{i=1}\sum^{n}_{k=1}(x_k^{(i)})^2+\frac{\lambda}{2}\sum^{n_u}_{j=1}\sum_{k=1}^n(\theta^{(j)}_k)^2 \\
min_{x^{(1)}, \dots, x^{(n_m)}, \theta^{(1)}, \dots, \theta^{(n_u)}}J(x^{(1)}, \dots, x^{(n_m)}, theta^{(1)}, \dots, \theta^{(n_u)})
$$
*the intercept term x_0 and \theta_0 convention is dropped for this application (because with the feature learning, the program will learn a certain feature x_i = 1 if needed)

#### 3. Collaborative Filtering algorithm

a. initialize $x^{(1)}, \dots, x^{(n_m)}, \theta^{(1)}, \dots, \theta^{(n_u)}$ to small random values

b. minimizing the cost function using optimization algorithm for every $j=1, ..., n_u, i=1, ..., n_m$

c. for a user with parameter $\theta$ and a movie with (learned) features x, predict a star rating of $\theta^Tx$

#### 4. Implementation

**Vectorization implementation: Low Rank Matrix Factorization**
$$
X = \begin{bmatrix}
- (x^{(1)})^T - \\
\vdots \\
- (x^{(n_m)})^T - 
\end{bmatrix} 
\Theta = \begin{bmatrix}
- (\theta^{(1)})^T - \\
\vdots \\
- (\theta^{(n_u)})^T - 
\end{bmatrix} \\
$$
predicted rating (low rank matrix)
$$
\begin{bmatrix}
 (\theta^{(1)})^T (x^{(1)}) &  (\theta^{(1)})^T (x^{(1)}) & \dots & (\theta^{(1)})^T (x^{(1)})\\
  (\theta^{(1)})^T (x^{(1)}) &  (\theta^{(1)})^T (x^{(1)}) & \dots & (\theta^{(1)})^T (x^{(1)})\\
\vdots &\vdots &\vdots &\vdots  \\
 (\theta^{(1)})^T (x^{(1)}) &  (\theta^{(1)})^T (x^{(1)}) & \dots & (\theta^{(1)})^T (x^{(1)})
\end{bmatrix} \\
$$
-> Finding related movies(how similar two movies are)

small $||x^{(i)}-x^{(j)}||$ => movie j is similar to movie i

**Mean Normalization**

from the rating matrix Y, compute the average \mu_i, and subtract the mean to get an average rating of 0 for each movie. and use the new set as the dataset

![image-20201028154008532](G:\Google Drive\Coursera-Machine Learning\Note Figures\image-20201028154008532.png)

prediction for user j, on movie i
$$
(\theta^{(j)})^T(x^{(i)})+\mu_i
$$

## Part E: Large scale machine learning

> Lecture 17

### A. Learning with large dataset, => sanity check

training with a small fraction of the dataset as a sanity check.

plotting the learning curve to find that if it shows a high variance learning algorithm and increasing the dataset will help

Instead, if the learning curve show high bias, instead of enlarging the dataset, might try the method of adding features etc. 

### B. Stochastic gradient descent

**Batch gradient descent** -> looking at all of the training examples m in each iteration

**stochastic gradient descent** (look at 1 example in each iteration)

#### 1. cost function

$$
cost(\theta, (x^{(i)},y^{(i)})) = \frac{1}{2}(h_{\theta}(x^{(i)})-y^{(i)})^2\\
J_{train}(\theta)=\frac{1}{m}\sum^m_{i=1}cost(\theta, (x^{(i)},y^{(i)}))
$$
#### 2. algorithm

```matlab
## a. randomly shuffle dataset

## b. sample-wise gradient descent
repeat {
  for i = 1, ...m {
      theta_j = theta_j - alpha\*(h_theta(x(i))-y(i))*x(i,j))
              (for every j=0,...,n)
    }
}
```

when the new sample come in, it will progressively make parameter towards global minimum

**advantage**

faster (and safer) than batch gradient descent

#### 3. Stochastic gradient descent convergence

1. learning curve

during learning, compute the cost of example before updating theta using (x(i), y(i))

Every 1000 iterations, plot cost averaged over las 1000 examples processed by the algorithm.

2. learning rate alpha

learning rate is typically held constant, can slowly decrease alpha over time if we want theta to converge. (eg. alpha = const1/(iterationNumber + const2))

### C. Mini-Batch Gradient descent

Use b examples in each iteration

**algorithm**

```matlab
# Say b=10, m= 1000

Repeat{

      for i = 1, 11, 21, 31, ..., 991{
      theta_j = theta_j - alpha\*1/b\*sum(h_theta(x(k))-y(k))*x(j,k)
          (for every j = 0, ..., n)

}

}
```

**advantage**

mini-batch gradient descent can be sometime faster than stochastic gradient descent. The reason is vectorization which will allow for partial parallelization over the b examples.

**disadvantage**

additional parameter of b

### D. Online Learning

learn from a continuous stream of data

**Example: shipping service**

choose your service (y=1)

feature x capture propertied of use,want to learn $p(y=1|x;\theta)$ to optimize price

algorithm

```matlab
# logistic regression
repeat forever {

   get (x, y ) corresponding to user

   update theta using (x, y) # not (x(i),y(i)) as in fixed training set

   theta(j) = theta(j) - alpha*(h(x)-y)*x(j)   (j=0,...,n)

}
```

* there is no longer a fixed training set. Instead, we learn from an example and then discard the example and never look again.

- this can also adapt to a changing user preferences over time 

**Example: product search (learning to search)**

user searches for "Android phone 1080p camera"

return 10 results among the 100

x = feature of phone, user query match etc

y= 1 if user click on link

learn $p(y=1|x;\theta)$   (predicted click-through-rate CTR)

use this to show the user 10 phones they are most likely to click on

- recommendations

### E. Map Reduce and data parallelism

Batch gradient descent, eg. 400 examples

split the training set in to n sets (evenly) -> compute -> combine results

Machine 1: use example 1~100
$$
temp_j^{(1)} = \sum^{100}_j(h_\theta(x^{(i)})-y^{(i)})*x_j^{(i)}
$$
....  four machines

Combine:
$$
\theta_j = \theta_j-\alpha*\frac{1}{400}(temp^{(1)}_j+temp^{(2)}_j+temp^{(3)}_j+temp^{(4)}_j)
$$

- also dependent on can your algorithms can be expressed as computing sums of functions over the training set (many learning algorithms can)

## Part F: Example of photo OCR (optical character recognition)

### A. Problem, Pipeline

pipeline

1. text detection
2. character segmentation
3. character classification

### B. Sliding Window

-> for text detection and character segmentation

Take the image patch (with expected aspect ratio)

run classifier

slide the patch by a certain step-size/stride

run classifier

...

take a larger image patch and repeat sliding and classification

**1D sliding window for character segmentation**

label the midpoint between two characters as positive (y=1)

train a classifier for split using this dataset

### C. Getting lots of data and artificial data

- Synthesizing data by introducing distortions. (eg. artificial warping to figure, adding noisy backgrounds to audio) Should be representative to the type you want to recognize. 
- Usually does not help to add purely random/meaningless noise
- make sure is low-bias (thus more data will help) before putting in the effort

### D. Ceiling analysis: what part of the pipeline to work on next

- estimate the errors due to each component
- give the upstream 100% accurate data and run through the pipeline and give the new accuracy of the following part of the pipeline

| component                   | accuracy |
| --------------------------- | -------- |
| Overall                     | 72%      |
| 100% text detection         | 89%      |
| 100% character segmentation | 90%      |
| 100% character recognition  | 100%     |

> from the table, spend more time on improving text detection will potential have an accuracy increase of 17%

## Personal side note

- Machine learning involves the fitting of many functions, and the prediction based on the fitted parameters, where NN is only one of them
- Classification: The  logistic regression relied on one vs all logic while NN output the probability for all classes.
- The calculation of cost function and its gradient is used by optimization functions to train for the best parameters. (optimization include gradient descent and more advanced ones)



## Questions

