# Optimization

codes for solving minima and least squares fit problems

## Golden search

POI LO SCRIVO

## Gradient descent

POI LO SCRIVO

## Gradient descent with momentum

POI LO SCRIVO

## ADADELTA

POI LO SCRIVO

## ADAM

POI LO SCRIVO

## Levenberg–Marquardt

Consider our fit function f which depends on an independent variable and a set of parameters $\theta$, which is basically a vector of $\mathbb{R}^m$. We can expand f into Taylor series around a value of our parameters:

$$
f (x_i, \theta_j + \delta_j ) \simeq f(x _i, \theta_j) + J_{ij} \delta_j
$$

where \delta is the step of each iteration and J:

$$
J_{ij} = \frac{\partial f(x_i, \theta_j)}{\partial \theta_j} = 
\begin{bmatrix} \dfrac{\partial f(x_1, \theta_1)}{\partial \theta_1} & \cdots & \dfrac{\partial f(x_1, \theta_m)}{\partial \theta_m} \\ \vdots & \ddots & \vdots \\ \dfrac{\partial f(x_n, \theta_1)}{\partial \theta_1} & \cdots & \dfrac{\partial f(x_n, \theta_m)}{\partial \theta_m}  \end{bmatrix}
$$



So we have:

$$
\begin{split}
S^2(\theta + \delta) &\simeq \sum_{i=1}^n \frac{(y_i - f (x_i, \beta) - J_{ij}\delta_j)^2}{\sigma_{y_i}^2} \\
&= (y - f(x, \theta) - J \delta)^{T} W (y - f(x, \theta) - J \delta)\\
&=(y - f(x, \theta))^{T} W (y - f(x, \theta)) - (y - f(x, \theta))^{T} W J \delta - (J \delta)^T W (y - f(x, \theta)) + (J \delta)^T W (J \delta) \\
&=(y - f(x, \theta))^{T} W (y - f(x, \theta)) - 2(y - f(x, \theta))^{T} W J \delta + \delta^T J^T W (J \delta)
\end{split}
$$



Where W is such that $W_{ii} = 1/ \sigma_{y_i}^2$ and differentiating with respect to delta we obtain the Gauss-Newton method:

$$
\frac{\partial S^2(\theta + \delta)}{\partial \delta} = - 2(y - f(x, \theta))^{T} W J + 2 \delta^T J^T W J=0
$$

$$
(J^T W J) \delta = J^T W (y - f(x, \theta))
$$

Which solves for $\delta$. To improve the convergence of the method, a damping parameter $\lambda$ is introduced and the equation becomes:

$$
(J^T W J - \lambda \, \text{diag}(J^T W J)) \delta = J^T W (y - f(x, \theta))
$$

The value of $\lambda$ is changed depending on whether or not we get close to the right solution. If we are getting close we reduce its value, going towards the Gauss-Newton method; while if we move away we increase the value so that the algorithm behaves more like a descending gradient (of which there will be an example in the appendix). The question is: how do we know if we are getting close to the solution? We calculate:

$$
\begin{split}
\rho(\delta) &= \frac{S^2(x, \theta) - S^2(x, \theta + \delta)}{|(y - f(x, \theta) - J \delta)^{T} W (y - f(x, \theta) - J \delta)|} \\
& = \frac{S^2(x, \theta) - S^2(x, \theta + \delta)}{| \delta^T (\lambda \text{diag}(J.T W J) \delta + J.T W (y - f(x, \theta)))|}
\end{split}
$$

if $\rho(\delta) > \varepsilon_1$ the move is accepted and we reduce \lambda otherwise we stay in the old position.
Another question to answer is: when did we arrive at convergence? we define:

$$
\begin{split}
R1 &= \text{max}(|J.T W (y - f(x, \theta))|) \\
R2 &= \text{max}(| \delta/ \theta |) \\
R3 &= |S^2(x, \theta)/(n - m) - 1|
\end{split}
$$

If one of these quantities is less than a certain tolerance then the algorithm terminates. Now there remains one last question to answer and we can move on to the code. Since we need the errors on the fit parameters: how do we compute the covariance matrix? Just calculate:

$$
\text{Cov} = (J^T W J)^{-1}
$$
