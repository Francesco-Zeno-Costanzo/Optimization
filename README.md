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

Consider our fit function f which depends on an independent variable and a set of parameters \theta, which is basically a vector of R^m. We can expand f into Taylor series around a value of our parameters:

<img src="https://latex.codecogs.com/svg.image?f&space;(x_i,&space;\theta_j&space;&plus;&space;\delta_j&space;)&space;\simeq&space;f(x&space;_i,&space;\theta_j)&space;&plus;&space;J_{ij}&space;\delta_j" title="https://latex.codecogs.com/svg.image?f (x_i, \theta_j + \delta_j ) \simeq f(x _i, \theta_j) + J_{ij} \delta_j" />

where \delta is the step of each iteration and J:

<img src="https://latex.codecogs.com/svg.image?J_{ij}&space;=&space;\frac{\partial&space;f(x_i,&space;\theta_j)}{\partial&space;\theta_j}&space;=&space;\begin{bmatrix}&space;\dfrac{\partial&space;f(x_1,&space;\theta_1)}{\partial&space;\theta_1}&space;&&space;\cdots&space;&&space;\dfrac{\partial&space;f(x_1,&space;\theta_m)}{\partial&space;\theta_m}&space;\\&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;\dfrac{\partial&space;f(x_n,&space;\theta_1)}{\partial&space;\theta_1}&space;&&space;\cdots&space;&&space;\dfrac{\partial&space;f(x_n,&space;\theta_m)}{\partial&space;\theta_m}&space;&space;\end{bmatrix}" title="https://latex.codecogs.com/svg.image?J_{ij} = \frac{\partial f(x_i, \theta_j)}{\partial \theta_j} = \begin{bmatrix} \dfrac{\partial f(x_1, \theta_1)}{\partial \theta_1} & \cdots & \dfrac{\partial f(x_1, \theta_m)}{\partial \theta_m} \\ \vdots & \ddots & \vdots \\ \dfrac{\partial f(x_n, \theta_1)}{\partial \theta_1} & \cdots & \dfrac{\partial f(x_n, \theta_m)}{\partial \theta_m} \end{bmatrix}" />


So we have:


<img src="https://latex.codecogs.com/svg.image?\\S^2(\theta&space;&plus;&space;\delta)&space;\simeq&space;\sum_{i=1}^n&space;\frac{(y_i&space;-&space;f&space;(x_i,&space;\beta)&space;-&space;J_{ij}\delta_j)^2}{\sigma_{y_i}^2}&space;\\=&space;(y&space;-&space;f(x,&space;\theta)&space;-&space;J&space;\delta)^{T}&space;W&space;(y&space;-&space;f(x,&space;\theta)&space;-&space;J&space;\delta)&space;\\=(y&space;-&space;f(x,&space;\theta))^{T}&space;W&space;(y&space;-&space;f(x,&space;\theta))&space;-&space;(y&space;-&space;f(x,&space;\theta))^{T}&space;W&space;J&space;\delta&space;-&space;(J&space;\delta)^T&space;W&space;(y&space;-&space;f(x,&space;\theta))&space;&plus;&space;(J&space;\delta)^T&space;W&space;(J&space;\delta)&space;\\=(y&space;-&space;f(x,&space;\theta))^{T}&space;W&space;(y&space;-&space;f(x,&space;\theta))&space;-&space;2(y&space;-&space;f(x,&space;\theta))^{T}&space;W&space;J&space;\delta&space;&plus;&space;\delta^T&space;J^T&space;W&space;(J&space;\delta)" title="https://latex.codecogs.com/svg.image?\\S^2(\theta + \delta) \simeq \sum_{i=1}^n \frac{(y_i - f (x_i, \beta) - J_{ij}\delta_j)^2}{\sigma_{y_i}^2} \\= (y - f(x, \theta) - J \delta)^{T} W (y - f(x, \theta) - J \delta) \\=(y - f(x, \theta))^{T} W (y - f(x, \theta)) - (y - f(x, \theta))^{T} W J \delta - (J \delta)^T W (y - f(x, \theta)) + (J \delta)^T W (J \delta) \\=(y - f(x, \theta))^{T} W (y - f(x, \theta)) - 2(y - f(x, \theta))^{T} W J \delta + \delta^T J^T W (J \delta)" />


Where W is such that W{ii} = 1/ \sigma_{y_i}^2 and differentiating with respect to delta we obtain the Gauss-Newton method:


<img src="https://latex.codecogs.com/svg.image?\\\frac{\partial&space;S^2(\theta&space;&plus;&space;\delta)}{\partial&space;\delta}&space;=&space;-&space;2(y&space;-&space;f(x,&space;\theta))^{T}&space;W&space;J&space;&plus;&space;2&space;\delta^T&space;J^T&space;W&space;J=0\\\\(J^T&space;W&space;J)&space;\delta&space;=&space;J^T&space;W&space;(y&space;-&space;f(x,&space;\theta))&space;" title="https://latex.codecogs.com/svg.image?\\\frac{\partial S^2(\theta + \delta)}{\partial \delta} = - 2(y - f(x, \theta))^{T} W J + 2 \delta^T J^T W J=0\\\\(J^T W J) \delta = J^T W (y - f(x, \theta)) " />


Which solves for \delta. To improve the convergence of the method, a damping parameter \lambda is introduced and the equation becomes:


<img src="https://latex.codecogs.com/svg.image?(J^T&space;W&space;J&space;-&space;\lambda&space;\,&space;\text{diag}(J^T&space;W&space;J))&space;\delta&space;=&space;J^T&space;W&space;(y&space;-&space;f(x,&space;\theta))" title="https://latex.codecogs.com/svg.image?(J^T W J - \lambda \, \text{diag}(J^T W J)) \delta = J^T W (y - f(x, \theta))" />


The value of \lambda is changed depending on whether or not we get close to the right solution. If we are getting close we reduce its value, going towards the Gauss-Newton method; while if we move away we increase the value so that the algorithm behaves more like a descending gradient (of which there will be an example in the appendix). The question is: how do we know if we are getting close to the solution? We calculate:


<img src="https://latex.codecogs.com/svg.image?\\\rho(\delta)&space;=&space;\frac{S^2(x,&space;\theta)&space;-&space;S^2(x,&space;\theta&space;&plus;&space;\delta)}{|(y&space;-&space;f(x,&space;\theta)&space;-&space;J&space;\delta)^{T}&space;W&space;(y&space;-&space;f(x,&space;\theta)&space;-&space;J&space;\delta)|}&space;\\\\=&space;\frac{S^2(x,&space;\theta)&space;-&space;S^2(x,&space;\theta&space;&plus;&space;\delta)}{|&space;\delta^T&space;(\lambda&space;\text{diag}(J.T&space;W&space;J)&space;\delta&space;&plus;&space;J.T&space;W&space;(y&space;-&space;f(x,&space;\theta)))|}" title="https://latex.codecogs.com/svg.image?\\\rho(\delta) = \frac{S^2(x, \theta) - S^2(x, \theta + \delta)}{|(y - f(x, \theta) - J \delta)^{T} W (y - f(x, \theta) - J \delta)|} \\\\= \frac{S^2(x, \theta) - S^2(x, \theta + \delta)}{| \delta^T (\lambda \text{diag}(J.T W J) \delta + J.T W (y - f(x, \theta)))|}" />


if \rho(\delta) > eps_1 the move is accepted and we reduce \lambda otherwise we stay in the old position.
Another question to answer is: when did we arrive at convergence? we define:


<img src="https://latex.codecogs.com/svg.image?\\R1&space;=&space;\text{max}(|J.T&space;W&space;(y&space;-&space;f(x,&space;\theta))|)&space;\\R2&space;=&space;\text{max}(|&space;\delta/&space;\theta&space;|)&space;\\R3&space;=&space;|S^2(x,&space;\theta)/(n&space;-&space;m)&space;-&space;1|&space;" title="https://latex.codecogs.com/svg.image?\\R1 = \text{max}(|J.T W (y - f(x, \theta))|) \\R2 = \text{max}(| \delta/ \theta |) \\R3 = |S^2(x, \theta)/(n - m) - 1| " />


If one of these quantities is less than a certain tolerance then the algorithm terminates. Now there remains one last question to answer and we can move on to the code. Since we need the errors on the fit parameters: how do we compute the covariance matrix? Just calculate:


<img src="https://latex.codecogs.com/svg.image?\text{Cov}&space;=&space;(J^T&space;W&space;J)^{-1}" title="https://latex.codecogs.com/svg.image?\text{Cov} = (J^T W J)^{-1}" />
