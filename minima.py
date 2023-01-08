"""
Small collection of optimization algorithms
ACHTUNG:
iterative optimization algorithms
finds only a local minimum.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as cm


## Test function

def F(x, y):
    """
    Himmelblau's function
    function to find the minimum
    4 equals minima ( x, y):
    I   = ( 3.0     ,  2.0     )
    II  = (-2.805118,  3.131312)
    III = (-3.779310, -3.283186)
    IV  = ( 3.584428, -1.848126)
    """
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def G(x):
    """
    function to find the minimum
    local minimun  =  0.8375654352833
    global minimum = -1.1071598716888
    """
    return (x**2 - 1)**2 + x


## gradient descent


def grad_disc(f, x0, tol, step):
    """
    implementation of gradient descent
    you have to be careful about the values
    ​​you pass in x0 if the function has more minima
    and also the value of steps is a delicate choice
    to be made wisely

    Parameters
    ----------
    f : callable
        function to find the minimum,
        can be f(x), f(x,y) and so on
    x0 : 1darray
        initial guess, to choose carefully
    tol : float
        required tollerance
        the function stops when all components
        of the gradient have smaller than tol
    step : float
        size of step to do, to choose carefully

    Returns
    -------
    X : ndarray
        array with all steps of solution
    iter : int
        number of iteration
    """
    iter = 0               #initialize iteration counter
    h = 1e-7               #increment for derivatives
    X = []                 #to store solution
    M = len(x0)            #number of variable
    s = np.zeros(M)        #auxiliary array for derivatives
    grad = np.zeros(M)     #gradient

    while True:
        #gradient computation
        for i in range(M):                       #loop over variables
            s[i] = 1                             #we select one variable at a time
            dz1 = x0 + s*h                       #step forward
            dz2 = x0 - s*h                       #step backward
            grad[i] = (f(*dz1) - f(*dz2))/(2*h)  #derivative along z's direction
            s[:] = 0                             #reset to select the other variables

        if all(abs(grad) < tol):
            break

        x0 = x0 - step*grad   #move towards the minimum
        X.append(x0)          #store iteration
        iter += 1             #update counter

    X = np.array(X)
    return X, iter


## gradient descent with momentum


def grad_disc_m(f, x0, tol, alpha, beta):
    """
    implementation of gradient descent with momentum
    you have to be careful about the values
    ​​you pass in x0 if the function has more minima
    and also the value of alpha an beta is a
    delicate choice to be made wisely

    Parameters
    ----------
    f : callable
        function to find the minimum,
        can be f(x), f(x,y) and so on
    x0 : 1darray
        initial guess, to choose carefully
    tol : float
        required tollerance
        the function stops when all components
        of the gradient have smaller than tol
    alpha : float
        size of step to do, to choose carefully
    beta : float
        size of step to do for velocity,
        to choose carefully, if beta = 0
        we get the method of gradient
        descent without momentum

    Returns
    -------
    X : ndarray
        array with all steps of solution
    iter : int
        number of iteration
    """
    iter = 0               #initialize iteration counter
    h = 1e-7               #increment for derivatives
    X = []                 #to store solution
    M = len(x0)            #number of variable
    s = np.zeros(M)        #auxiliary array for derivatives
    grad = np.zeros(M)     #gradient
    w = np.zeros(M)        #velocity, momentum

    while True:
        #gradient computation
        for i in range(M):                       #loop over variables
            s[i] = 1                             #we select one variable at a time
            dz1 = x0 + s*h                       #step forward
            dz2 = x0 - s*h                       #step backward
            grad[i] = (f(*dz1) - f(*dz2))/(2*h)  #derivative along z's direction
            s[:] = 0                             #reset to select the other variables

        if all(abs(grad) < tol):
            break

        w = beta*w + grad     #update velocity
        x0 = x0 - alpha*w     #update position move towards the minimum
        X.append(x0)          #store iteration
        iter += 1             #update counter

    X = np.array(X)
    return X, iter


## Adadelta


def adadelta(f, x0, tol, rho, eps):
    """
    implementation of adadelta alghoritm
    you have to be careful about the values
    ​​you pass in x0 if the function has more minima
    and also the value of alpha an beta is a
    delicate choice to be made wisely

    Parameters
    ----------
    f : callable
        function to find the minimum,
        can be f(x), f(x,y) and so on
    x0 : 1darray
        initial guess, to choose carefully
    tol : float
        required tollerance
        the function stops when all components
        of the gradient have smaller than tol
    rho : float
        parameter of alghoritm, decay rate, to choose carefully
    eps : float
        parameter of alghoritm, to choose carefully

    Returns
    -------
    X : ndarray
        array with all steps of solution
    iter : int
        number of iteration
    """
    iter = 0               #initialize iteration counter
    h = 1e-7               #increment for derivatives
    X = []                 #to store solution
    M = len(x0)            #number of variable
    s = np.zeros(M)        #auxiliary array for derivatives
    grad = np.zeros(M)     #gradient
    E_gr = np.zeros(M)     #running everage gradient
    E_dx = np.zeros(M)     #running everage dx

    while True:
        #gradient computation
        for i in range(M):                       #loop over variables
            s[i] = 1                             #we select one variable at a time
            dz1 = x0 + s*h                       #step forward
            dz2 = x0 - s*h                       #step backward
            grad[i] = (f(*dz1) - f(*dz2))/(2*h)  #derivative along z's direction
            s[:] = 0                             #reset to select the other variables

        if all(abs(grad) < tol):
            break

        E_gr = E_gr*rho + (1 - rho)*grad**2                      #E[g^2]
        dx   = - np.sqrt(E_dx + eps)/np.sqrt(E_gr + eps) * grad  #delta_x
        E_dx = E_dx*rho + (1 - rho)*dx**2                        #E[dx^2]

        x0 = x0 + dx      #update position move towards the minimum
        X.append(x0)      #store iteration
        iter += 1         #update counter

    X = np.array(X)
    return X, iter


## Adam


def adam(f, x0, tol, a, b1, b2, eps):
    """
    implementation of Adam alghoritm, Adaptive Moment Estimation
    you have to be careful about the values
    ​​you pass in x0 if the function has more minima
    and also the value of alpha an beta is a
    delicate choice to be made wisely

    Parameters
    ----------
    f : callable
        function to find the minimum,
        can be f(x), f(x,y) and so on
    x0 : 1darray
        initial guess, to choose carefully
    tol : float
        required tollerance
        the function stops when all components
        of the gradient have smaller than tol
    a : float
        size of step to do, to choose carefully, typical value is 0.001
    b1 : float
        Decay factor for first momentum, typical value is 0.9
    b2 : float
        Decay factor for second momentum, typical value is 0.999
    eps : float
        parameter of alghoritm, to choose carefully, typical value is 1e-8

    Returns
    -------
    X : ndarray
        array with all steps of solution
    iter : int
        number of iteration
    """
    iter = 0               #initialize iteration counter
    h = 1e-7               #increment for derivatives
    X = []                 #to store solution
    M = len(x0)            #number of variable
    s = np.zeros(M)        #auxiliary array for derivatives
    grad = np.zeros(M)     #gradient
    m = np.zeros(M)        #first moment
    v = np.zeros(M)        #second moment

    while True:
        #gradient computation
        for i in range(M):                       #loop over variables
            s[i] = 1                             #we select one variable at a time
            dz1 = x0 + s*h                       #step forward
            dz2 = x0 - s*h                       #step backward
            grad[i] = (f(*dz1) - f(*dz2))/(2*h)  #derivative along z's direction
            s[:] = 0                             #reset to select the other variables

        if all(abs(grad) < tol):
            break

        m = b1*m + (1 - b1)*grad
        v = b2*v + (1 - b2)*grad**2
        m_hat = m/(1 - b1**(iter+1))
        v_hat = v/(1 - b2**(iter+1))
        dx = a*m_hat/(np.sqrt(v_hat) + eps)
        x0 = x0 - dx      #update position move towards the minimum
        X.append(x0)      #store iteration
        iter += 1         #update counter

    X = np.array(X)
    return X, iter


##TEST


def test1d():
    """
    test for one variable's function
    """
    print("test 1D:")
    #---------------------------------------------------
    # No momentum
    #---------------------------------------------------
    print("no momentum")
    x0 = np.array([1.5])
    sol, iter = grad_disc(G, x0, 1e-8, 1e-3)

    xs1, = sol.T
    x_min = xs1[-1]
    min_f = G(x_min)

    print(f"Punto di minimo x_min = {x_min:.8f}")
    print(f"Valore nel minimo G(x_min) = {min_f:.8f}")
    print(f"numero di iterazioni = {iter}\n")
    #---------------------------------------------------
    # With momentum
    #---------------------------------------------------
    print("with momentum")
    x0 = np.array([1.5])
    sol, iter = grad_disc_m(G, x0, 1e-8, 1e-3, 0.953)  #global
    #sol, iter = grad_disc_m(G, x0, 1e-8, 1e-3, 0.9) #local

    xs2, = sol.T
    x_min = xs2[-1]
    min_f = G(x_min)

    print(f"Punto di minimo x_min = {x_min:.8f}")
    print(f"Valore nel minimo G(x_min) = {min_f:.8f}")
    print(f"numero di iterazioni = {iter}\n")
    #---------------------------------------------------
    # ADADELTA
    #---------------------------------------------------
    print("Adadelta")
    x0 = np.array([1.5])
    sol, iter = adadelta(G, x0, 1e-8, 0.95, 1e-6)

    xs3, = sol.T
    x_min = xs3[-1]
    min_f = G(x_min)

    print(f"Punto di minimo x_min = {x_min:.8f}")
    print(f"Valore nel minimo G(x_min) = {min_f:.8f}")
    print(f"numero di iterazioni = {iter}\n")
    #---------------------------------------------------
    # ADAM
    #---------------------------------------------------
    print("Adam")
    x0 = np.array([1.5])
    #sol, iter = adam(G, x0, 1e-8, 0.001, 0.999, 0.998, 1e-8) #global
    sol, iter = adam(G, x0, 1e-8, 0.001, 0.9, 0.999, 1e-8)   #local

    xs4, = sol.T
    x_min = xs4[-1]
    min_f = G(x_min)

    print(f"Punto di minimo x_min = {x_min:.8f}")
    print(f"Valore nel minimo G(x_min) = {min_f:.8f}")
    print(f"numero di iterazioni = {iter}\n")
    #---------------------------------------------------
    # Plot
    #---------------------------------------------------
    plt.figure(1)
    plt.title("Traiettorie soluzioni")
    t = np.linspace(-x0[0], x0[0], 1000)
    plt.plot(t, G(t), 'k', label='function')
    plt.plot(xs1, G(xs1)+0.1, 'r', label='no momentum')
    plt.plot(xs2, G(xs2)+0.2, 'b', label='with momentum')
    plt.plot(xs3, G(xs3)+0.3, 'g', label='adadelta')
    plt.plot(xs4, G(xs4)+0.4, 'y', label='adam')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('G(x)')
    plt.grid()
    plt.show()


def test2d():
    """
    test for two variable's function
    """
    print("test 2D:\n")
    #---------------------------------------------------
    # No momentum
    #---------------------------------------------------
    print("no momentum")
    x0 = np.array([-0.2, -0.9])
    sol, iter = grad_disc(F, x0, 1e-8, 1e-3)

    xs1, ys1 = sol.T
    x_min, y_min = xs1[-1], ys1[-1]
    min_f = F(x_min, y_min)

    print(f"Punto di minimo (x_min, y_min) = ({x_min:.8f}, {y_min:.8f})")
    print(f"Valore nel minimo F(x_min, y_min) = {min_f:.8f}")
    print(f"numero di iterazioni = {iter}\n")
    #---------------------------------------------------
    # With momentum
    #---------------------------------------------------
    print("with momentum")
    x0 = np.array([-0.2, -0.9])
    sol, iter = grad_disc_m(F, x0, 1e-8, 1e-3, 0.8)

    xs2, ys2 = sol.T
    x_min, y_min = xs2[-1], ys2[-1]
    min_f = F(x_min, y_min)

    print(f"Punto di minimo (x_min, y_min) = ({x_min:.8f}, {y_min:.8f})")
    print(f"Valore nel minimo F(x_min, y_min) = {min_f:.8f}")
    print(f"numero di iterazioni = {iter}\n")
    #---------------------------------------------------
    # ADADELTA
    #---------------------------------------------------
    print("Adadelta")
    x0 = np.array([-0.2, -0.9])
    sol, iter = adadelta(F, x0, 1e-8, 0.95, 1e-6)

    xs3, ys3 = sol.T
    x_min, y_min = xs3[-1], ys3[-1]
    min_f = F(x_min, y_min)

    print(f"Punto di minimo (x_min, y_min) = ({x_min:.8f}, {y_min:.8f})")
    print(f"Valore nel minimo F(x_min, y_min) = {min_f:.8f}")
    print(f"numero di iterazioni = {iter}\n")
    #---------------------------------------------------
    # ADAM
    #---------------------------------------------------
    print("Adam")
    x0 = np.array([-0.2, -0.9])
    sol, iter = adam(F, x0, 1e-8, 0.001, 0.9, 0.999, 1e-8)

    xs4, ys4 = sol.T
    x_min, y_min = xs4[-1], ys4[-1]
    min_f = F(x_min, y_min)

    print(f"Punto di minimo (x_min, y_min) = ({x_min:.8f}, {y_min:.8f})")
    print(f"Valore nel minimo F(x_min, y_min) = {min_f:.8f}")
    print(f"numero di iterazioni = {iter}")
    #---------------------------------------------------
    # Plot
    #---------------------------------------------------
    N = 200
    x = np.linspace(-6, 6, N)
    y = np.linspace(-6, 6, N)
    x, y = np.meshgrid(x, y)


    plt.figure(2)
    plt.title("Traiettorie soluzioni")
    plt.xlabel('x')
    plt.ylabel('y')
    levels = np.linspace(0, 300, 30)
    c=plt.contourf(x, y, F(x, y), levels=levels, cmap='jet')
    plt.colorbar(c)
    plt.grid()
    plt.plot(xs1, ys1, 'r', label='no momentum')
    plt.plot(xs2, ys2, 'b', label='with momentum')
    plt.plot(xs3, ys3, 'g', label='adadelta')
    plt.plot(xs4, ys4, 'y', label='adam')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":

    test1d()
    test2d()