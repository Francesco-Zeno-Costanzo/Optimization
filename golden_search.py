"""
search for the minimum of a function in a
variable with the golden-section search method
"""

def golden_search(f, a, b, tol):
    """
    Parameters
    ----------
    f : callable
        function to find the minimum,
        can be f(x)
    a : float
        first extreme
    b : float
        second extreme
    tol : float
        required tollerance
        the function stops when all components
        of the gradient have smaller than tol

    Returns
    -------
    sol : float
        minimum
    iter : int
        number of iteration
    """

    iter = 0
    gr = (5**(1/2) + 1)/2

    c = b - (b - a)/gr
    d = a + (b - a)/gr

    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c

        c = b - (b - a)/gr
        d = a + (b - a)/gr
        iter += 1

    sol = (b + a)/2

    return sol, iter

if __name__ == "__main__":

    def F(x):
        """
        function to find the minimum
        local minimun  =  0.8375654352833
        global minimum = -1.1071598716888
        """
        return (x**2 - 1)**2 + x

    x_min, iter = golden_search(F, -2, 5, 1e-10)

    print(f"Punto di minimo x_min = {x_min:.10f}")
    print(f"Valore nel minimo F(x_min) = {F(x_min):.10f}")
    print(f"numero di iterazioni = {iter}")