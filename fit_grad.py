"""
the code performs a linear and non linear regression
using the descending gradient method. You have to
choose some parameters delicately to make the result
make sense
"""
import numpy as np
import matplotlib.pyplot as plt


def gdm_fit(f, x0, tol, alpha, beta, data, dense_output=False):
    """
    Implementation of gradient descent with momentum
    you have to be careful about the values
    ​​you pass in x0 if the function has more minima,
    it is iterative optimization algorithms so finds
    only a local minimum.
    Also the value of alpha an beta is a
    delicate choice to be made wisely

    Parameters
    ----------
    f : callable
        fit function
    x0 : 1darray
        initial guess
    tol : float
        required tollerance
        the function stops when all components
        of the gradient have smoller than tol
    alpha : float
        size of step to do, to choose carefully
    beta : float
        size of step to do for velocity,
        to choose carefully, if beta = 0
        we get the method of gradient
        descent without momentum
    data : tuple
        data to fit, data = (x, y, dy)
    dense_output : bool, optional dafult False
        if true all iteration are returned

    Returns
    -------
    x0 : ndarray
        array solution
    iter : int
        number of iteration
    """
    def F(data, pars):
        """squared deviation, function to be minimized
        """
        res = sum(((data[1] - f(data[0], *pars))/data[2])**2)
        return res

    iter = 0               #initialize iteration counter
    h = 1e-7               #increment for derivatives
    M = len(x0)            #number of variable
    s = np.zeros(M)        #auxiliary array for derivatives
    grad = np.zeros(M)     #gradient
    w = np.zeros(M)        #velocity, momentum
    if dense_output:       #to store solution
        X = []
        X.append(x0)

    while True:

        for i in range(M):                                 #loop over variables
            s[i] = 1                                       #we select one variable at a time
            dz1 = x0 + s*h                                 #step forward
            dz2 = x0 - s*h                                 #step backward
            grad[i] = (F(data, dz1) - F(data, dz2))/(2*h)  #derivative along z's direction
            s[:] = 0                                       #reset to select the other variables

        if all(abs(grad) < tol):
            break

        w = beta*w + grad     #update velocity
        x0 = x0 - alpha*w     #update position move towards the minimum
        iter += 1             #update counter

        if dense_output:      #to store solution
            X.append(x0)

    if not dense_output:
        return x0, iter
    else :
        X = np.array(X)
        return X, iter


def f(x, m, q):
    """fit function
    """
    return m*np.cos(q*x)

##data
x = np.linspace(1, 5, 27)
y = f(x, 0.5, 10)
rng = np.random.default_rng(seed=69420)
y_noise = 0.1 * rng.normal(size=x.size)
y  = y + y_noise
dy = np.array(y.size*[0.1])

##fit
init  = np.array([-1, 10.])  #|
tau   = 1e-8                 #|> be careful
alpha = 1e-5                 #|
beta  = 0.8                  #|

pars, iter = gdm_fit(f, init, tau, alpha, beta, data=(x, y, dy))
for i, p in enumerate(pars):
    print(f"pars{i} = {p:.5f}")
print(f"numero di iterazioni = {iter}")

#Calcoliamo il chi quadro,indice ,per quanto possibile, della bontà del fit:
chisq = sum(((y - f(x, *pars))/dy)**2.)
ndof = len(y) - len(pars)
print(f'chi quadro = {chisq:.3f} ({ndof:d} dof)')

##Plot fit
#Grafichiamo il risultato
fig1 = plt.figure(1)
#Parte superiore contenetnte il fit:
frame1=fig1.add_axes((.1,.35,.8,.6))
#frame1=fig1.add_axes((trasla lateralmente, trasla verticamente, larghezza, altezza))
frame1.set_title('Fit dati simulati',fontsize=10)
plt.ylabel('y [u.a.]',fontsize=10)
plt.grid()


plt.errorbar(x, y, dy, fmt='.', color='black', label='dati') #grafico i punti
t = np.linspace(np.min(x), np.max(x), 10000)
plt.plot(t, f(t, *pars), color='blue', alpha=0.5, label='best fit') #grafico del best fit
plt.legend(loc='best')#inserisce la legenda nel posto migliorte


#Parte inferiore contenente i residui
frame2=fig1.add_axes((.1,.1,.8,.2))

#Calcolo i residui normalizzari
ff = (y - f(x, *pars))/dy
frame2.set_ylabel('Residui Normalizzati')
plt.xlabel('x [u.a.]',fontsize=10)

plt.plot(t, 0*t, color='red', linestyle='--', alpha=0.5) #grafico la retta costantemente zero
plt.plot(x, ff, '.', color='black') #grafico i residui normalizzati
plt.grid()

##Plot tariettoria
N = 200
p1 = np.linspace(-1, 1.25, N)
p2 = np.linspace(8, 11.5, N)

S2 = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        S2[i, j] = (((y - f(x, p1[i], p2[j]))/dy)**2).sum()

init1 = np.array([-1, 10.])
init2 = np.array([-1, 10.1])
init3 = np.array([-1, 9.9])
tau   = 1e-8
alpha = 1e-5
beta  = 0.85

popt1, _  = gdm_fit(f, init1, tau, alpha, beta, data=(x, y, dy), dense_output=True)
popt2, _  = gdm_fit(f, init2, tau, alpha, beta, data=(x, y, dy), dense_output=True)
popt3, _  = gdm_fit(f, init3, tau, alpha, beta, data=(x, y, dy), dense_output=True)

plt.figure(2)
plt.title("Traiettorie soluzioni")
plt.xlabel('x')
plt.ylabel('y')
levels = np.linspace(np.min(S2), np.max(S2), 40)
p1, p2 = np.meshgrid(p1, p2)
c=plt.contourf(p1, p2, S2.T , levels=levels, cmap='jet')
plt.colorbar(c)
plt.grid()
plt.plot(popt1[:,0], popt1[:,1], 'k', label='tariettoria1')
plt.plot(popt2[:,0], popt2[:,1], 'k', label='tariettoria2')
plt.plot(popt3[:,0], popt3[:,1], 'k', label='tariettoria3')
plt.legend(loc='best')

plt.show()
