"""
the code performs a linear and non linear regression
Levenberg–Marquardt algorithm. You have to choose
some parameters delicately to make the result make sense
"""

import numpy as np
import matplotlib.pyplot as plt


def lm_fit(func, x0, tol, step, data, dense_output=False):
    """
    Implementation of Levenberg–Marquardt algorithm
    for non-linear least squares. This algorithm interpolates
    between the Gauss–Newton algorithm (GNA) and the method
    of gradient descent. It is iterative optimization algorithms
    so finds only a local minimum. So you have to be careful
    about the values ​​you pass in x0

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
    step : float
        size of spet to do, to choose carefully
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

    iter = 0               #initialize iteration counter
    h = 1e-7               #increment for derivatives
    l = 1e-3               #damping factor
    f = 10                 #factor for update damping factor
    M = len(x0)            #number of variable
    N = len(data[0])       #number of data
    s = np.zeros(M)        #auxiliary array for derivatives
    J = np.zeros((N, M))   #gradient
    x, y, dy = data        #data
    if dense_output:       #to store solution
        X = []
        X.append(x0)

    while True:
        #jacobian computation
        for i in range(M):                                  #loop over variables
            s[i] = 1                                        #we select one variable at a time
            dz1 = x0 + s*h                                  #step forward
            dz2 = x0 - s*h                                  #step backward
            J[:,i] = (func(x, *dz1) - func(x, *dz2))/(2*h)  #derivative along z's direction
            s[:] = 0                                        #reset to select the other variables

        JtJ = J.T @ J                             #matrix multiplication, JtJ is an MxM matrix
        dia = np.eye(M)*diag(JtJ)                 #dia_ii = JtJ_ii ; dia_ij = 0
        res = (y - func(x, *x0))/dy               #residuals
        b   = J.T @ res                           #ordinate or “dependent variable” values
        d   = np.linalg.solve(JtJ + l*dia, b)     #system solution
        x_n = x0 + d                              #solution at new time

        res_new = (y - func(x, *x_n))/dy          #new residuarls
        rms_res = np.sqrt(sum(res**2))            #=np.linalg.norm, nomr of residuals
        rms_rsn = np.sqrt(sum(res_new**2))        #norm of nwe residual

        if rms_rsn < rms_res :                    #if i'm closer to the solution
            x0 = x_n                              #update solution
            l /= f                                #reduce damping factor
        else:
            l *= f                                #else magnify

        if abs(rms_rsn - rms_res) < tol:          #break condition
            break

        iter += 1

        if dense_output:
            X.append(x0)

    if not dense_output:
        return x0, iter
    else :
        X = np.array(X)
        return X, iter


def f(x, m, q):
    """fit function
    """
    return m*np.cos(x*q)

##data
x = np.linspace(1, 5, 27)
y = f(x, 0.5, 10)
rng = np.random.default_rng(seed=69420)
y_noise = 0.1 * rng.normal(size=x.size)
y  = y + y_noise
dy = np.array(y.size*[0.1])

##fit

init = np.array([-1, 10.])   #|
tau  = 1e-8                  #|> be careful
step = 1e-4                  #|

pars, iter = lm_fit(f, init, tau, step, data=(x, y, dy))
for i, p in enumerate(pars):
    print(f"pars{i} = {p:.5f}")
print(f"numero di iterazioni = {iter}")

#Calcoliamo il chi quadro,indice ,per quanto possibile, della bontà del fit:
chisq = sum(((y - f(x, *pars))/dy)**2.)
ndof = len(y) - len(pars)
print(f'chi quadro = {chisq:.3f} ({ndof:d} dof)')

##Plot
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
init2 = np.array([-1, 10.5])
init3 = np.array([-1, 9.5])
tau   = 1e-8
step  = 1e-4

popt1, _  = lm_fit(f, init1, tau, step, data=(x, y, dy), dense_output=True)
popt2, _  = lm_fit(f, init2, tau, step, data=(x, y, dy), dense_output=True)
popt3, _  = lm_fit(f, init3, tau, step, data=(x, y, dy), dense_output=True)

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

