# Advection-Diffusion Equation
# ========================================
#python -i advection.py -n 20 -exact sin
#python -i advection.py -n 40 -exact exp -b 1 -wind 1. -uRHS 1.0 -nu 1 -ksp_type cg -pc_type none
#python -i advection.py -n 1000 -exact exp -b 1 -wind 1. -uRHS 1.0 -nu 1e-3 -ksp_type lsqr -pc_type svd -ksp_view -ksp_atol 1e-6 -ksp_rtol 1e-20 -ksp_monitor -ksp_lsqr_set_standard_error -ksp_lsqr_exact_mat_norm

import sys
import petsc4py
import numpy as np
import scipy.linalg as la
import mpmath as mp
mp.dps = 25

petsc4py.init(sys.argv)
from petsc4py import PETSc

OptDB = PETSc.Options()
n = OptDB.getInt('n', 5)
precision = OptDB.getInt('precision', 25)
a = OptDB.getReal('a', 0.0)
b = OptDB.getReal('b', np.pi)
w = OptDB.getReal('wind', 0.0)
nu = OptDB.getReal('nu', 1.0)

uRHS = OptDB.getReal('uRHS', 0.0)
uLHS = OptDB.getReal('uLHS', 0.0)

case = OptDB.getString('exact', 'none')
precond = OptDB.getString('precond', 'self')
upwind = OptDB.getBool('upwind', True)
pre_upwind = OptDB.getBool('precond_upwind', True)
pre_complex = OptDB.getBool('precond_complex', False)
plot = OptDB.getBool('plot', False)
force = OptDB.getBool('force', False)
invert = OptDB.getBool('invert', False)

print("Preconditioner with complex value:", pre_complex)
print("Preconditioner upwind:", pre_upwind)
print("Upwind:", upwind)

if n>0:
    h = (b-a) / (n + 1)
if n==0:
    h =  0.5*nu/abs(w)
    n = int((b-a)/h)
    h = (b-a) / (n + 1)


if (case == 'exp' and (h*abs(w))/(2*nu)>1) and not force:
    raise ValueError('You need a finer mesh to resolve the boundary layer')

#   Constructing the discretization matrix for a Laplacian
D = PETSc.Mat()
D.create(comm=PETSc.COMM_WORLD)
D.setSizes((n, n))
D.setType(PETSc.Mat.Type.AIJ)
D.setFromOptions()
D.setPreallocationNNZ(3)

rstart, rend = D.getOwnershipRange()
for row in range(rstart, rend):
    D[row, row] = (2.0*(w**2))/h**2 
    if row > 0:
        D[row, row - 1] = (-1.0*(w**2))/(h**2)
    if row < n - 1:
        D[row, row + 1] = (-1.0*(w**2))/(h**2)

D.assemblyBegin()
D.assemblyEnd()

# Constructing the discretization matrix 
A = PETSc.Mat()
A.create(comm=PETSc.COMM_WORLD)
A.setSizes((n, n))
A.setType(PETSc.Mat.Type.AIJ)
A.setFromOptions()
A.setPreallocationNNZ(3)

rstart, rend = A.getOwnershipRange()
for row in range(rstart, rend):
    if upwind:
        A[row, row] = (2.0*nu)/h**2 + w/h
        if row > 0:
            A[row, row - 1] = (-1.0*nu)/(h**2) -w/h
        if row < n - 1:
            A[row, row + 1] = (-1.0*nu)/(h**2)
    else:
        A[row, row] = (2.0*nu)/h**2
        if row > 0:
            A[row, row - 1] = (-1.0*nu)/(h**2) - 0.5*w/h
        if row < n - 1:
            A[row, row + 1] = (-1.0*nu)/(h**2) + 0.5*w/h

A.assemblyBegin()
A.assemblyEnd()

# Constructing the precondtioner
P = PETSc.Mat()
P.create(comm=PETSc.COMM_WORLD)
P.setSizes((n, n))
P.setType(PETSc.Mat.Type.AIJ)
P.setFromOptions()
P.setPreallocationNNZ(3)

rstart, rend = P.getOwnershipRange()
for row in range(rstart, rend):
    if pre_upwind:
        if not pre_complex:
            P[row, row] = w/h
            if row > 0:
                P[row, row - 1] = -w/h
        else:
            P[row, row] = 1j*0.5*w/h
            if row > 0:
                P[row, row - 1] = -1j*0.5*w/h
    else:
        if row > 0:
            P[row, row - 1] = -0.5*w/h
        if row < n - 1:
            P[row, row + 1] = 0.5*w/h

P.assemblyBegin()
P.assemblyEnd()

# PETSc represents all linear solvers as preconditioned Krylov subspace methods
# of type `PETSc.KSP`. Here we create a KSP object for a conjugate gradient
# solver preconditioned with an algebraic multigrid method.

I = PETSc.Mat().createConstantDiagonal((n, n), 1.0)
I.convert('aij')
PT = P.copy()
PT = PT.transpose()
PTP = PT.matMatMult(P, I)
PPT = P.matMatMult(PT, I)


ksp = PETSc.KSP()
ksp.create(comm=A.getComm())
if  precond == 'PTP':
    C = PTP
elif precond == 'PPT':
    C = PPT
elif precond == 'P':
    C = P
elif precond == 'PT':
    C = PT
elif precond == 'self':
    C = A
elif precond == 'PLR':
    Adense = A.copy()
    Adense.convert('dense')
    Anp = Adense.getDenseArray()
    U,P = la.polar(Anp, side='right')
    C =PETSc.Mat().createDense((n,n),1,P)
    D = PETSc.Mat().createDense((n,n),1,U)
    C.convert('aij')
    D.convert('aij')
    testA = D.matMatMult(C, I)
    print('Test A:', (Adense-testA).norm(), np.linalg.norm(Anp-U@P))
elif precond == "PLR_SVD":
    mp.dps = precision
    Adense = A.copy()
    Adense.convert('dense')
    Anp = Adense.getDenseArray()
    Anp = np.array(Anp)
    Amp = mp.matrix(Anp.tolist())
    W, S, Vh = mp.svd_r(Amp)
    Pmp = Vh.T@mp.diag(S)@Vh
    Ump = W@Vh
    U = np.array(Ump.tolist(), dtype=np.float64)
    P = np.array(Pmp.tolist(), dtype=np.float64)
    C =PETSc.Mat().createDense((n,n),1,P)
    D = PETSc.Mat().createDense((n,n),1,U)
    C.convert('aij')
    D.convert('aij')
    testA = D.matMatMult(C, I)
    testAmp = Ump@Pmp
    testAnp = np.array(testAmp.tolist(), dtype=np.float64)
    print('Test A:', (Adense-testA).norm(), np.linalg.norm(Anp-testAnp))

elif precond == 'PLL':
    Adense = A.copy()
    Adense.convert('dense')
    Anp = Adense.getDenseArray()
    U,P = la.polar(Anp, side='left')
    C =PETSc.Mat().createDense((n,n),1,P)
    C.convert('aij')
elif precond == 'QR' :
    Adense = A.copy()
    Adense.convert('dense')
    Anp = Adense.getDenseArray()
    Q,R = la.qr(Anp)
    C =PETSc.Mat().createDense((n,n),1,R)
    C.convert('aij')
elif precond == 'RQ':
    Adense = A.copy()
    Adense.convert('dense')
    Anp = Adense.getDenseArray()
    R,Q = la.rq(Anp)
    C =PETSc.Mat().createDense((n,n),1,R)
    C.convert('aij')
    C.convert('aij')
elif precond == 'P_square_root':
    PTPdense = PTP.copy()
    PTPdense.convert('dense')
    PTPnp = PTPdense.getDenseArray()
    S = la.sqrtm(PTPnp)
    C =PETSc.Mat().createDense((n,n),1,S)
    C.convert('aij')
elif precond == 'PQR':
    Pdense = P.copy()
    Pdense.convert('dense')
    Pnp = Pdense.getDenseArray()
    Q, R = la.qr(Pnp)
    C =PETSc.Mat().createDense((n,n),1,R)
    C.convert('aij')
elif precond == 'PTPQR':
    Pdense = P.copy()
    Pdense.convert('dense')
    Pnp = Pdense.getDenseArray()
    Q, R = la.qr(Pnp)
    C =PETSc.Mat().createDense((n,n),1,R)
    C.convert('aij')
    CT = C.copy()
    CT = CT.transpose()
    CTC = CT.matMatMult(C, I)
    C = CTC
elif precond == 'PTP_square_root':
    PTPdense = PTP.copy()
    PTPdense.convert('dense')
    PTPnp = PTPdense.getDenseArray()
    S = la.sqrtm(PTPnp)
    C =PETSc.Mat().createDense((n,n),1,S)
    C.convert('aij')
    CT = C.copy()
    CT = CT.transpose()
    CTC = CT.matMatMult(C, I)
    C = CTC
elif precond == 'poisson_square_root':
    D.convert('dense')
    Dnp = D.getDenseArray()
    S = la.sqrtm(Dnp)
    C =PETSc.Mat().createDense((n,n),1,S)
elif precond == 'poisson':
    C = D
elif precond == 'poisson_qr':
    D.convert('dense')
    Dnp = D.getDenseArray()
    Q, R = la.qr(Dnp)
    C =PETSc.Mat().createDense((n,n),1,R)
    C.convert('aij')
else:
    raise ValueError('Unknown preconditioner')

if not invert:
    A = -1*A

ksp.setOperators(A,C)
ksp.setFromOptions()

# Since the matrix knows its size and parallel distribution, we can retrieve
# appropriately-scaled vectors using `Mat.createVecs`. PETSc vectors are
# objects of type `PETSc.Vec`. Here we set the right-hand side of our system to
# a vector of ones, and then solve.

x, f = A.createVecs()
omega = np.linspace(a, b, n+2)[1:-1]
if case == 'none':
    f.setArray(np.zeros_like(omega))
elif case == 'sin':
    f.setArray(np.sin(omega))
elif case == 'exp':
    f.setArray(np.zeros_like(omega))
f.setValue(rstart, f.getValue(rstart)+ nu*uLHS/h**2 + w*uLHS/(2*h))
f.setValue(rend-1, f.getValue(rend-1)+ nu*uRHS/h**2 - w*uRHS/(2*h))
if not invert:
    f = -1*f

def monitor(ksp, it, rnorm):
    x = ksp.getSolution()
    r = x.duplicate()
    A.mult(x, r)
    r = r - f
    print("\t||Ax - b|| = {}".format(r.norm()))
ksp.setMonitor(monitor)

ksp.solve(f, x)

# Finally, allow the user to print the solution by passing ``-view_sol`` to the
# script.

from matplotlib import pyplot as plt
exact = x.duplicate()
if case == 'none':
    exact.setArray(np.zeros_like(omega))
elif case == 'sin':
    exact.setArray(np.sin(omega))
elif case == 'exp':
    sol = lambda x: (mp.exp((w/nu)*x)-1)/(mp.exp(w/nu)-1)
    np_sol = np.zeros_like(omega)
    for i in range(n):
        np_sol[i] = sol(omega[i])
    exact.setArray(np_sol)
else:
    raise ValueError('Unknown exact solution')

error = exact - x
print('Error norm:', error.norm())

if plot:
    fine_omega = np.linspace(a, b, 100000)
    np_fine_sol = np.zeros_like(fine_omega)
    for i in range(100000):
        np_fine_sol[i] = sol(fine_omega[i])
    plt.plot(fine_omega, np_fine_sol, label='Exact Solution')
    plt.plot([a]+list(omega)+[b], [uLHS]+list(x.getArray())+[uRHS], ".", label='Discrete Solution')
    plt.legend()
    plt.show()