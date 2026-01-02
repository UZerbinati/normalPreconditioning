from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sksparse.cholmod import cholesky
import scipy.linalg as la
import pandas as pd

OptDB = PETSc.Options()
n = OptDB.getInt('n', 128)
nu = OptDB.getReal('nu', 1e-2)
refs = OptDB.getInt('ref', 2)
spectral_equiv = OptDB.getBool('spec_equiv', False)
wind_type = OptDB.getInt('wind', 1)


print("N: {}, nu: {}: ref: {} -- real N: {}".format(n,nu,refs, (2**refs)*n))
print("Wind type: {}".format(wind_type))


mesh = UnitSquareMesh(n, n)
mh = MeshHierarchy(mesh, refs)
mesh = mh[-1]
x,y = SpatialCoordinate(mesh)
W = FunctionSpace(mesh, 'CG', 1)
V = RestrictedFunctionSpace(W,boundary_set=[1,2,3,4])
delta = Constant(1e-4)

res = []
# Define the advection velocity and viscosity
if wind_type == 1:
    w = Constant((1.0, 0.0))
    w = 1/(inner(w,w))*w
    wperp = Constant((0.0, 1.0))
    wperp = 1/(inner(wperp,wperp))*wperp
elif wind_type == 2:
    w = Constant((1.0, 1.0))
    w = 1/(inner(w,w))*w
    wperp = Constant((-1.0, 1.0))
    wperp = 1/(inner(wperp,wperp))*wperp
elif wind_type == 3:
    w = as_vector((x, -y))
    w = 1/(inner(w,w))*w
    wperp = as_vector((y, x))
    wperp = 1/(inner(wperp,wperp))*wperp
else:
    raise RuntimeError("Wind not defined...")
nu = Constant(nu)

#Define the bilinear form
u = TrialFunction(V)
v = TestFunction(V)
a = (nu*inner(grad(u), grad(v)) + inner(w, grad(u)) * v) * dx + delta*inner(w, grad(u))*inner(w, grad(v))*dx-delta*nu*inner(div(grad(u)), inner(w, grad(v)))*dx
m = nu*inner(grad(u), grad(v))*dx
p = (1/nu)*inner(w*u,w*v)*dx + nu*inner(grad(u), grad(v))*dx
# Define the linear form
f = Constant(1.0) * v * dx + delta*Constant(1.0)*inner(w, grad(v))*dx

A = assemble(a)
M = assemble(m)
P = assemble(p)
b = assemble(f)
u = Function(V)

#Construct PETSc objects
A_petsc = A.M.handle
AT_petsc = A_petsc.copy()
AT_petsc = AT_petsc.transpose()
M_petsc = M.M.handle
P_petsc = P.M.handle

#Computing the Cholesky factorization of the mass matrix
Msp = csr_matrix(M_petsc.getValuesCSR()[::-1], shape=M_petsc.size)
Msp = Msp.tocsc()
chol_factor = cholesky(Msp)

with b.dat.vec as v:
    b_petsc = v
with u.dat.vec as v:
    u_petsc = v

#Cosntruct a PETSc Krylov solver GMRES
ksp = PETSc.KSP()
ksp.create(comm=mesh.comm)
ksp.setOperators(A_petsc)
ksp.setOptionsPrefix("lin_sys_")
ksp.setFromOptions()
ksp.solve(b_petsc, u_petsc)

with u.dat.vec as v:
    v.copy(u_petsc)

#Output the solution
File('output/fem_advection.pvd').write(u)
print("Solved GMRES reference system")
class CinvA_petsc(object):
    """
    A^T M^{-1} A = A^T (P^TC C^TP)^{-1} A 
    A^T M^{-1} A = A^T P^TC^{-T} C^{-1}P A

    """
    def mult(self, x, y) -> None:
        v = x.duplicate()
        A_petsc.mult(x, v)
        y.setArray(chol_factor.solve_L(chol_factor.apply_P(v.getArray()),use_LDLt_decomposition=False))
    def multTranspose(self, x, y) -> None:
        v = x.copy()
        w = x.copy()
        u = x.copy()
        v.setArray(chol_factor.solve_Lt(w.getArray(), use_LDLt_decomposition=False))
        u.setArray(chol_factor.apply_Pt(v.getArray()))
        AT_petsc.mult(u, y)

CinvA = PETSc.Mat().create(comm=mesh.comm)
CinvA.setSizes(A_petsc.getSizes())
CinvA.setType(PETSc.Mat.Type.PYTHON)
CinvA.setPythonContext(CinvA_petsc)
CinvA.setUp()

#Defining the operator for the normal equations
class ATMinvA_petsc(object):
    def mult(self, x, y) -> None:
        v = x.duplicate()
        A_petsc.mult(x, v)
        w = x.duplicate()
        ksp = PETSc.KSP()
        ksp.create(comm=mesh.comm)
        ksp.setOperators(M_petsc)
        ksp.setOptionsPrefix("mass_inv_")
        ksp.setFromOptions()
        ksp.solve(v, w)
        AT_petsc.mult(w, y)

ATMinvA = PETSc.Mat().create(comm=mesh.comm)
ATMinvA.setSizes(A_petsc.getSizes())
ATMinvA.setType(PETSc.Mat.Type.PYTHON)
ATMinvA.setPythonContext(ATMinvA_petsc)
ATMinvA.setUp()

#Construct the right hand side for the normal equations
ksp = PETSc.KSP()

if OptDB.getString('normal_eq_ksp_type', 'unspec') in ['fcg', 'cg']:
    ksp.create(comm=mesh.comm)
    ksp.setOperators(M_petsc)
    ksp.setOptionsPrefix("mass_inv_")
    ksp.setFromOptions()
    z_petsc = b_petsc.copy()
    y_petsc = b_petsc.duplicate()
    ksp.solve(b_petsc, y_petsc)
    AT_petsc.mult(y_petsc, z_petsc)

    un = Function(V)
    with un.dat.vec as v:
        w_petsc = v

    #Construct a PETSc Krylov solver normal equations
    def monitor(ksp, it, rnorm):
        x = ksp.getSolution()
        r = x.duplicate()
        A_petsc.mult(x, r)
        r = r - b_petsc
        global res
        res = res + [r.norm()]
        print("\t||Ax - b|| = {}" .format(r.norm()))

    ksp = PETSc.KSP()
    ksp.create(comm=mesh.comm)
    #Solve the normal equations
    print("Solving normal equations")
    ksp.setOperators(ATMinvA, P_petsc)
    ksp.setOptionsPrefix("normal_eq_")
    ksp.setFromOptions()
    ksp.setMonitor(monitor)
    ksp.solve(z_petsc, w_petsc)
elif OptDB.getString('normal_eq_ksp_type', 'unspec') in ['cgne']:
    v_petsc = b_petsc.copy()
    z_petsc = b_petsc.copy()
    z_petsc.setArray(chol_factor.solve_L(chol_factor.apply_P(v_petsc.getArray()),use_LDLt_decomposition=False))
    un = Function(V)
    with un.dat.vec as v:
        w_petsc = v
    #Constructing Polar factor
    print("Constructing polar factor")
    Pdense = P_petsc.copy()
    Pdense.convert('dense')
    Pnp = Pdense.getDenseArray()
    R = la.sqrtm(Pnp)
    R_petsc =PETSc.Mat().createDense(A_petsc.getSize(),1,R)
    R_petsc.convert('aij')
    print("Polar factor constructed")
        
    #Construct a PETSc Krylov solver normal equations
    def monitor(ksp, it, rnorm):
        x = ksp.getSolution()
        r = x.duplicate()
        A_petsc.mult(x, r)
        r = r - b_petsc
        print("\t||Ax - b|| = {}" .format(r.norm()))
    ksp = PETSc.KSP()
    ksp.create(comm=mesh.comm)
    #Solve the normal equations
    print("Solving normal equations")
    ksp.setOperators(CinvA, R_petsc)
    ksp.setOptionsPrefix("normal_eq_")
    ksp.setFromOptions()
    ksp.setMonitor(monitor)
    ksp.solve(z_petsc, w_petsc)
elif OptDB.getString('normal_eq_ksp_type', 'unspec') in ['lsqr']:
    print("Solving normal equations using LSQR")
    v_petsc = b_petsc.copy()
    z_petsc = b_petsc.copy()
    z_petsc.setArray(chol_factor.solve_L(chol_factor.apply_P(v_petsc.getArray()),use_LDLt_decomposition=False))
    un = Function(V)
    with un.dat.vec as v:
        w_petsc = v
    #Construct a PETSc Krylov solver normal equations
    def monitor(ksp, it, rnorm):
        x = ksp.getSolution()
        r = x.duplicate()
        A_petsc.mult(x, r)
        r = r - b_petsc
        print("\t||Ax - b|| = {}".format(r.norm()))
    ksp = PETSc.KSP()
    ksp.create(comm=mesh.comm)
    #Solve the normal equations
    ksp.setOperators(CinvA, P_petsc)
    ksp.setOptionsPrefix("normal_eq_")
    ksp.setFromOptions()
    ksp.setMonitor(monitor)
    ksp.solve(z_petsc, w_petsc)
else:
    raise ValueError("Unknown solver type")


with un.dat.vec as v:
    v.copy(w_petsc)

from eps_utils import *
if spectral_equiv:
    print("SPECTRAL EQUIVALENCE")
    A, P = ksp.getOperators()
    lam_min, lam_max, kappa = spectral_equivalence_bounds(A, P, tol=1e-8, max_it=2000)
    est_it = 0.5*np.sqrt(kappa)*np.log(1e6)
    print(lam_min, lam_max, kappa, est_it)

#Output the solution
divbu = Function(W)
divbu.interpolate(exp(-inner(div(w*un),div(w*un)))) 
File('output/fem_advection_normal_eq_div_diagflow_{}.pvd'.format(float(nu))).write(divbu)
File('output/fem_advection_normal_eq_diagflow_{}.pvd'.format(float(nu))).write(un)
#Dump everything to a pandas dataframe
df = pd.DataFrame({'res': res})
df.to_csv('output/fem_advection_mass_normal_eq_diagflow_{}_{}.csv'.format(float(nu),float(n)))
