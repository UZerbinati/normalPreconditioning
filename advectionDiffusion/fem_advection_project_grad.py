#python -i fem_advection_imp_2.py -lin_sys_ksp_monitor -lin_sys_pc_type gamg -lin_sys_ksp_atol 1e-8 -lin_sys_ksp_rtol 1e-18 -lin_sys_ksp_max_it 3 -mass_inv_ksp_type preonly -mass_inv_pc_type qr -normal_eq_ksp_rtol 1e-32 -normal_eq_ksp_atol 1e-8 -normal_eq_ksp_monitor -normal_eq_ksp_type fcg -normal_eq_pc_type lu -nu 1.25e-3 -n 32 -normal_eq_ksp_max_it 500

from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sksparse.cholmod import cholesky
import scipy.linalg as la
import scipy.sparse.linalg as spla

OptDB = PETSc.Options()
n = OptDB.getInt('n', 128)
nu = OptDB.getReal('nu', 1e-2)
refs = OptDB.getInt('ref', 2)

mesh = UnitSquareMesh(n, n)
mh = MeshHierarchy(mesh, refs)
mesh = mh[-1]
x,y = SpatialCoordinate(mesh)
V = RestrictedFunctionSpace(FunctionSpace(mesh,'CG',1),boundary_set=[1,2,3,4])
delta = Constant(1e-4)


# Define the advection velocity and viscosity
w = Constant((1.0, 1.0))
w = 1/(inner(w,w))*w
#w = as_vector([2*y*(1-x**2), -2*x*(1-y**2)])
wperp = Constant((0.0, 1.0))
nu = Constant(nu)

#Define the bilinear form
u = TrialFunction(V); v = TestFunction(V)
a = (nu*inner(grad(u), grad(v)) + inner(w, grad(u)) * v) * dx + delta*inner(w, grad(u))*inner(w, grad(v))*dx-delta*nu*inner(div(grad(u)), inner(w, grad(v)))*dx
m = nu*inner(grad(u), grad(v))*dx
p1 = (1/nu)*inner(u,v)*dx
p2 = nu*inner(grad(u), grad(v))*dx
c = (1/sqrt(nu))*inner(w*u, grad(v))*dx
d = inner(grad(u),grad(v))*dx 
# Define the linear form
f = Constant(1.0) * v * dx + delta*Constant(1.0)*inner(w, grad(v))*dx

A = assemble(a)
M = assemble(m)
P1 = assemble(p1)
P2 = assemble(p2)
C = assemble(c)
D = assemble(d)
b = assemble(f)
u = Function(V)

#Construct PETSc objects
A_petsc = A.M.handle
AT_petsc = A_petsc.copy()
AT_petsc = AT_petsc.transpose()
M_petsc = M.M.handle
P1_petsc = P1.M.handle
P2_petsc = P2.M.handle
C_petsc = C.M.handle
CT_petsc = C_petsc.copy()
CT_petsc = CT_petsc.transpose()
D_petsc = D.M.handle

#Construct the operator for the normal equations
D_sp = csr_matrix(D_petsc.getValuesCSR()[::-1], shape=P1_petsc.size)
D_sp_inv = spla.inv(D_sp)
D_inv_petsc = PETSc.Mat().createAIJ(size=D_petsc.size, csr=(D_sp_inv.indptr, D_sp_inv.indices, D_sp_inv.data))
projP1_petsc = CT_petsc.matMatMult(D_inv_petsc, C_petsc)
P_petsc = projP1_petsc + P2_petsc

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

    w = Function(V)
    with w.dat.vec as v:
        w_petsc = v

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
    ksp.setOperators(ATMinvA, P_petsc)
    ksp.setOptionsPrefix("normal_eq_")
    ksp.setFromOptions()
    ksp.setMonitor(monitor)
    ksp.solve(z_petsc, w_petsc)
elif OptDB.getString('normal_eq_ksp_type', 'unspec') in ['cgne']:
    v_petsc = b_petsc.copy()
    z_petsc = b_petsc.copy()
    z_petsc.setArray(chol_factor.solve_L(chol_factor.apply_P(v_petsc.getArray()),use_LDLt_decomposition=False))
    w = Function(V)
    with w.dat.vec as v:
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
    w = Function(V)
    with w.dat.vec as v:
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


with w.dat.vec as v:
    w.copy(w_petsc)

#Output the solution
File('output/fem_advection_normal_eq_imp.pvd').write(w)
