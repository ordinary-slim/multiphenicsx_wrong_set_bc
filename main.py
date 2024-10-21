from dolfinx import fem, mesh, io
import ufl
import numpy as np
from mpi4py import MPI
import multiphenicsx
import multiphenicsx.fem.petsc
import petsc4py
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def main():
    # Mesh and problems
    points_per_side = 8
    domain  = mesh.create_unit_square(MPI.COMM_WORLD, points_per_side, points_per_side, mesh.CellType.quadrilateral)
    tdim = domain.topology.dim
    CG1  = fem.functionspace(domain, ("Lagrange", 1),)
    DG0  = fem.functionspace(domain, ("Discontinuous Lagrange", 0),)
    domain.topology.create_entities(tdim)
    domain.topology.create_entities(tdim-1)
    domain.topology.create_connectivity(tdim,tdim)
    cell_map = domain.topology.index_map(tdim)

    # Activation
    active_els = fem.locate_dofs_geometrical(DG0, lambda x : x[0] <= 0.5 )
    active_els_func = fem.Function(DG0,name="active_els")
    active_els_func.x.array[active_els] = 1.0
    active_dofs = fem.locate_dofs_topological(CG1, tdim, active_els,remote=True)
    restriction = multiphenicsx.fem.DofMapRestriction(CG1.dofmap, active_dofs)

    # Parameters
    dt = fem.Constant(domain, 0.1)
    k   = fem.Constant(domain, 1.0)
    c_p = fem.Constant(domain, 1.0)
    rho = fem.Constant(domain, 1.0)
    T_hot = 0.0
    T_cold = 2.0
    num_timesteps = 5

    unp1 = fem.Function(CG1, name="unp1")
    un   = fem.Function(CG1, name="un")
    udc  = fem.Function(CG1, name="dirichlet")
    du   = fem.Function(CG1, name="delta_uh")
    alg_rhs = fem.Function(CG1, name="rhs")

    # Dirichlet BC
    def left_marker_dirichlet(x):
        return np.logical_or( np.isclose(x[1],1), np.logical_or(
                np.isclose(x[0],0), np.isclose(x[1],0)) )
    dirichlet_dofs = fem.locate_dofs_geometrical(CG1,left_marker_dirichlet)
    dirichlet_bcs = [fem.dirichletbc(udc, dirichlet_dofs)]
    
    # Forms
    local_active_els = active_els_func.x.array.nonzero()[0][:np.searchsorted(active_els_func.x.array.nonzero()[0], cell_map.size_local)]
    subdomain_idx = 1
    dx = ufl.Measure("dx", subdomain_data=[(subdomain_idx,local_active_els)])
    (u, v) = (ufl.TrialFunction(CG1), ufl.TestFunction(CG1))
    a_ufl = k*ufl.dot(ufl.grad(u), ufl.grad(v))*dx(subdomain_idx)
    a_ufl += (rho*c_p/dt)*u*v*dx(subdomain_idx)
    l_ufl = (rho*c_p/dt)*un*v*dx(subdomain_idx)

    a_compiled = fem.form(a_ufl)
    l_compiled = fem.form(l_ufl)

    '''
    # Normal solve
    A = multiphenicsx.fem.petsc.create_matrix(a_compiled, (restriction, restriction),)
    L = multiphenicsx.fem.petsc.create_vector(l_compiled, restriction)
    x = multiphenicsx.fem.petsc.create_vector(l_compiled, restriction=restriction)

    time = 0.0
    unp1.x.array[:] = T_cold
    udc.x.array[:] = T_hot
    vtk_writer = io.VTKFile(domain.comm, f"post/normal.pvd", "wb")
    for titer in range(num_timesteps):
        time += dt.value
        un.x.array[:] = unp1.x.array[:]
        # ASSEMBLE
        A.zeroEntries()
        multiphenicsx.fem.petsc.assemble_matrix(A,
                                                a_compiled,
                                                bcs=dirichlet_bcs,
                                                restriction=(restriction, restriction))
        A.assemble()
        with L.localForm() as l_local:
            l_local.set(0.0)
        multiphenicsx.fem.petsc.assemble_vector(L,
                                                l_compiled,
                                                restriction=restriction,)
        multiphenicsx.fem.petsc.apply_lifting(L, [a_compiled], [dirichlet_bcs], restriction=restriction,)
        L.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        multiphenicsx.fem.petsc.set_bc(L,dirichlet_bcs,restriction=restriction)
        # LINEAR SOLVE
        opts = {"pc_type" : "lu", "pc_factor_mat_solver_type" : "mumps",}
        with x.localForm() as x_local:
            x_local.set(0.0)
        ksp = petsc4py.PETSc.KSP()
        ksp.create(domain.comm)
        ksp.setOperators(A)
        ksp_opts = PETSc.Options()
        for key,value in opts.items():
            ksp_opts[key] = value
        ksp.setFromOptions()
        ksp.solve(L, x)
        x.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
        ksp.destroy()
        # Unrestrict solution
        with unp1.x.petsc_vec.localForm() as usub_vector_local, \
                multiphenicsx.fem.petsc.VecSubVectorWrapper(x, CG1.dofmap, restriction) as x_wrapper:
                    usub_vector_local[:] = x_wrapper
        unp1.x.scatter_forward()

        vtk_writer.write_function([unp1,un,udc,active_els_func],t=time)
    vtk_writer.close()
    for la_ds in [A, L, x]:
        la_ds.destroy()
    '''

    # Incremental solve
    (u, v) = (unp1, ufl.TestFunction(CG1))
    a_ufl = k*ufl.dot(ufl.grad(u), ufl.grad(v))*dx(subdomain_idx)
    a_ufl += (rho*c_p/dt)*u*v*dx(subdomain_idx)
    l_ufl = (rho*c_p/dt)*un*v*dx(subdomain_idx)
    r_ufl = a_ufl - l_ufl

    j_ufl = ufl.derivative(a_ufl,unp1)
    j_compiled = fem.form(j_ufl)
    r_compiled = fem.form(r_ufl)
    A = multiphenicsx.fem.petsc.create_matrix(j_compiled, (restriction, restriction),)
    L = multiphenicsx.fem.petsc.create_vector(r_compiled, restriction)
    x = multiphenicsx.fem.petsc.create_vector(r_compiled, restriction=restriction)

    time = 0.0
    unp1.x.array[:] = T_cold
    udc.x.array[:] = T_hot
    vtk_writer = io.VTKFile(domain.comm, f"post/incremenetal.pvd", "wb")
    for titer in range(num_timesteps):
        time += dt.value
        un.x.array[:] = unp1.x.array[:]
        # ASSEMBLE
        A.zeroEntries()
        multiphenicsx.fem.petsc.assemble_matrix(A,
                                                j_compiled,
                                                bcs=dirichlet_bcs,
                                                restriction=(restriction, restriction))
        A.assemble()
        with L.localForm() as l_local:
            l_local.set(0.0)
        multiphenicsx.fem.petsc.assemble_vector(L,
                                                r_compiled,
                                                restriction=restriction,)
        L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        L.scale(-1)

        # Dirichlet
        multiphenicsx.fem.petsc.apply_lifting(L, [j_compiled], [dirichlet_bcs], x0=[unp1.x.petsc_vec], restriction=restriction)
        multiphenicsx.fem.petsc.set_bc(L,dirichlet_bcs, x0=unp1.x.petsc_vec, restriction=restriction)
        L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

        # RHS to function
        with alg_rhs.x.petsc_vec.localForm() as lfunsub_vector_local, \
                multiphenicsx.fem.petsc.VecSubVectorWrapper(L, CG1.dofmap, restriction) as l_wrapper:
                    lfunsub_vector_local[:] = l_wrapper
        alg_rhs.x.scatter_forward()


        # LINEAR SOLVE
        opts = {"pc_type" : "lu", "pc_factor_mat_solver_type" : "mumps",}
        with x.localForm() as x_local:
            x_local.set(0.0)
        ksp = petsc4py.PETSc.KSP()
        ksp.create(domain.comm)
        ksp.setOperators(A)
        ksp_opts = PETSc.Options()
        for k,v in opts.items():
            ksp_opts[k] = v
        ksp.setFromOptions()
        ksp.solve(L, x)
        x.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
        ksp.destroy()
        # Unrestrict solution
        with du.x.petsc_vec.localForm() as dusub_vector_local, \
                multiphenicsx.fem.petsc.VecSubVectorWrapper(x, CG1.dofmap, restriction) as x_wrapper:
                    dusub_vector_local[:] = x_wrapper
        du.x.scatter_forward()
        unp1.x.array[:] += du.x.array[:]

        vtk_writer.write_function([unp1,un,udc,active_els_func,alg_rhs],t=time)
    vtk_writer.close()
    for la_ds in [A, L, x]:
        la_ds.destroy()

if __name__=="__main__":
    main()
