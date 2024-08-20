import matplotlib.pyplot as plt
import numpy as np
import sys
import healpy as hp
import inspect
import multiprocessing as mp
sys.path.append('../code/')
from Theory_camb import theory,cosmo

desy3_nz = np.loadtxt('../data/des_y3_nz.txt')

nzs = []
nzs.append(desy3_nz[desy3_nz[:, 0] == 0, 1])
    
for i in range(4):
    mask_bin = desy3_nz[:, 0] == i
    nzs.append(desy3_nz[mask_bin, 2] / np.trapz(desy3_nz[mask_bin, 1] * desy3_nz[mask_bin, 2], desy3_nz[mask_bin, 1]))

nzs = np.array(nzs).T

n_dv = mp.cpu_count()

s8_range = np.random.uniform(low = 0.64, high = 1.04, size = n_dv)
om_range = np.random.uniform(low = 0.16, high = 0.36, size = n_dv)

def kd(i, j):
    return 1 if i == j else 0

def generate_dv(s8, om, nzs, h = 0.67, ob = 0.0493, ns = 0.9649, mv = 0.02, n_bins = 4, sigma_eff = 0.26, n_eff_per_bin = 5.12/4., f_sky = 0.12):

    Cosmo = cosmo(H0 = h * 100., ombh2 = ob * h**2, omch2 = (om - ob) * h**2, As = 2e-9, ns = ns, mnu = mv, num_massive_neutrinos = 3)
    Theory = theory(cosmo = Cosmo, halofit_version = "takahashi", sigma_8 = s8, fast = True, max_zs = 2., lmax = 1000)

    Theory.get_Wshear(nzs)
    Theory.limber(xtype = "gg", nonlinear = True)
    
    l = (10**np.linspace(np.log10(10), np.log10(990), 20)).astype(int)
    t1 = []
    ll = []
    t2 = []
    cl_DV = []
    for i in range(n_bins):
        for j in range(n_bins):
            if i >= j:
                cl_DV.extend(Theory.clgg[i,j][l])
                t1.extend([i]*len(l))
                t2.extend([j]*len(l)) 
                ll.extend(l)

    c = sigma_eff**2/(n_eff_per_bin *3600 * 3282.0)

    cov = np.zeros((len(cl_DV),len(cl_DV)))
    for index1 in range(len(cl_DV)):
        for index2 in range(len(cl_DV)):
            if index1 == index2:
                i = t1[index1]
                j = t2[index1]
                k = t1[index2]
                l = t2[index2]
                l_ = ll[index2]
                cov[index1,index2] = (1./(f_sky*(2*l_+1)))* ((Theory.clgg[i,j][l_] + kd(i,j)*c)*(Theory.clgg[j,l][l_] + kd(j,l)*c)+(Theory.clgg[i,l][l_] + kd(i,l)*c)*(Theory.clgg[j,k][l_] + kd(j,k)*c))
    
    cl_DV += np.random.normal(0,np.sqrt(cov.diagonal()))

    return cl_DV


if __name__ == "__main__":
    num_cpus = mp.cpu_count()
    print(f"number of cpus is {num_cpus}")
    '''  
    with mp.Pool(num_cpus) as pool:
        params = [(s8_range[i], om_range[i], nzs) for i in range(n_dv)]
        results = pool.starmap(generate_dv, params)
    
    dict = {
            "s8": s8_range, 
            "om": om_range, 
            "final_DV": result for cl_DV in results
            }
    print(np.shape(cl_DV))

    '''
    import timeit
    st = timeit.default_timer()

    generate_dv(0.8,0.3,nzs)

    end = timeit.default_timer()

    print ('')
    print (end-st)



from mpi4py import MPI 
def function_run_treecorr(i,other_pars = ...):
    #e.g., run the i-th rho stat. here you have to load the right catalog and run treecorr.

run_count = 0
number_or_runs = 6 #e.g. 6 like the rho stats you need to compute.
while run_count<number_or_runs:
                comm = MPI.COMM_WORLD
                if run_count+comm.rank<number_or_runs:
        
                    function_run_treecorr(run_count+comm.rank,other_pars = other_pars)
                run_count+=comm.size
                comm.bcast(run_count,root = 0)
                comm.Barrier() 
10:13

