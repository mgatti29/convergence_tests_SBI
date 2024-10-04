import matplotlib.pyplot as plt
import numpy as np
import sys
import healpy as hp
import os
import inspect
import multiprocessing as mp
sys.path.append('../code/')
from Theory_camb import theory,cosmo
from mpi4py import MPI 
import timeit

desy3_nz = np.loadtxt('../data/des_y3_nz.txt')

nzs = []
nzs.append(desy3_nz[desy3_nz[:, 0] == 0, 1])
    
for i in range(4):
    mask_bin = desy3_nz[:, 0] == i
    nzs.append(desy3_nz[mask_bin,2]/np.trapz(desy3_nz[mask_bin,2],desy3_nz[mask_bin,1]))
nzs = np.array(nzs).T



def kd(i, j):
    return 1 if i == j else 0

def generate_dv(s8, om, nzs, h = 0.67, ob = 0.0493, ns = 0.9649, mv = 0.02, n_bins = 4, sigma_eff = 0.26, n_eff_per_bin = 5.12/4., f_sky = 0.12,w = -1):

    Cosmo = cosmo(H0 = h * 100., ombh2 = ob * h**2, omch2 = (om - ob) * h**2, As = 2e-9, ns = ns, mnu = mv, num_massive_neutrinos = 3, w=w )
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

    # desy3 numbers
    arcmin2_to_sterad = 8.461e-8
    c = sigma_eff**2/(n_eff_per_bin /arcmin2_to_sterad)

    cov = np.zeros((len(cl_DV),len(cl_DV)))
    for index1 in range(len(cl_DV)):
        for index2 in range(len(cl_DV)):
            if index1 == index2:
                i = t1[index1]
                j = t2[index1]
                k = t1[index2]
                l = t2[index2]
                l_ = ll[index2]
                cov[index1,index2] = (1./(f_sky*(2.*l_+1)))* ((Theory.clgg[i,j][l_] + kd(i,j)*c)*(Theory.clgg[j,l][l_] + kd(j,l)*c)+(Theory.clgg[i,l][l_] + kd(i,l)*c)*(Theory.clgg[j,k][l_] + kd(j,k)*c))

          
    cl_DV += np.random.normal(0,np.sqrt(cov.diagonal()))

    return cl_DV

def run(i,s8_array,om_array,w_array,nzs):
    cl_DV = generate_dv(s8_array[i],om_array[i],nzs,w = w_array[i])
    return cl_DV,s8_array[i],om_array[i],w_array[i]
if __name__ == "__main__":


    st = timeit.default_timer()

    output_folder = '/pscratch/sd/m/mgatti/Anthony/'
    # lcdm runs ------
    number_or_runs = 10000
    s8_array =  np.random.uniform(0.5,1,number_or_runs)
    om_array =  np.random.uniform(0.1,0.4,number_or_runs)
    w_array = np.ones(number_or_runs)*-1.

   # '''
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    run_count = rank 


    while run_count < number_or_runs:
        if not os.path.exists(output_folder+'lcdm_{0}.npy'.format(run_count)):
            cl_DV,s,o,w = run(run_count,s8_array,om_array,w_array,nzs)
            np.save(output_folder+'lcdm_{0}'.format(run_count),{'output':[cl_DV,s,o,w]})
        run_count += size

    comm.Barrier()   
   # '''     
        
    # wcdm runs ------
    number_or_runs = 10000
    s8_array =  np.random.uniform(0.5,1,number_or_runs)
    om_array =  np.random.uniform(0.1,0.4,number_or_runs)
    w_array = np.random.uniform(-1.5,0-.5,number_or_runs)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    run_count = rank 


    while run_count < number_or_runs:
        if not os.path.exists(output_folder+'wcdm_{0}.npy'.format(run_count)):
            cl_DV,s,o,w = run(run_count,s8_array,om_array,w_array,nzs)
            np.save(output_folder+'wcdm_{0}'.format(run_count),{'output':[cl_DV,s,o,w]})
        run_count += size

    comm.Barrier()   
    
    end = timeit.default_timer()

    print ('')
    print (end-st)

'''  
module load python
source activate cmb_lensing_env
srun --nodes=4 --tasks-per-node=64 python parallel_computation_script.py
'''