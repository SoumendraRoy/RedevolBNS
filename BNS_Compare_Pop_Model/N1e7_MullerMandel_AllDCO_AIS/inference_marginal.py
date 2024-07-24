import h5py
import numpy as np
from intensity_marginal import cosmo_model
import jax
from numpyro.infer import MCMC, NUTS
import arviz as az
from scipy.interpolate import RectBivariateSpline

with h5py.File('../optimal_snr_CosmicExplorerWidebandP1600143.h5', 'r') as inp:
    ms = np.array(inp['ms'])
    osnrs = np.array(inp['SNR'])

osnr_interp = RectBivariateSpline(ms, ms, osnrs)

def optimal_snr(Mcz, logq, dl):
    m1z = Mcz*(np.exp(logq)**(-3/5))*(1+np.exp(logq))**(1/5)
    m2z = Mcz*(np.exp(logq)**(2/5))*(1+np.exp(logq))**(1/5)
    return osnr_interp.ev(m1z, m2z)/dl

def rho(optimal_snr, Theta):
    return optimal_snr*Theta

Ndraw = 50000000
Mczdraws = np.random.uniform(0.98,30.,Ndraw)
logqdraws = np.random.uniform(-1.,0.,Ndraw)
Thetadraws = np.random.uniform(0.,1.,Ndraw)
dldraws = np.random.uniform(0.,200.,Ndraw)

rhos = rho(optimal_snr(Mczdraws, logqdraws, dldraws), Thetadraws)

mask = rhos>8.0
Mczdetdraws = Mczdraws[mask]
logqdetdraws = logqdraws[mask]
Thetadetdraws = Thetadraws[mask]
dldetdraws = dldraws[mask]

nmcmc = 1000
nchain = 1

with h5py.File("inference_marginal.h5", "w") as file:
    for i in [1]:
        with h5py.File("mock_data.h5",'r') as hf:
            Mczs = np.array(hf['Mczs'+str(i)])[:800,:8000]
            logqs = np.array(hf['logqs'+str(i)])[:800,:8000]
            Thetas = np.array(hf['Thetas'+str(i)])[:800,:8000]
            ds = np.array(hf['ds'+str(i)])[:800,:8000]
        hf.close()
        
        print('(Nobs, Nsamp)=', Mczs.shape, logqs.shape, Thetas.shape, ds.shape,
              'Ndetdraw=', Mczdetdraws.shape[0], logqdetdraws.shape[0], Thetadetdraws.shape[0],
              dldetdraws.shape[0])

        kernel = NUTS(cosmo_model)
        mcmc = MCMC(kernel, num_warmup=nmcmc, num_samples=nmcmc, num_chains=nchain)
        mcmc.run(jax.random.PRNGKey(np.random.randint(1<<32)), Mczs, logqs, Thetas, ds,
                 Mczdetdraws, logqdetdraws, Thetadetdraws, dldetdraws, Ndraw)
        trace_full = az.from_numpyro(mcmc)

        h_infer = np.array(trace_full.posterior.h).flatten()
        Om_infer = np.array(trace_full.posterior.Om).flatten()
    
        file.create_dataset('h'+str(i), data=h_infer)
        file.create_dataset('Om'+str(i), data=Om_infer)
file.close()