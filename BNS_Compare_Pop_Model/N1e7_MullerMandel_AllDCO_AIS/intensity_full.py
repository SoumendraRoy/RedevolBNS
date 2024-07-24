import jax.numpy as jnp
import numpy as np
import numpyro 
import numpyro.distributions as dist
import dataclasses
from dataclasses import dataclass
from utils import jnp_cumtrapz
import jax
import jax.scipy.special as jss
from astropy.cosmology import Planck18
import jax.scipy.stats as jsst

@dataclass
class FlatwCDMCosmology(object):
    """
    Function-like object representing a flat w-CDM cosmology.
    """
    h: object
    Om: object
    w: object
    zmax: object = 100.0
    ninterp: object = 10**5
    zinterp: object = dataclasses.field(init=False)
    dcinterp: object = dataclasses.field(init=False)
    dlinterp: object = dataclasses.field(init=False)
    ddlinterp: object = dataclasses.field(init=False)
    vcinterp: object = dataclasses.field(init=False)
    dvcinterp: object = dataclasses.field(init=False)

    def __post_init__(self):
        self.zinterp = jnp.expm1(np.linspace(np.log(1), np.log(1+self.zmax), self.ninterp))
        self.dcinterp = self.dH*jnp_cumtrapz(1/self.E(self.zinterp), self.zinterp)
        self.dlinterp = self.dcinterp*(1+self.zinterp)
        self.ddlinterp = self.dcinterp + self.dH*(1+self.zinterp)/self.E(self.zinterp)
        self.vcinterp = 4/3*np.pi*self.dcinterp*self.dcinterp*self.dcinterp
        self.dvcinterp = 4*np.pi*jnp.square(self.dcinterp)*self.dH/self.E(self.zinterp)

    @property
    def dH(self):
        return 2.99792 / self.h
    
    @property
    def Ol(self):
        return 1-self.Om
    
    @property
    def om(self):
        return self.Om*jnp.square(self.h)
    
    @property
    def ol(self):
        return self.Ol*jnp.square(self.h)
    
    def E(self, z):
        opz = 1 + z
        opz3 = opz*opz*opz
        return jnp.sqrt(self.Om*opz3 + (1-self.Om)*opz**(3*(1 + self.w)))

    def dC(self, z):
        return jnp.interp(z, self.zinterp, self.dcinterp)
    def dL(self, z):
        return jnp.interp(z, self.zinterp, self.dlinterp)
    def VC(self, z):
        return jnp.interp(z, self.zinterp, self.vcinterp)
    def dVCdz(self, z):
        return jnp.interp(z, self.zinterp, self.dvcinterp)
    
    def ddL_dz(self, z):
        return jnp.interp(z, self.zinterp, self.ddlinterp)

    def z_of_dC(self, dC):
        return jnp.interp(dC, self.dcinterp, self.zinterp)
    def z_of_dL(self, dL):
        return jnp.interp(dL, self.dlinterp, self.zinterp)

from astropy.cosmology import Planck18
import astropy.units as u
import scipy.integrate as si
import copy
from jax.scipy.stats import multivariate_normal

np.random.seed(123)
import h5py
variant = 'N1e7_MullerMandel_AllDCO_AIS'
with h5py.File("../gmm.h5",'r') as hf:
    mean = jnp.array(hf['mean_'+variant+'_all'])
    cov = jnp.array(hf['cov_'+variant+'_all'])
    weight = jnp.array(hf['weight_'+variant+'_all'])
hf.close()

def weighted_sum_logpdf(data, means, covariances, weights):

    logwpdf1 = jnp.log(weights[0])+jsst.multivariate_normal.logpdf(data, means[0], covariances[0])
    logwpdf2 = jnp.log(weights[1])+jsst.multivariate_normal.logpdf(data, means[1], covariances[1])
    logwpdf3 = jnp.log(weights[2])+jsst.multivariate_normal.logpdf(data, means[2], covariances[2])
    logwpdf4 = jnp.log(weights[3])+jsst.multivariate_normal.logpdf(data, means[3], covariances[3])
    logwpdf5 = jnp.log(weights[4])+jsst.multivariate_normal.logpdf(data, means[4], covariances[4])
    logwpdf6 = jnp.log(weights[5])+jsst.multivariate_normal.logpdf(data, means[5], covariances[5])
    logwpdf7 = jnp.log(weights[6])+jsst.multivariate_normal.logpdf(data, means[6], covariances[6])
    logwpdf8 = jnp.log(weights[7])+jsst.multivariate_normal.logpdf(data, means[7], covariances[7])
    logwpdf9 = jnp.log(weights[8])+jsst.multivariate_normal.logpdf(data, means[8], covariances[8])
    logwpdf10 = jnp.log(weights[9])+jsst.multivariate_normal.logpdf(data, means[9], covariances[9])
    logwpdf11 = jnp.log(weights[10])+jsst.multivariate_normal.logpdf(data, means[10], covariances[10])
    logwpdf12 = jnp.log(weights[11])+jsst.multivariate_normal.logpdf(data, means[11], covariances[11])
    logwpdf13 = jnp.log(weights[12])+jsst.multivariate_normal.logpdf(data, means[12], covariances[12])
    logwpdf14 = jnp.log(weights[13])+jsst.multivariate_normal.logpdf(data, means[13], covariances[13])
    logwpdf15 = jnp.log(weights[14])+jsst.multivariate_normal.logpdf(data, means[14], covariances[14])
    logwpdf16 = jnp.log(weights[15])+jsst.multivariate_normal.logpdf(data, means[15], covariances[15])
    return jss.logsumexp(jnp.array([logwpdf1, logwpdf2, logwpdf3, logwpdf4, logwpdf5, logwpdf6, logwpdf7, logwpdf8, logwpdf9, logwpdf10, logwpdf11, logwpdf12, logwpdf13, logwpdf14, logwpdf15, logwpdf16]), axis=0)

w = -1

def cosmo_model(Mczs, logqs, Thetas, ds, Mczsdetdraw, logqsdetdraw, Thetasdetdraw, dsdetdraw, Ndraw, optimal_snr):
    logqsr = -logqs
    Mczs, logqs, logqsr, ds = map(jnp.array, (Mczs, logqs, logqsr, ds))
    Nobs = ds.shape[0]
    Nsamp = ds.shape[1]
    
    h = numpyro.sample('h', dist.TruncatedNormal(loc=0.6766, scale=0.05, low=0.2, high=1.5))
    Om = numpyro.sample('Om', dist.TruncatedNormal(loc=0.30966, scale=0.05, low=0, high=1))
    #w = numpyro.sample('w', dist.Uniform(low=-1.5, high=-0.5))
    
    z = FlatwCDMCosmology(h,Om,w).z_of_dL(ds)
    
    Mc = Mczs/(1+z)
    var = -jnp.log(1+z)
    #var1 = -2.0*jnp.log(ds)
    
    ddLdz = FlatwCDMCosmology(h,Om,w).ddL_dz(z)
    var2 = -jnp.log(ddLdz)
    
    data = jnp.array([Mc.flatten(),logqs.flatten(),z.flatten()]).T
    var3 = weighted_sum_logpdf(data, mean, cov, weight).reshape(Nobs,Nsamp)
    data_r = jnp.array([Mc.flatten(),logqsr.flatten(),z.flatten()]).T
    var3_r = weighted_sum_logpdf(data_r, mean, cov, weight).reshape(Nobs,Nsamp)
    var3t = jss.logsumexp(jnp.array([var3, var3_r]), axis=0)
    
    var4 = jsst.beta.logpdf(Thetas, 2, 4)
    
    var5 = jss.logsumexp(var+var2+var3t+var4, axis=1)
    
    # selection function
    logqsrdetdraw = -logqsdetdraw
    Mczsdetdraw, logqsdetdraw, logqsrdetdraw, dsdetdraw = map(jnp.array, (Mczsdetdraw, logqsdetdraw, logqsrdetdraw, dsdetdraw))
    
    zdetdraw = FlatwCDMCosmology(h,Om,w).z_of_dL(dsdetdraw)
    
    Mcdetdraw = Mczsdetdraw/(1+zdetdraw)
    varsel = -jnp.log(1+zdetdraw)
    
    ddLdzdetdraw = FlatwCDMCosmology(h,Om,w).ddL_dz(zdetdraw)
    var2sel = -jnp.log(ddLdzdetdraw)
    
    data_sel = jnp.array([Mcdetdraw,logqsdetdraw,zdetdraw]).T
    var3_sel = weighted_sum_logpdf(data_sel, mean, cov, weight)
    data_r_sel = jnp.array([Mcdetdraw,logqsrdetdraw,zdetdraw]).T
    var3_r_sel = weighted_sum_logpdf(data_r_sel, mean, cov, weight)
    var3t_sel = jss.logsumexp(jnp.array([var3_sel, var3_r_sel]), axis=0)
    
    var4sel = jsst.beta.logpdf(Thetasdetdraw, 2, 4)
    var5sel = jss.logsumexp(varsel+var2sel+var3t_sel+var4sel)
    
    _ = numpyro.factor('pos', jnp.sum(var5)-Nobs*var5sel)