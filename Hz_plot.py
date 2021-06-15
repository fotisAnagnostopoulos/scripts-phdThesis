import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lambertw
from scipy.integrate import odeint
from scipy.optimize import newton

##### The following compilation of H(z) data is a subset of Yu, Ratra and Wang, 2018.
# We keep data obtained with CC method only.
# Ref: https://iopscience.iop.org/article/10.3847/1538-4357/aab0a2/pdf
hz = dataFromRatraNew = np.array([
        [0.070, 69.,19.6],
        [0.090,69.,12.],
        [0.120,68.6,26.2],
        [0.170,83.,8.],
        [0.179,75.,4.],
        [0.199,75.,5.],
        [0.200,72.9,29.6],
        [0.270,77.0,14.],
        [0.280,88.8,36.6],
        [0.352,83.,14.],
        [0.3802,83.,13.5],
        [0.400,95.,17.],
        [0.4004,77.,10.2],
        [0.4247,87.1,11.2],
        [0.4497,92.8,12.9],
        [0.47, 89.0, 50.0],
        [0.4783,80.9,9.],
        [0.480,97.,62.],
        [0.593,104.,13.],
        [0.680,92.,8.],
        [0.781,105.,12.],
        [0.875,125.,17.],
        [0.880,90.,40.],
        [0.900,117.,23.],
        [1.037,154.,20.],
        [1.300,168.0,17.],
        [1.363,160.,33.6],
        [1.430,177.,18.],
        [1.530,140.,14.],
        [1.750,202.,40.],
        [1.965,186.5,50.4] ])
z = hz[:,0]
Hvls = hz[:,1]
sigms = hz[:,2]

###################################### Hubble rates for different models######################################

def Hlcdm(z,Y):
    Om_m0 = Y[0]
    Ho = 100*Y[1]
    return Ho*np.power((Om_m0*np.power((1+z),3) + (1-Om_m0)),0.5)

def H_de(z,Y):
    Om_m0 = Y[0]
    h = Y[1]
    c = Y[2]
    y0 = 1 - Om_m0
    dOdz = lambda om_de,zz: -(1/(1+zz))*om_de*(1-om_de)*(1 + 2*(om_de**1.5)*(c)**0.5)
    O_de = odeint(dOdz,y0,z)[:,0]
    H = np.power((Om_m0*np.power((1+z),3) / (1-O_de)),0.5)
    return H*100*h

def HfQ_tl(z,Y):
    '''
    The Hubble rate for f(Q) = Qexp(aQ/Q_0).
    For details see arXiv:2104.15123 and refs therein
    '''
    Om_m0 = Y[0]
    h = Y[1]
    T_cmb = 2.7255
    Om_r = Om_m0/(1. + 2.5e4 * Om_m0* h**2. * (T_cmb/2.7)**(-4.))
    Og_mod = 31500. * (T_cmb/2.7)**(-4.)
    expr = (Om_m0 + Om_r)/(-2*np.exp(0.5))
    a = (0.5 + lambertw(expr,k=0)).real
    Om_l = 1. - Om_m0 - Om_r
    x0E = (Om_m0*np.power((1+z),3) + Om_r*np.power((1+z),4) + Om_l)**0.5
    equation_E = lambda E: (E**2.- 2*a)* np.exp(a*1./E**2)- Om_m0*np.power((1+z),3)- Om_r*np.power((1+z),4)
    res = newton(equation_E, x0E)*100*h
    return res
def H_bengochea(z,Y):
    '''
    This is the full numerical Hubble rate for f1_CDM, for details
    see arXiv:1907.07533 and references therein.
    '''
    h = Y[1]
    Om_m0 = Y[0]
    b = Y[2]
    T_cmb = 2.7255
    Om_r = Om_m0/(1. + 2.5e4 * Om_m0 * h**2. * (T_cmb/2.7)**(-4.))
    Om_l = 1.- Om_m0 - Om_r
    x0a = (Om_m0*(1.+z)**3. + Om_r*(1.+z)**4.+Om_l)**0.5
    equation_b = lambda E: E**2.-(Om_m0*(1.+z)**3. + Om_r*(1.+z)**4.+(Om_l)*(E**(2.*b)))
    res = newton(equation_b, x0a)*100*h
    return res
def H_kofinas(z,Y):
    '''
    Hubble function of the kofinas model, without curvature
    for details see arXiv:1806.10580 and references therein.
    '''
    Om = Y[0]
    b = 2.08
    A = Y[2]
    h = Y[1]
    H_0 = h * 100.
    res = (Om * (1.+z)**3. + ((1.-Om)**(-1./b) + A*(z/(1. + z) ) )**(-b))**0.5
    return res*H_0
###################################### Cosmological parameters ######################################
#lcdm from Aghanim et al, 2018, arXiv:1807.06209
Om_m0l = 0.315
h_l = 0.674
# Kofinas-Zarikas model, from Anagnostopoulos et al, 2018, arXiv:1806.10580
Om_kof = 0.270
h_kof = 0.684
A = -0.122
# Bengochea-Ferraro model, Anagnostopoulos et al, 2019 arXiv:1907.07533
Om_ft = 0.291
h_ft = 0.6921
b = 0.021
#Exponential f(Q) model, Anagnostopoulos et al 2021, arXiv:2104.15123
Om_fq = 0.356
h_fq = 0.6882
######## Holographic DE model, from Zhang and Wu, arXiv:astro-ph/0701405
c = 0.91
Om_hde = 0.29
######################################################################################################

### Plotting (a) the Hubble rate for \Lambda CDM with specific H_0 and for many allowed \Omega_{m0}
### and (b) the Hubble rate for \Lambda CDM along with competitors
n = 200
Ommatter_list = np.linspace(0.,1.,n)
zeds = np.linspace(0,2.0,100)
############################################
fig, (ax1, ax2) = plt.subplots(2,sharex=True)
ax1.plot(zeds,Hlcdm(zeds,[Om_m0l,h_l]),color='black',alpha=1,zorder=4)
for i in range(n):
    ax1.plot(zeds,Hlcdm(zeds,[Ommatter_list[i],h_l]),color='r',alpha=0.1,zorder=1)
ax1.errorbar(z,Hvls,yerr=sigms,fmt='.',color='black',capsize=3,zorder=3)
ax1.set(xlabel='$z$', ylabel='$H(z)[kms^{-1}Mpc^{-1}]$')
ax1.set_ylim([40,250])
ax1.set_xlim([0,2])
ax1.grid()

ax2.errorbar(z,Hvls,yerr=sigms,fmt='.',color='black',zorder=2,capsize=3)
ax2.plot(zeds,HfQ_tl(zeds,[Om_fq,h_fq]),color='r',label='f(Q)')
ax2.plot(zeds,H_de(zeds,[Om_m0l,h_l,c]),color='blue',label='HDE')
ax2.plot(zeds,Hlcdm(zeds,[Om_m0l,h_l]),color='black',label='$\Lambda CDM$')
ax2.plot(zeds,H_bengochea(zeds,[Om_ft,h_ft,b]),color='magenta',label='$f(T)$')
ax2.plot(zeds,H_kofinas(zeds,[Om_kof,h_kof,A]),color='green',label='KZ')
ax2.set(xlabel='$z$', ylabel='$H(z)[kms^{-1}Mpc^{-1}]$')
# ax2.legend()
ax2.legend(title='Models',bbox_to_anchor=(0.0, 1.00),ncol=3, loc='upper left', fontsize='small')
ax2.grid()
plt.tight_layout()
# plt.savefig("problem-Statement-fig.pdf")
plt.show()