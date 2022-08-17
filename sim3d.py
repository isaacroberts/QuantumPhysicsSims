import numpy as np
import numba

import scipy
from scipy.fft import fftn, ifftn, fftshift

import time
import threading
import math

import graph
import graph_util
from m_util import *
import sys

# Helper functions for gaussian wave-packets
def gauss_x(x, a, x0, y0, z0, kx0, ky0, kz0):
    """
    a gaussian wave packet of width a, centered at x0, with momentum k0
    """

    momentum = \
        (kx0 * x[:,None,None])**2 + \
        (ky0 * x[None,:,None])**2 + \
        (kz0 * x[None,None,:])**2
    momentum = momentum.astype(np.cdouble)

    g = np.exp( -.5 * ( \
                    ((x[:,None,None]-x0) / (a))**2 +
                    ((x[None,:,None]-y0) / (a))**2 +
                    ((x[None,None,:]-z0) / (a))**2 +
                     1j* np.sqrt(momentum)))

    g /= (np.abs(g)**2).sum()
    return g

class WaveFn():

    def __init__(self, FreeCase, Bound):
        self.shouldStop=False
        self.frames = 0
        self.kframes = 0
        self.steps = 0

        self.N = 32
        assert(self.N <= 64)
        self.dt = .001
        self.hbar = 1   # planck's constant
        self.m = .1

        psize = .1
        L = .5

        #Initial position
        pos = (np.random.sample(3)-.5)
        xp, yp, zp  = pos[0], pos[1], pos[2]
        #Initial velocity
        vel = np.random.sample(3)*2 - 1
        xk, yk, zk = vel[0], vel[1], vel[2]

        # specify range in x coordinate
        dx = L

        xl = dx * self.N* (np.linspace(-.5, .5, num=self.N, dtype=np.csingle))
        # x, y = np.meshgrid(xl, xl)

        size = (xl[-1]-xl[0])

        self.dx = xl[1]-xl[0]

        # specify initial momentum and quantities derived from it
        d =  psize * self.N
        print ('initial width = ', d)

        self.dk = (math.tau) / (self.dx * self.N)
        kl = self.dk * np.linspace(-1, 1, num=self.N, dtype=np.csingle)

        kl = kl**2
        self.kSq = kl[:,None,None] + kl[None,:,None] + kl[None,None,:]
        self.kSq = fftshift(self.kSq)

        kx, ky, kl = [],[],[]

        if FreeCase:
            k0 = self.dk/2 * self.N
        else:
            k0=0
        s0 = L*self.N/4
        print ('pos', xp*s0, yp*s0, zp*s0)

        self.psi_x = gauss_x(xl, d, xp*s0, yp*s0, zp*s0, xk*k0, yk*k0, zk*k0)
        print (self.psi_x.shape)
        self.set_psi_x(self.psi_x)
        self.compute_k_from_x()
        self.psi_k = self.get_psi_k()
        print ('K Mag = ', mag(self.psi_mod_k))

        self.V_x = np.zeros((self.N,self.N, self.N), np.csingle)

        if not FreeCase:
            r_term = np.sqrt(xl[:,None,None]**2 +
                             xl[None,:,None]**2 +
                             xl[None,None,:]**2 )/(size*np.sqrt(3))

            self.V_x += 10 * r_term**2
            self.V_x += 1/(1+r_term)

        if Bound:
            BS = 3
            bound_val = 1e3
            self.V_x[ :BS  ] += bound_val
            self.V_x[ -BS: ] += bound_val
            self.V_x[:, :BS] += bound_val
            self.V_x[:,-BS:] += bound_val
            self.V_x[:, :, :BS] += bound_val
            self.V_x[:, :, -BS:]+= bound_val

        self.V_x = fftshift(self.V_x)

        self.set_dt(self.dt)

        self.lock = threading.Lock()


    def set_dt(self, dt):
        self.dt = dt
        print ('dt =', self.dt)
        self.x_evolve_half = np.exp(-.5j * self.V_x
                                     / self.hbar * self.dt)

        self.k_evolve = np.exp(-.5j * self.hbar / self.m
                                * self.kSq * self.dt)


    def set_int_dt(self, n):
        ac = self.lock.acquire(True, timeout=-1)
        if not ac:
            return

        dt = 2** (-(10-n))
        self.set_dt(dt)
        self.lock.release()

    def get_psi_x(self):
        psi_x = self.psi_mod_x
        psi_x = fftshift(psi_x)
        return norm(psi_x)

    def set_psi_x(self, psi_x):
        psi_x = fftshift(psi_x)
        psi_x = norm(psi_x)
        self.psi_mod_x = psi_x
        self.compute_k_from_x()

    def get_psi_k(self):
        psi_k = self.psi_mod_k
        psi_k = fftshift(psi_k)
        return psi_k

    def set_psi_k(self, psi_k):
        psi_k = fftshift(psi_k)
        self.psi_mod_k = psi_k
        self.compute_x_from_k()
        self.compute_k_from_x()

    def compute_k_from_x(self):
        self.psi_mod_k = fftn(self.psi_mod_x, workers=-1)

    def compute_x_from_k(self):
        self.psi_mod_x = ifftn(self.psi_mod_k, workers=-1)

    def get_norm(self):
        return self.wf_norm(self.psi_mod_x)

    def wf_norm(self, wave_fn):
        """
        Returns the norm of a wave function.

        Parameters
        ----------
        wave_fn : array
            Length-N array of the wavefunction in the position representation
        """
        assert wave_fn.shape == self.kSq.shape
        return np.sqrt((abs(wave_fn) ** 2).sum() * 2 * np.pi / self.dx)

    def update(self):
        """
        psi = e**(i/h * (px- Et))

        h = h bar, planck constant
        E = energy
        p = momentum
        t = time

        # Diffrentiate P/dx
            # d(psi)/dx = i/h*p * e**(i/h * (px-Et))
        # Sub d(psi)/dx to replace original e^(...) term
            d(psi)/dx = i/h*p * psi
        # Rearrange
            p = -i * hBar * d/dx

    >>  d/dx = p/(-i * hBar)

        """

        self.psi_mod_x *= self.x_evolve_half
        self.compute_k_from_x()

        self.psi_mod_k *= self.k_evolve
        self.compute_x_from_k()

        #Update potential would go here

        self.psi_mod_x *= self.x_evolve_half
        self.compute_k_from_x()

        norm = self.get_norm()
        if abs(norm-1) > .0001:
            #Errors build up because of floating point errors
            print ('norm', self.get_norm())
            self.psi_mod_x /= norm
            self.compute_k_from_x()

        if self.psi_x is None:
            self.psi_x = self.get_psi_x()
        if self.psi_k is None:
            self.psi_k = self.get_psi_k()

    def run_thread(self):
        print ('running')
        self.steps = 0
        while not self.shouldStop:
            self.lock.acquire(True, timeout=-1)
            self.update()
            self.lock.release()
            self.steps += 1
            time.sleep(.01)
        print()

    def get_psi_x_for_ui(self):
        if self.psi_x is None:
            return None
        x = self.psi_x * self.N
        self.psi_x = None
        return x

    def get_psi_k_for_ui(self):
        if self.psi_k is None:
            return None
        k = self.psi_k
        self.psi_k = None
        return k

    def get_start_img(self):
        return self.psi_x

    def get_start_k(self):
        return self.psi_k

    def update_image(self, oim, it, sec):
        if self.frames > self.steps and self.frames != 0:
            return None
        self.frames += 1
        return self.get_psi_x_for_ui()

    def update_k(self, oim, it, sec):
        if self.kframes > self.steps and self.kframes != 0:
            return None
        self.kframes += 1
        return self.get_psi_k_for_ui()

print()

arg = '' if len(sys.argv)==1 else sys.argv[1]
freeCase = 'f' in arg
bound = 'b' in arg
wavef = WaveFn(freeCase, bound)

thread = threading.Thread(target=wavef.run_thread)
thread.start()

cmap = 'abs'
disp_x = graph.Voxel3D(wavef.update_image, wavef.get_start_img(), title='Position', cmap=cmap)
disp_k = graph.Voxel3D(wavef.update_k, wavef.get_start_k(), title='Momentum', cmap=cmap)

disp_x.camera.link(disp_k.camera)

disp_x.number_press = wavef.set_int_dt
disp_k.number_press = wavef.set_int_dt

if 'r' in arg:
    disp_x.toggle_record()
    disp_k.toggle_record()

multi = graph.MultiGraph([disp_x, disp_k])

multi.start(False)


print()
print ('end')
wavef.shouldStop=True
thread.join()
