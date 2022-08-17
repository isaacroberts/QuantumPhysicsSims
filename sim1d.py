import numpy as np
import numba

import graph
from scipy import fft
import math

import time
import threading

import graph_util

import soundfile
import sys

def gauss_x(x, a, x0, k0):
    """
    a gaussian wave packet of width a, centered at x0, with momentum k0
    """

    return ((a * np.sqrt(np.pi)) ** (-0.5)
            * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x * k0))

def rand_cplx(shape=None):
    r = np.random.sample(shape)
    ang = np.pi * 2 * np.random.uniform(size=shape)
    return r ** np.exp(1j *  ang)

def rand_angl(shape=None):
    r = 1
    ang = np.pi * 2 * np.random.uniform(size=shape)
    return r ** np.exp(1j *  ang)

def rand_start(n, a, x0, k0):
    return rand_cplx(n.shape)

def rand_one(n, a, x0, k0):
    psi = np.zeros(n.shape, dtype=np.cdouble)
    i = np.random.randint(0, n.shape[0])
    psi[i] = rand_angl()
    return psi

def theta(x):
    """
    theta function :
      returns 0 if x<=0, and 1 if x>0
    """
    x = np.asarray(x)
    y = np.zeros(x.shape)
    return y


def square_barrier(x, width, height):
    return height * (theta(x) - theta(x - width))

class WaveFn():

    def __init__(self, Case, Bound):
        """

        """
        Cases = ['Bowl', 'Free', 'Tunnel']
        assert(Case in Cases)

        self.N = 2**11

        self.dt = .0001
        self.hbar = 1
        self.m = 1

        psize = .1
        L = 1

        dx = L * self.N

        # specify range in x coordinate
        N = self.N
        self.x = dx * (  np.linspace(-.5, .5, num=self.N))

        self.dx = self.x[1]-self.x[0]

        if Case=='Bowl':
            self.V_x = np.linspace(1, -1, num=self.N)**2* self.hbar
            x0 = (np.random.random()*2-1)  * dx * (1-psize*2.8/L) #* self.N
            k0=0

        elif Case=='Tunnel':
            pts = [(0, 1), (.25, .5), (.5, .55), (.75, 0), (1, 1)]
            xpts = np.array([p[0] for p in pts])
            ypts = np.array([p[1] for p in pts])
            import scipy.interpolate
            f = scipy.interpolate.interp1d(xpts, ypts, 'quadratic')
            x = np.linspace(0, 1, num=self.N)
            self.V_x = f(x) * self.hbar
            x0 = -.5 * dx/2
            k0=0

        elif Case=='Free':
            self.V_x = np.zeros(self.N)
            r = np.random.random()*2-1
            x0 = r  * dx/2 * (1-psize*3/L)
            print (f"x0= {r} x {dx}/2 * (1-{psize}*3/{L}) = {x0}")
            k0 = (np.random.sample()-.5)/2

        if Bound:
            self.B_S = max(5, self.N//32) #36
            bound_val = self.N ** 2
            self.V_x[:self.B_S] += np.linspace(bound_val, 0, num=self.B_S)
            self.V_x[-self.B_S:] += np.linspace(0, bound_val, num=self.B_S)

        d =  psize * self.N/2
        print ('initial size= ', d)

        self.dk = 2 * np.pi / (self.N * self.dx)
        self.k = self.dk * (np.linspace(-.5, .5, num=self.N)) * self.N

        k0 = self.dk/2 * k0 * self.N

        print ('k0', k0)

        psi_x0 = gauss_x(self.x, d, x0, k0)

        self.set_psi_x(psi_x0)
        self.minus_dt = False
        self.set_dt(self.dt)

        self.lock = threading.Lock()

    def set_dt(self, dt):
        self.dt = dt
        print ('dt =', self.dt)
        self.x_evolve_half = np.exp(-0.5 * 1j * self.V_x
                        / self.hbar * self.dt)
        self.k_evolve = np.exp(-0.5 * 1j * self.hbar / self.m
                        * (self.k * self.k) * self.dt)

    def set_int_dt(self, n):
        ac = self.lock.acquire(True, timeout=-1)
        if not ac:
            return
        print (n)
        dt = 2** (-(10-n))
        if self.minus_dt:
            dt = -dt
        self.set_dt(dt)
        self.lock.release()

    def run_update(self, end, it):
        its = 0
        while time.time() < end or its==0:
            if self.lock.acquire(True, timeout=-1):
                self.update()
                self.lock.release()
            its+=1

        self.psi_x = self.get_psi_x()

    def get_start_img(self):
        self.DS = 4
        self.hsv = np.zeros((self.N//self.DS,self.N//self.DS,3))
        self.hsv[:, :, :] = 1
        return self.hsv

    def update_image(self, oim, it, sec):
        self.run_update(sec, it)

        oim = graph_util.complex_to_rgb(self.psi_x[None, ::self.DS], self.hsv, 'ghost')
        return oim

    def get_start_line(self):
        line = np.zeros((1, self.N), np.cdouble)
        line[0] = self.get_psi_x()
        return line

    def update_line(self, line, it, sec):
        self.run_update(sec, it)

        line[0] = self.get_psi_x()
        return line

    def get_psi_x(self):
        return (self.psi_mod_x * np.exp(1j * self.k[0] * self.x)
                * np.sqrt(2 * np.pi) / self.dx)

    def get_psi_k(self):
        return self.psi_mod_k * np.exp(-1j * self.x[0] * self.dk
                                        * np.arange(self.N))

    def set_psi_x(self, psi_x):
        assert psi_x.shape == self.x.shape
        self.psi_mod_x = (psi_x * np.exp(-1j * self.k[0] * self.x) * self.dx / np.sqrt(2 * np.pi))

        self.psi_mod_x /= self.get_norm()
        self.compute_k_from_x()

    def set_psi_k(self, psi_k):
        assert psi_k.shape == self.x.shape
        self.psi_mod_k = psi_k * np.exp(1j * self.x[0] * self.dk
                                        * np.arange(self.N))
        self.compute_x_from_k()
        self.compute_k_from_x()

    def compute_k_from_x(self):
        self.psi_mod_k = fft.fft(self.psi_mod_x)

    def compute_x_from_k(self):
        self.psi_mod_x = fft.ifft(self.psi_mod_k)

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
        assert wave_fn.shape == self.x.shape
        return np.sqrt((abs(wave_fn) ** 2).sum() * 2 * np.pi / self.dx)

    def update(self):
        """
        psi = e**(i/h * (px- Et))

        h = h bar, planck constant
        E = energy
        p = momentum
        t = time

        #

        # Diffrentiate P/dx
            d(psi)/dx = i/h*p * e**(i/h * (px-Et))
        # Sub d(psi)/dx to replace original e^(...) term
            d(psi)/dx = i/h*p * psi
        # Rearrange
            p = -i * hBar * d/dx

    >>  d/dx = p/(-i * hBar)

        """

        #This rotates by the potential at each point
        self.psi_mod_x *= self.x_evolve_half

        self.compute_k_from_x()

        self.psi_mod_k *= self.k_evolve
        self.compute_x_from_k()

        self.psi_mod_x *= self.x_evolve_half
        self.compute_k_from_x()

        norm = self.get_norm()
        if abs(norm-1) > .000001:
            #Errors build up because of floating point errors
            print ('norm', self.get_norm())
            self.psi_mod_x /= norm
            self.compute_k_from_x()

        self.psi_x = self.get_psi_x()


print()
arg = sys.argv[1] if len(sys.argv)>1 else ''

case = 'Bowl'
if 'f' in arg:
    case='Free'
if 't' in arg:
    case = 'Tunnel'

bound = 'b' in arg

wavef = WaveFn(case, bound)

disp = graph.ComplexLineIn3D(wavef.update_line, wavef.get_start_line())

if 'r' in arg:
    disp.toggle_record()

multi = graph.MultiGraph([disp])

disp.number_press = wavef.set_int_dt

multi.start()

print()
