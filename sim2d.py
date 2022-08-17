import numpy as np
import numba

import scipy
from scipy.fft import fft2, ifft2, fftshift

import time
import threading
import math

import graph
import graph_util
from m_util import *

import sys


# Helper functions for gaussian wave-packets
def gauss_x(x, y, a, x0, y0, kx0, ky0):
    """
    a gaussian wave packet of width a, centered at x0, with momentum k0
    """
    momentum = (x*kx0)**2 + (y*ky0)**2
    momentum = momentum.astype(np.cdouble)
    g = np.exp( -.5 * ( \
                    ((x-x0) / (a))**2 +
                    ((y-y0) / (a))**2 +
                     1j* np.abs(np.sqrt(momentum))))

    g /= (np.abs(g)**2).sum()
    return g

class WaveFn():

    def __init__(self, FreeCase, Bound):
        self.shouldStop=False
        self.frames = 0
        self.kframes = 0
        self.steps = 0

        self.N = 256
        self.dt = .0001
        self.hbar = 1  # planck's constant
        self.m = 1

        psize = .1

        print ('dt =',self.dt)

        # specify range in x coordinate
        L = .5
        pos = (np.random.sample(2)-.5)
        xp, yp  = pos[0], pos[1]
        #Initial velocity
        vel = (np.random.sample(2)-.5)
        xk, yk = vel[0], vel[1]

        xl = self.N * L * (np.linspace(-.5, .5, num=self.N))
        self.dx = xl[1]-xl[0]
        x, y = np.meshgrid(xl, xl)
        size = L * self.N/2

        d = psize * self.N/2

        self.dk = math.tau/(self.N * self.dx)
        kl = self.dk  * np.linspace(-.5, .5, num=self.N) * self.N

        self.kSq = kl[:,None]**2 + kl[None,:]**2
        self.kSq = fftshift(self.kSq)
        self.k = np.sqrt(self.kSq)
        kl = []

        k0 = self.dk/2  * self.N

        print ('pos', xp*size, yp*size)
        print ('k0', xk*k0, yk*k0)
        self.psi_x = gauss_x(x, y, d, xp*size, yp*size, xk*k0, yk*k0)
        print (self.psi_x.shape)

        self.set_psi_x(self.psi_x)
        self.compute_k_from_x()
        self.psi_k = self.get_psi_k()

        self.V_x = np.zeros((self.N,self.N))

        if not FreeCase:
            hole_center = np.random.sample(2)-.5 * self.N
            hole_center[:]=0
            r_term = np.sqrt((x-hole_center[0])**2 + (y-hole_center[1])**2)/(size*np.sqrt(2))

            self.V_x += (r_term**2) * self.m
            self.V_x += 1e-2/(.01+r_term)
            self.V_x*=6

        print ('V max', self.V_x.max())

        if Bound:
            BS = 15
            bound_val = 1e5
            self.V_x[ :BS  ] += np.linspace(bound_val, 0, num=BS)[:,None]
            self.V_x[ -BS: ] += np.linspace(0, bound_val, num=BS)[:,None]
            self.V_x[:, :BS] += np.linspace(bound_val, 0, num=BS)[None,:]
            self.V_x[:,-BS:] += np.linspace(0, bound_val, num=BS)[None,:]

        self.V_x = fftshift(self.V_x)

        self.set_dt(self.dt)

        self.lock = threading.Lock()

    def set_dt(self, dt):
        self.dt = dt
        print ('dt =', self.dt)
        self.x_evolve_half = np.exp(-.5j * self.V_x
        / self.hbar * self.dt)

        self.k_evolve = np.exp(-.5j * self.hbar / self.m * self.kSq * self.dt)

    def set_int_dt(self, n):
        ac = self.lock.acquire(True, timeout=-1)
        if not ac:
            return

        dt = 2** (-(10-n))
        print (n)
        self.set_dt(dt)
        self.lock.release()

    def get_psi_x(self):
        psi_x = self.psi_mod_x
        psi_x = fftshift(psi_x)
        return norm(psi_x)

    def set_psi_x(self, psi_x):
        psi_x = fftshift(psi_x)
        self.psi_mod_x = psi_x
        self.psi_mod_x /= self.get_norm()
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
        self.psi_mod_k = fft2(self.psi_mod_x)

    def compute_x_from_k(self):
        self.psi_mod_x = ifft2(self.psi_mod_k)

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
        self.psi_mod_x *= self.x_evolve_half
        self.compute_k_from_x()

        self.psi_mod_k *= self.k_evolve
        self.compute_x_from_k()

        #Update potential would go here
        self.psi_mod_x *= self.x_evolve_half
        self.compute_k_from_x()

        norm = self.get_norm()
        if abs(norm-1) > .000001:
            #Errors build up because of floating point errors
            print ('norm', self.get_norm())
            self.psi_mod_x /= norm
            self.compute_k_from_x()

        if self.psi_x is None:
            self.psi_x = self.get_psi_x()
        if self.psi_k is None:
            self.psi_k = self.get_psi_k()

        # print ('end')

    def _interact(self, pos, arr, ax, ay, rng=50, force=False):
        rng = self.N//20

        X, Y = np.ogrid[:self.N, :self.N]
        dist_from_center = np.sqrt((X - ax)**2 + (Y-ay)**2)
        mask = dist_from_center <= rng

        magn = prob(arr)
        p = prob(arr[mask]) / magn

        if (force and p>0) or np.random.random() < p:
            arr[~mask]=0
        else:
            arr[mask]=0
        # Calling function will normalize
        return arr

    def interact_x(self, event):
        pos = event.pos
        W_SIZE = 800
        W_SIZE = 256
        pos = pos/W_SIZE
        #Left click = force
        force = 2 in event.buttons
        self._interact_x(pos, force)

    def _interact_x(self, pos, force=False):
        ax = pos[1] *self.psi_mod_x.shape[0]
        ay = pos[0] *self.psi_mod_x.shape[1]

        ac = self.lock.acquire(True, timeout=.2)
        if not ac:
            print ('dropped bc lock')
            return

        energy = self.calculate_total_energy()
        self.psi_x = self.get_psi_x()
        self.psi_x = self._interact(pos, self.psi_x, ax, ay, force=force)
        self.set_psi_x(self.psi_x)
        self.compute_k_from_x()
        self.psi_mod_x /= self.get_norm()

        # TODO
        # new_energy = self.calculate_total_energy()
        # #Preserve energy
        # self.psi_mod_k */= energy / new_energy
        # self.compute_x_from_k()

        self.lock.release()

    def interact_k(self, event):
        pos = event.pos
        print (event.buttons, pos)
        W_SIZE = 800
        W_SIZE = 256
        pos = pos/W_SIZE

        #Left click = force
        force = 2 in event.buttons
        self._interact_k(pos, force)

    def _interact_k(self, pos, force=False):
        ax = pos[1] *self.psi_mod_x.shape[0]
        # print('pos:', ax, ay)

        ac = self.lock.acquire(True, timeout=.2)
        if not ac:
            print ('dropped bc lock')
            return
        self.compute_k_from_x()
        energy = self.calculate_total_energy()
        self.psi_k = self.get_psi_k()

        self.psi_k = self._interact(pos, self.psi_k, ax, ay, force=force)
        self.set_psi_k(self.psi_k)
        self.compute_x_from_k()
        self.psi_mod_x /= self.get_norm()

        self.lock.release()

    def run_thread(self):
        print ('running')
        self.steps = 0
        while not self.shouldStop:
            self.lock.acquire(True, timeout=-1)
            self.update()
            self.steps += 1
            self.lock.release()
            time.sleep(.05)
            if self.steps % 200 == 0:
                print ('T =',self.steps)
        print()

    def get_psi_x_for_ui(self):
        if self.psi_x is None:
            return None
        x = self.psi_x *self.N
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
        return self.psi_x

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

    def get_start_line(self):
        line = np.zeros((1, self.N), np.cdouble)
        line[0] = self.get_psi_x()
        return line

    def update_line(self, line, it, sec):
        self.get_psi()
        line[0] = self.get_psi_x()
        return line

print()

arg = '' if len(sys.argv)==1 else sys.argv[1]
freeCase = 'f' in arg
bound = 'b' in arg
wavef = WaveFn(freeCase, bound)

# try:
thread = threading.Thread(target=wavef.run_thread)
thread.start()

col = 'hsv_black'
disp_x = graph.Display2D(wavef.update_image, wavef.get_start_img(), col, 'Position')
disp_k = graph.Display2D(wavef.update_k, wavef.get_start_k(), col, 'Momentum')

if 'r' in arg:
    disp_x.toggle_record()
    disp_k.toggle_record()

disp_x.on_mouse_release = wavef.interact_x
disp_k.on_mouse_release = wavef.interact_k

disp_x.number_press = wavef.set_int_dt
disp_k.number_press = wavef.set_int_dt

multi = graph.MultiGraph([disp_x, disp_k])

multi.start(False)

print()

print ('end')
wavef.shouldStop=True
thread.join()
