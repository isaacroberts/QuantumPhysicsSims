### Quantum Physics Simulations

Some Numpy simulations that I wrote to help myself visually understand quantum mechanics. These simulations use the Schrodinger equation, meaning they only apply to bosons (Photons, W, Z, and Higgs). I'm working on a simulation of the Dirac equation for Fermions, 

# This is currently under construction, as I'm very busy right now

In quantum mechanics, it's easier to think of quantum "particles" as waves that sometimes localize into one spot (which we call "measurement"). A lot of quantum mechanics stop being so weird when you think about it as waves. My rule of thumb is to think about if sound does the same thing.

* Also, I'm going to keep using the term "particle" because that's what everyone else uses, but I don't really like the term since they behave more like waves. 

### Position & Momentum

Here we see the position and momentum of the wave. 

\[Image here] 

Momentum is the fourier transform of the wave, meaning its frequency. A higher frequency wave goes faster.

\[Image of side-by-side low-frequency and high-freqency 1d wave]

In addition, a wave can have two frequencies "superimposed" on top of each other. This looks a lot like two particles moving at different speeds, which fits into the broader principle of superposition. 

\[Image here]

Something about heisenburg's uncertainty principle and FFTs

Heisenberg uncertainty principle for sound

### Title 

This code shows off the 1, 2, and 3 dimensional cases for single particles in orbital wells, free particles, and particle-in-a-box.

![Gif](https://github.com/isaacroberts/QuantumPhysicsSims/blob/main/hi%20res%20bounce%201d.gif)

https://user-images.githubusercontent.com/26425166/185217882-80321175-87c9-4975-b94e-9a55109d8507.mov


3D Harmonics from a particle-in-a-box
https://user-images.githubusercontent.com/26425166/185218768-ec065aa0-3f4c-403c-a41c-047f31a955b5.mp4

### Quantum Interactions / Observations 

![medium rainfall orbital bound_1](https://user-images.githubusercontent.com/26425166/185244875-600f44e9-9811-4a06-b865-183735874e42.gif)

Here, the wavicle is being "interacted with" at random points at each time step. Each time, the interaction has a chance to find the particle at that location, with a probability equal to the sum of the amplitudes squared. 

Most of the interactions miss, which sets the probability of that area to zero. However, some hit, which localizes the particle entirely within that region. After being "hit", the particle scatters in a principle that I can't currently find the name of

You will notice that the momentum spreads out when the waveicle is localized, this is because of the heisenberg uncertainty principle 


### Credit

Big thanks to [pySchrodinger](https://github.com/jakevdp/pySchrodinger "pySchrodinger") and [Philip Mocz's Medium Article](https://levelup.gitconnected.com/create-your-own-quantum-mechanics-simulation-with-python-51e215346798?gi=9b16411cffee  "Philip Mocz"), who provided the initial code for this project. 

For a better explanation of the math behind these simulations, read the blog posts linked above. 



### Controls

0-9 to speed up or slow down
R to record (if you have cv2 & imageio installed)
Space pauses the visual but not the simulation

Drag to rotate
Scroll to resize
Shift+Drag to pan 
Ctrl+Shift+Drag to change FOV on the 3d viewports

### Command line Params:
F: Free case, forces acting on the particle
T: Tunneling case, sets up a split potential to show quantum tunneling
B: Bound, puts forces at the walls to prevent the particle from wrapping around the screen 
R: Record, starts recording the instant playback is started 
