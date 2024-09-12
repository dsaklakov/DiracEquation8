import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn

def dirac_wavefunction(r, kappa, n, Z):
    # Constants
    alpha = 1/137  # Fine structure constant
    
    # Calculate energy levels
    gamma = np.sqrt(kappa**2 - (Z*alpha)**2)
    N = np.sqrt(n**2 + 2*n*gamma + kappa**2)
    energy = 1 / np.sqrt(1 + (Z*alpha/N)**2)
    
    # Normalization constant
    C = np.sqrt((1 - energy**2) * gamma * (n + gamma) / (2 * energy * n))
    
    # Upper and lower components
    g = C * spherical_jn(kappa, r * np.sqrt(1 - energy**2))
    f = C * np.sign(kappa) * np.sqrt((1 - energy) / (1 + energy)) * spherical_jn(-kappa-1, r * np.sqrt(1 - energy**2))
    
    return g, f

def plot_dirac_wavefunction(Z, n, kappa):
    r = np.linspace(0.01, 50, 1000)
    g, f = dirac_wavefunction(r, kappa, n, Z)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r, g, label='Upper component (g)')
    ax.plot(r, f, label='Lower component (f)')
    ax.set_xlabel('r')
    ax.set_ylabel('Wavefunction')
    ax.set_title(f'Dirac Wavefunction (Z={Z}, n={n}, κ={kappa})')
    ax.legend()
    ax.grid(True)
    
    return fig

# Streamlit app
st.title('Dirac Equation Visualization')

# User inputs
Z = st.slider('Atomic number (Z)', 1, 100, 1)
n = st.slider('Principal quantum number (n)', 1, 10, 1)
kappa = st.slider('Relativistic angular momentum (κ)', -10, 10, 1, 1)

# Generate and display plot
fig = plot_dirac_wavefunction(Z, n, kappa)
st.pyplot(fig)

# Explanation
st.markdown("""
This visualization shows the radial part of the Dirac wavefunction for a hydrogen-like atom.
The upper (g) and lower (f) components of the wavefunction are plotted against the radial distance.

- Z: Atomic number of the nucleus
- n: Principal quantum number
- κ: Relativistic angular momentum quantum number

The Dirac equation provides a relativistic description of fermions, such as electrons,
taking into account both positive and negative energy states.
""")