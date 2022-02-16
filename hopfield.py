# -*- coding: utf-8 -*-

# TEAM MEMBERS

# Alejandro Jarabo Peñas
# aljp@kth.se

# Miguel Garcia Naude
# magn2@kth.se

# Jonne van Haastregt
# jmvh@kth.se 

# Load packages
import numpy as np
import matplotlib.pyplot as plt

# Auxiliary variables
colors = ['#1E90FF','#FF69B4']

# RADIAL BASIS FUNCTIONS
class Hopfield:
    def __init__(self, dim):
        """ Constructor of the Hopfield Network."""
        self.dim = dim
        self.W = np.zeros((dim,dim))

    def train_W(self,X) :
        # Compute weight matrix
        N = np.shape(X)[1]
        # Averaged outer vector
        W = (1/N)*(X.T @ X)
        # Fill diagonal with zeros
        np.fill_diagonal(W,0)
        self.W = np.copy(W)

    def plot_weights(self) :
        plt.imshow(self.W)
        plt.colorbar()

    def asynchronous_recall(self,x,n_steps) :
        # Store evolution of vector and energy over iterations
        xs = []
        energys = []
        energy_old = np.infty
        energy_new = energy(x, self.W)
        i = 0
        # we keep running until we reach the lowest energy level
        while (energy_old > energy_new) and i < n_steps: 
            # Update iteration and store values
            i += 1
            energy_old = energy_new
            xs.append(np.copy(x))
            energys.append(energy_old)
            # Asynchronous update
            for ind in np.random.permutation(range(self.dim)):
                x[:,ind] = np.sign(x @ self.W[:,[ind]]) 
            # Update energy function
            energy_new = energy(x, self.W)
        
        return xs, energys

    def synchronous_recall(self,x,n_steps) :
        # Store evolution of vector and energy over iterations
        xs = []
        energys = []
        energy_old = np.infty
        energy_new = energy(x, self.W)
        iteration = 0
        # we keep running until we reach the lowest energy level
        while (energy_old > energy_new) and i < n_steps: 
            # Update iteration and store values
            i += 1
            energy_old = energy_new
            xs.append(np.copy(x))
            energys.append(energy_old)
            # Synchronous update
            x = np.sign(self.W @ x)
            # Update energy function
            energy_new = energy(x, self.W)
        
        return xs, energys

# HELPER FUNCTIONS
def energy(x, W):
    return -0.5 * x @ W @ x.T

def show_patterns(X) :
    n_patterns = np.shape(X)[1]
    fig, ax = plt.subplots(1,n_patterns, figsize=(18,6))
    for i in range(n_patterns) :
        ax[i].imshow(X[:,[i]],cmap='binary')
        ax[i].set_title('x{}'.format(i+1))

def show_patterns_distorted(X,Xd,recovered=False) :
    n_patterns = np.shape(X)[1]
    fig, ax = plt.subplots(1,2*n_patterns, figsize=(18,6))
    for i in range(n_patterns) :
        ax[2*i].imshow(X[:,[i]],cmap='binary')
        ax[2*i].set_title('x{}'.format(i+1))
        ax[2*i+1].imshow(Xd[:,[i]],cmap='binary')
        ax[2*i+1].set_title('x{}d'.format(i+1) + '_rec' if recovered else '')
        



