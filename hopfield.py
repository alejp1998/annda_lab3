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
import itertools

# Auxiliary variables
colors = ['#1E90FF','#FF69B4']

# HOPFIELD NETWORK
class Hopfield:
    def __init__(self, dim):
        """ Constructor of the Hopfield Network."""
        self.dim = dim
        self.W = np.zeros((dim,dim))

    def train_W(self,X,activity=0,zero_diag=False) :
        # Compute weight matrix
        N = X.shape[1]
        if activity == 0:
            W = (1/N)*(X.T @ X)
        else:
            W = ((X-activity).T @ (X-activity))
        if zero_diag :
            # Fill diagonal with zeros
            np.fill_diagonal(W,0)
        self.W = np.copy(W)

    def random_W(self,symmetric=False) :
        # Compute random weights matrix
        if symmetric :
            W_rand = np.random.normal(size=(self.dim,self.dim))
            self.W = 0.5*(W_rand+W_rand.T)
        else :
            self.W = np.random.normal(size=(self.dim,self.dim))

    def plot_weights(self) :
        plt.imshow(self.W)
        plt.colorbar()

    def find_attractors(self,n_steps) :
        # Generate possible input vectors
        X = np.array(list(itertools.product([-1, 1], repeat=self.dim)))
        n_combs = X.shape[0]
        # Apply update rule to all of them and find store attractors
        attractors = []
        for i in range(n_combs) :
            # Input pattern
            x = X[[i],:]
            # Iterate until we reach attractor
            xs, energys = self.async_recall(x,n_steps)
            attractor = list(xs[-1].reshape(-1))
            # Append attractor if not already in the list
            if attractor not in attractors :
                attractors.append(attractor)

        return np.array(attractors)

    def stable_memories(self,Xorig,Xd,bias=0) :
        # Find # of patterns that remain stable
        n_patterns = Xd.shape[0]
        n_matches = 0
        for i in range(n_patterns) :
            # Input pattern
            x = Xd[[i],:]
            # Run one step async recall over pattern
            xs, energys = self.async_recall(x,2,bias)
            # Check if it matches with assoc memory
            n_matches += (Xorig[[i],:] == xs[-1]).all()
        
        return n_matches

    def async_recall(self,x,n_steps,bias=0) :
        # Store evolution of vector and energy over iterations
        xs = []
        energys = []
        energy_old = np.infty
        energy_new = energy(x, self.W)
        i = 0
        n_patterns = x.shape[0]
        # We keep running until we reach the lowest energy level
        while (energy_old > energy_new) and i < n_steps: 
            # Update iteration and store values
            i += 1
            energy_old = energy_new
            xs.append(np.copy(x))
            energys.append(energy_old)
            # Asynchronous update
            for ind in np.random.permutation(range(self.dim)):
                if bias == 0:
                    x[:,ind] = np.sign(x @ self.W[:,[ind]]) 
                else:
                    x[:,ind] = 0.5 + 0.5*np.sign(x @ self.W[:,[ind]] - bias)
                    sumj = 0
                    for k in range(n_patterns):
                        sumj = sumj + x @ self.W[:,[k]] - bias
                    x[:,ind] = 0.5 + 0.5*sumj

            # Update energy function
            energy_new = energy(x, self.W)
        
        return xs, energys

    def sync_recall(self,x,n_steps) :
        # Store evolution of vector and energy over iterations
        xs = []
        energys = []
        energy_old = np.infty
        energy_new = energy(x, self.W)
        i = 0
        # We keep running until we reach the lowest energy level
        while (energy_old > energy_new) and i < n_steps: 
            # Update iteration and store values
            i += 1
            energy_old = energy_new
            xs.append(np.copy(x))
            energys.append(energy_old)
            # Synchronous update
            x = np.sign(x @ self.W)
            # Update energy function
            energy_new = energy(x, self.W)
        
        return xs, energys

# HELPER FUNCTIONS
def energy(x, W):
    return float(-0.5 * x @ W @ x.T)

def distort_patterns(X,n_bits) :
    Xd = np.copy(X)
    n_patterns = X.shape[0]
    dim = X.shape[1]
    bits_indices = list(range(dim))
    for i in range(n_patterns) :
        bits_to_distort = np.random.choice(bits_indices,size=n_bits,replace=False)
        for bit_index in bits_to_distort :
            Xd[i,bit_index] = -X[i,bit_index]
    return Xd

def random_patterns(dim,n_patterns,bias=0) :
    if bias == 0 : 
        X = np.random.randint(low=0,high=2,size=(n_patterns,dim))*2 - 1
    else :
        X = np.sign(bias + np.random.normal(size=(n_patterns,dim)))
    return X

def show_patterns(X,name='') :
    n_patterns = X.shape[1]
    fig, ax = plt.subplots(1,n_patterns, figsize=(18,6))
    for i in range(n_patterns) :
        ax[i].imshow(X[:,[i]],cmap='binary')
        ax[i].set_title('x{}{}'.format(i+1,name))

def show_patterns_distorted(X,Xd,name='',recovered=False) :
    n_patterns = X.shape[1]
    fig, ax = plt.subplots(1,2*n_patterns, figsize=(18,6))
    for i in range(n_patterns) :
        ax[2*i].imshow(X[:,[i]],cmap='binary')
        ax[2*i].set_title('x{}{}'.format(i+1,name))
        ax[2*i+1].imshow(Xd[:,[i]],cmap='binary')
        ax[2*i+1].set_title('x{}d'.format(i+1) + ('_rec' if recovered else ''))

def show_img_patterns(X,name='') :
    n_patterns = X.shape[0]
    dim = X.shape[1]
    sidesize = int(np.sqrt(dim))
    fig, ax = plt.subplots(1,n_patterns, figsize=(4*n_patterns,4))
    for i in range(n_patterns) :
        ax[i].imshow(X[[i],:].reshape((sidesize,sidesize)).T,cmap='binary')
        ax[i].set_title('x{}{}'.format(i+1,name))

def show_img_patterns_lin_combs(X,names) :
    n_patterns = X.shape[0]
    dim = X.shape[1]
    sidesize = int(np.sqrt(dim))
    fig, ax = plt.subplots(int(n_patterns/4)-1,4, figsize=(4*4,4*int(n_patterns/4)))
    for i in range(4) :
        for k in range(int(n_patterns/4)) :
            try : 
                ax[i,k].imshow(X[[i*4+k],:].reshape((sidesize,sidesize)).T,cmap='binary')
                ax[i,k].set_title(names[i*4+k])
            except : 
                pass

def show_img_patterns_distorted(X,Xd,name='',recovered=False) :
    n_patterns = X.shape[0]
    dim = X.shape[1]
    sidesize = int(np.sqrt(dim))
    fig, ax = plt.subplots(n_patterns,2, figsize=(4*2,4*n_patterns))
    for i in range(n_patterns) :
        ax[i,0].imshow(X[[i],:].reshape((sidesize,sidesize)).T,cmap='binary')
        ax[i,0].set_title('x{}{}'.format(i+1,name))
        ax[i,1].imshow(Xd[[i],:].reshape((sidesize,sidesize)).T,cmap='binary')
        ax[i,1].set_title('x{}d'.format(i+1) + ('_rec' if recovered else ''))
        
def show_img_patterns_gradually(Xs,grad_steps=5,name='') :
    n_patterns = len(Xs)
    dim = Xs[0][0].shape[1]
    sidesize = int(np.sqrt(dim))
    fig, ax = plt.subplots(n_patterns,grad_steps+1, figsize=(4*(grad_steps+1),4*n_patterns))
    for i in range(n_patterns) :
        n_steps = len(Xs[i])
        stepsize = int(n_steps/grad_steps)
        for k in range(grad_steps+1) :
            if k != grad_steps :
                ax[i,k].imshow(Xs[i][k*stepsize].reshape((sidesize,sidesize)).T,cmap='binary')
                ax[i,k].set_title('{}. x{}{}'.format(k*stepsize,i+1,name))
            else :
                ax[i,k].imshow(Xs[i][-1].reshape((sidesize,sidesize)).T,cmap='binary')
                ax[i,k].set_title('{}. x{}{}'.format(n_steps-1,i+1,name))

def show_img_patterns_gradually_noisy(Xs,grad_steps,name,noises) :
    n_patterns = len(Xs)
    dim = Xs[0][0].shape[1]
    sidesize = int(np.sqrt(dim))
    fig, ax = plt.subplots(n_patterns,grad_steps+1, figsize=(4*(grad_steps+1),4*n_patterns))
    for i in range(n_patterns) :
        n_steps = len(Xs[i])
        stepsize = int(n_steps/grad_steps)
        for k in range(grad_steps+1) :
            if k != grad_steps :
                ax[i,k].imshow(Xs[i][k*stepsize].reshape((sidesize,sidesize)).T,cmap='binary')
                ax[i,k].set_title('{}. x{} - {}'.format(k*stepsize,name,noises[i]))
            else :
                ax[i,k].imshow(Xs[i][-1].reshape((sidesize,sidesize)).T,cmap='binary')
                ax[i,k].set_title('{}. x{} - {}'.format(n_steps-1,name,noises[i]))

def plot_energy(Es,name='') :
    n_patterns = len(Es)
    fig, ax = plt.subplots(figsize=(10,8))
    for i in range(n_patterns) :
        ax.plot(Es[i],label='x{}{}'.format(i+1,name))
    ax.set_xlabel('Steps')
    ax.set_ylabel('Energy')
    ax.legend()

def plot_energy_noises(Es,name,noises) :
    n_patterns = len(Es)
    fig, ax = plt.subplots(figsize=(10,8))
    for i in range(n_patterns) :
        ax.plot(Es[i],label='x{} - {}'.format(name,noises[i]))
    ax.set_xlabel('Steps')
    ax.set_ylabel('Energy')
    ax.legend()

def plot_n_stables_noises(n_stables_noises,noises) :
    n_patterns = len(n_stables_noises)
    fig, ax = plt.subplots(figsize=(12,10))
    for i in range(n_patterns) :
        ax.plot(n_stables_noises[i],label='{} bit errors'.format(noises[i]))
    ax.set_xlabel('# of stored memories')
    ax.set_ylabel('Stable memories')
    ax.legend()

def linear_combs(X) :
    n_patterns = X.shape[0]
    Xcombs = []
    names = []
    # Negatives
    Xcombs.append(np.sign(-X[0,:]))
    names.append('-x1')
    Xcombs.append(np.sign(-X[1,:]))
    names.append('-x2')
    Xcombs.append(np.sign(-X[2,:]))
    names.append('-x3')
    # Combinations of 2 patterns
    Xcombs.append(np.sign(X[0,:] + X[1,:]))
    names.append('x1+x2')
    Xcombs.append(np.sign(X[0,:] + X[2,:]))
    names.append('x1+x3')
    Xcombs.append(np.sign(X[1,:] + X[2,:]))
    names.append('x2+x3')
    Xcombs.append(np.sign(X[0,:] - X[1,:]))
    names.append('x1-x2')
    Xcombs.append(np.sign(X[0,:] - X[2,:]))
    names.append('x1-x3')
    Xcombs.append(np.sign(X[1,:] - X[2,:]))
    names.append('x2-x3')
    Xcombs.append(np.sign(-X[0,:] + X[1,:]))
    names.append('-x1+x2')
    Xcombs.append(np.sign(-X[0,:] + X[2,:]))
    names.append('-x1+x3')
    Xcombs.append(np.sign(-X[1,:] + X[2,:]))
    names.append('-x2+x3')
    # Combinations of 3 patters
    Xcombs.append(np.sign(X[0,:] + X[1,:] + X[2,:]))
    names.append('x1+x2+x3')
    Xcombs.append(np.sign(X[0,:] + X[1,:] - X[2,:]))
    names.append('x1+x2-x3')
    Xcombs.append(np.sign(X[0,:] - X[1,:] - X[2,:]))
    names.append('x1-x2-x3')
    Xcombs.append(np.sign(-X[0,:] - X[1,:] - X[2,:]))
    names.append('-x1-x2-x3')
    Xcombs.append(np.sign(X[0,:] - X[1,:] + X[2,:]))
    names.append('x1-x2+x3')
    Xcombs.append(np.sign(-X[0,:] - X[1,:] + X[2,:]))
    names.append('-x1-x2+x3')
    Xcombs.append(np.sign(-X[0,:] - X[1,:] + X[2,:]))
    names.append('x1+x2-x3')
    Xcombs.append(np.sign(-X[0,:] - X[1,:] + X[2,:]))
    names.append('-x1+x2-x3')

    return np.array(Xcombs),names