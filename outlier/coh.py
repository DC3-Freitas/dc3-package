import numpy as np
from ovito.data import NearestNeighborFinder
from scipy.special import sph_harm

def coherence(data, l, N_neigh):
    N_b = NearestNeighborFinder(N_neigh,data)
    num_atoms = data.particles.count

    coh_fac = np.zeros(num_atoms)

    E = np.zeros(num_atoms, sum(l+l+1)
    
    for atom in range(num_atoms):
        for neighbor in N_b.find(atom):
        
            #angles in 3D space to find vector using spherical harmonics

            phi = arctan2(neighbor.delta[1]/neighbor.delta[0])
            theta = arccos(neighbor.delta[2]/neighbor.distance)

            value = 0

            for i in range(len(l)):
                E[atom, value: value+2*l[i]+1] += sph_harm(range(-l[i],l[i]+1),l[i],phi,theta)

                value += 2*l[i] + 1
            
        E[atom] /= N_neigh

    #dot product between two atoms

    for atom in range(num_atoms):
        for neighbor in N_b.find.(atom):
            E_i = E[atom]/norm(E[atom])
            E_j = E[neighbor.index]/norm(E[neighbor.index])

            coh_fac[atom] += dot(E_i,E_j)

        coh_fac[atom] /= N_neigh

    return coh_fac
