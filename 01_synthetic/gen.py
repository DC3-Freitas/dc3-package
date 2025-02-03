import numpy as np
import ovito

class Generator:
    def __init__(self):
        pass
        
    def load_np(lattice: np.ndarray):
        '''
        Initialize a generator with a given lattice; use OVITO for analysis
        - lattice: expects a numpy array (n x 3) of perfect lattice positions
        ''' 
        self.lattice = lattice
        # calculate nearest neighbor distance
        self.nn_distance = np.linalg.norm(lattice[0] - lattice[1])


    def load_ovito(lattice: ovito.ParticleProperty):
        '''
        Initialize a generator with a given lattice; use OVITO for analysis
        - lattice: expects a ParticleProperty object of perfect lattice positions
        '''
        self.lattice = lattice
        # calculate nearest neighbor distance
        self.nn_distance = np.linalg.norm(lattice[0] - lattice[1])
    
    def load_ovito_file(filename: str):
        '''
        Initialize a generator with a given lattice; use OVITO for analysis
        - lattice: expects a string of the path to a LAMMPS data file
        '''
        self.lattice = ovito.io.import_file(lattice).particles.positions
        
    
    def generate(self, alpha):
        '''
        Generate a synthetic sample of initialized lattice with a given thermal alpha.
        - alpha: percentage of nearest neighbor distance to displace atoms
        '''
        displaced_lattice = self.lattice.clone()
        displacement_radius = alpha * self.nn_distance
        n_atoms = self.lattice.particles.count

        # generate random displacements
        # https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
        phi = np.random.uniform(0, 2*np.pi, n_atoms)
        theta = np.random.uniform(0, np.pi, n_atoms)
        r = np.cbrt(np.random.uniform(0, displacement_radius, n_atoms))

        # apply displacements
        displaced_lattice.particles_.positions_[:] += np.array([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ]).T

    def generate_range(self, alpha_min, alpha_max, n):
        '''
        Generate a range of synthetic samples of initialized lattice with thermal alphas
        ranging from alpha_min to alpha_max.
        '''
        alphas = np.linspace(alpha_min, alpha_max, n)
        return [
            self.generate(alpha)
            for alpha in alphas
        ]

    def save(self, path):
        '''
        Save the generated samples to a LAMMPS data file.
        '''
        ovito.io.export_file(
            self.lattice,
            path,
            columns = ['Particle Identifier', 'Particle Type', 'Position.X', 'Position.Y', 'Position.Z']
        )