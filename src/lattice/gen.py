import numpy as np
import ovito


class LatticeGenerator:
    def load_np(self, lattice: np.ndarray):
        """
        Initialize a generator with a given lattice; use OVITO for analysis

        Args:
            lattice: a numpy array (n x 3) of perfect lattice positions
        """
        self.lattice = ovito.data.DataCollection()
        particles = ovito.data.Particles()
        particles.create_property("Position", data=lattice)
        self.lattice.objects.append(particles)
        cell = ovito.data.SimulationCell(pbc=(False, False, False))
        cell[...] = [[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0]]
        self.lattice.objects.append(cell)
        self.calculate_nn_distance()

    def load_ovito(self, lattice: ovito.data.DataCollection):
        """
        Initialize a generator with a given lattice; use OVITO for analysis

        Args:
            lattice: an OVITO DataCollection object of perfect lattice positions
        """
        self.lattice = lattice
        self.calculate_nn_distance()

    def load_lammps(self, filename: str):
        """
        Initialize a generator with a given lattice; use OVITO for analysis

        Args:
            filename: a string of the path to a LAMMPS data file
        """
        pipeline = ovito.io.import_file(filename)
        self.lattice = pipeline.compute()
        self.calculate_nn_distance()

    def calculate_nn_distance(self):
        """
        Calculate the nearest neighbor distance of the initialized lattice using OVITO.
        Loads from internal lattice. NearestNeighborFinder is a generator which results
        in mildly strange syntax in this use case
        """
        self.nn_distance = next(
            ovito.data.NearestNeighborFinder(1, self.lattice).find(0)
        ).distance
        print(f"Nearest neighbor distance: {self.nn_distance}")

    def generate(self, alpha):
        """
        Generate a synthetic sample of initialized lattice with a given thermal alpha.

        Args:
            alpha: percentage of nearest neighbor distance to displace atoms

        Returns:
            displaced_lattice: an OVITO DataCollection object of the displaced lattice
        """
        displaced_lattice = self.lattice.clone()
        displacement_radius = alpha * self.nn_distance
        n_atoms = self.lattice.particles.count

        # generate random displacements
        # https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
        # we could potentially speed this up by sampling from a cube and rejecting points outside the sphere
        phi = np.random.uniform(0, 2 * np.pi, n_atoms)
        # theta = np.random.uniform(0, np.pi, n_atoms)
        theta = np.arccos(np.random.uniform(-1, 1, n_atoms))
        r = displacement_radius * np.cbrt(np.random.uniform(0, 1, n_atoms))

        # apply displacements
        displaced_lattice.particles_.positions_[:] += np.array(
            [
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta),
            ]
        ).T

        return displaced_lattice

    def generate_range(self, alpha_min, alpha_max, n):
        """
        Generate a range of synthetic samples of initialized lattice with thermal alphas
        ranging from alpha_min to alpha_max.

        Args:
            alpha_min: minimum thermal alpha
            alpha_max: maximum thermal alpha
            n: number of samples to generate (linear interpolation between alpha_min and alpha_max)

        Returns:
            list of OVITO DataCollection objects of the displaced lattices
        """
        alphas = np.linspace(alpha_min, alpha_max, n)
        return [self.generate(alpha) for alpha in alphas]

    def save(self, path, displaced_lattice):
        """
        Save the generated samples to a LAMMPS data file. (Probably shouldn't be used in practice)
        """
        ovito.io.export_file(
            displaced_lattice,
            path,
            columns=[
                "Particle Identifier",
                "Particle Type",
                "Position.X",
                "Position.Y",
                "Position.Z",
            ],
        )
