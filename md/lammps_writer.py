# The timestep for all simulations was 1fs. The Bussi-Donadio-Parrinello thermostat [74] was employed with a
# relaxation time of 0.1ps. In order to maintain the system at zero pressure we employed an isotropic Nosé-Hoover
# chain barostat [75–77] with chain length of three and relaxation time of 1ps. The system size was chosen such
# that it contained at least 17000 atoms while maintaining the system dimensions as close to a cube as possible.

# For each crystal structure (and corresponding interatomic potential) the system was initialized with atoms in
# a perfect crystal structure with lattice parameter corresponding to the zero temperature equilibrium value.
# The system was equilibrated at the target temperature and zero pressure for 10ps followed by a period of 90ps
# during which snapshots of the atomic coordinates were collected every 10ps.

from dataclasses import dataclass
from textwrap import dedent

@dataclass
class LammpsInput:
    """ Dataclass for LAMMPS input file. """
    # === Experiment specific parameters (required) ===
    exp_name: str            # Experiment name.
    melting_point: float     # Melting point [K].

    # === Lattice and interatomic potentials (required) ===
    lattice_type: str
    lattice_parameter: float
    pair_style: str
    pair_coeff: str          # Put in the form: ${potentials_dir}/file (can stack many if necessary)
    element_name: str        # Ex: Al
    mass: float

    # === General simulation parameters (preset) ===
    dt: float = 0.001        # Timestep [ps] = 1 fs.
    t_eq: int = 10000        # Equilibration time [ts] = 10 ps.
    t: int = 90000           # Simulation time [ts] = 90 ps.
    dt_t: int = 100          # Thermo information stride [ts].
    n: int = 10              # System size.
    dt_d: int = 10000        # Dump output stride [ts].
    P: float = 0             # System pressure.
    damp_T: float = 0.1      # Thermostat damping [ps].
    damp_P: float = 1.0      # Barostat damping [ps].
    RANDOM: int = 42         # Random number generator seed.

    # === Barostat and thermostat parameters (preset) ===
    relaxation_time: float = 0.1
    barostat_relaxation_time: float = 1.0
    chain_length: int = 3

    def create_lammps_in(self, sim_temperature_fraction):
        temp = self.melting_point * sim_temperature_fraction

        final = dedent(f"""
            #---------------------------- Atomic setup ------------------------------------#
            units            metal
            timestep         {self.dt}
            boundary         p p p
            lattice          {self.lattice_type} {self.lattice_parameter}
            region           sim_box block 0 {self.n} 0 {self.n} 0 {self.n}
            create_box       1 sim_box
            create_atoms     1 box

            pair_style       {self.pair_style}                                             # interatomic potential (Lennard-Jones, EAM, etc.)
            pair_coeff       * * {self.pair_coeff} {self.element_name if "lj/cut" not in self.pair_style else ""} # interatomic potential parameters (ex. eam.alloy ${{potentials_dir}}/Cu_u3.eam)
            pair_modify      tail yes
            neigh_modify     delay 0                                                       # neighbor list update frequency (0 = every timestep)
            mass             1 {self.mass}                                                 # atomic mass
            variable         rnd equal round(random(0,999999,{self.RANDOM}))
            """)

        final += dedent(f"""
            #----------------------------- Equilibriation ---------------------------------#
            velocity         all create {temp} ${{rnd}} dist gaussian              # initial velocities
            fix              f2 all nph iso {self.P} {self.P} {self.damp_P}        # barostat
            variable         rnd equal round(random(0,999999,0))
            fix              f3 all temp/csvr {temp} {temp} {self.damp_T} ${{rnd}} # thermostat
            run              {self.t_eq}                                           # equilibriation run
            """)

        final += dedent(f"""
            #----------------------------- Run simulation ---------------------------------#
            dump             d1 all custom {self.dt_d} md/data/{self.exp_name}/dump_{sim_temperature_fraction}.gz id type x y z # snapshots
            run              {self.t}
            undump           d1
            min_modify       line forcezero
            minimize         0 0 100000 100000                                                                               # minimize energy
            write_dump       all custom md/data/{self.exp_name}/dump_{sim_temperature_fraction}_relaxed.gz id type x y z
            #------------------------------------------------------------------------------#
            """)
        return final
    
    def mini_str(self):
        s = str(self)
        for _ in range(10):
            s = s.replace("\n\n", "\n")
            s = s.replace("  ", " ")
        return "\n".join([
            line[:line.find("#")] if "#" in line else line
            for line in s.split("\n")
        ])
 
    def write(self, sim_temperature_fraction, filename):
        with open(filename, "w") as f:
            f.write(self.create_lammps_in(sim_temperature_fraction))