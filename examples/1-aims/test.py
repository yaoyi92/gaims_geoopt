from pymatgen.core import Molecule
from jobflow import run_locally, job
from pymatgen.io.ase import AseAtomsAdaptor
import ase
import numpy as np
import logging
from gaims_geoopt.flows import MLIPAssistedGeoOptMaker
from ase.calculators.singlepoint import SinglePointCalculator
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)



molecule = Molecule.from_str(
"""3
Properties=species:S:1:pos:R:3 pbc="F F F"
O 0.0 0.0 0.0
H 1.0 0.0 0.0
H 0.0 1.0 0.0
""",
fmt="xyz",
)
atoms = molecule.to_ase_atoms()
atomic_energies = {"O": -2043.567004718,  "H": -13.598030178}

list_training = []
list_valid = []

elements = set(atoms.get_chemical_symbols())
for element in elements:
    atoms_freeatom = ase.Atoms(element)
    atoms_freeatom.calc = SinglePointCalculator(atoms = atoms_freeatom, energy=atomic_energies[element])
    atoms_freeatom.info['REF_energy'] = atoms_freeatom.get_potential_energy()
    atoms_freeatom.arrays['REF_forces'] = np.array([[0.0, 0.0, 0.0]])
    atoms_freeatom.info['REF_virial'] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    atoms_freeatom.info['config_type'] = "IsolatedAtom"
    atoms_freeatom.calc = None
    list_training.append(atoms_freeatom)

adapter = AseAtomsAdaptor()
database_dict = {
    "train.extxyz": [
        adapter.get_structure(atoms_tmp, Molecule)
        for atoms_tmp in list_training
    ],
    "test.extxyz": [
        adapter.get_structure(atoms_tmp, Molecule)
        for atoms_tmp in list_valid
    ],
}

species_dir=Path("/home/yiy/Test/fhi-aims/FHIaims/species_defaults/defaults_2020")

parameters = {
    "species_dir": (species_dir / "light").as_posix(),
    "compute_forces": True,
}

fl = MLIPAssistedGeoOptMaker().make(molecule, database_dict, 0.02, 
                                    calculator = "aims", calculator_kwargs=parameters)
response = run_locally(fl, create_folders=True)

flow_now = fl
while True:
    if flow_now is None:
        break
    for job in flow_now:
        if job.name=="evaluate_max_force":
            print(response[job.uuid][1].output)
        if job.name=="get_mace_relax_job":
            relax_job_uuid = response[job.uuid][1].replace[0].uuid
            print(response[relax_job_uuid][1].output.output.n_steps, end = " ")
    for job in flow_now:
        if job.name=="check_convergence_and_next":
            uuid_next = job.uuid
    flow_now = response[uuid_next][1].replace
    #print(flow_now)
