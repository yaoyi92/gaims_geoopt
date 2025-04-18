from pymatgen.core import Molecule
from jobflow import run_locally, job
from pymatgen.io.ase import AseAtomsAdaptor
import ase
from tblite.ase import TBLite
import numpy as np
from gaims_geoopt.flows import MLIPAssistedGeoOptMaker
from ase.calculators.singlepoint import SinglePointCalculator
from pathlib import Path

molecule = Molecule.from_str(
"""3
Properties=species:S:1:pos:R:3 pbc="F F F"
O 0.0 0.0 0.0
H 1.0 0.0 0.0
H 0.0 1.0 0.0
""",
fmt="xyz",
)

list_training = []
list_valid = []

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




fl = MLIPAssistedGeoOptMaker().make(molecule, database_dict, 0.02, database_size_limit=5, 
        machine_learning_fit_kwargs = {"max_num_epochs": 300}, relax_calculator_kwargs={"max_steps":300},
        calculator = "aims", calculator_kwargs=parameters)
response = run_locally(fl, create_folders=True)

energies = []
max_forces = []
structures = []
flow_now = fl
while True:
    if flow_now is None:
        break
    for job in flow_now:
        if job.name=="SCF Calculation":
            energies.append(response[job.uuid][1].output.output.energy)
            structure_tmp = response[job.uuid][1].output.output.structure.as_dict()
            for site in structure_tmp['sites']:
                if isinstance(site['properties']['force'], np.ndarray):
                    site['properties']['force'] = site['properties']['force'].tolist()
            structures.append(structure_tmp)
        if job.name=="evaluate_max_force":
            max_forces.append(response[job.uuid][1].output)
        if job.name=="get_mace_relax_job":
            relax_job_uuid = response[job.uuid][1].replace[0].uuid
            #print(response[relax_job_uuid][1].output.output.n_steps, end = " ")
    for job in flow_now:
        if job.name=="check_convergence_and_next":
            uuid_next = job.uuid
    flow_now = response[uuid_next][1].replace
data={"energies": energies, "max_forces": max_forces, "structures": structures}
with open('gaims_geoopt_result.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

