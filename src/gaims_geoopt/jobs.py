from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from atomate2.forcefields import MLFF
from jobflow import Flow, job, Response
import numpy as np

@job
def evaluate_max_force(forces):
    forces = np.array(forces)
    return np.max(np.sum(forces**2, axis=1)**0.5)

@job
def add_structure_database(database_dict, mol_or_struct):
    mol_or_struct_copy = mol_or_struct.copy()
    mol_or_struct_copy.properties["REF_energy"] = mol_or_struct.properties["energy"]
    mol_or_struct_copy.properties["REF_virial"] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for i in range(len(mol_or_struct)):
        mol_or_struct_copy.sites[i].properties["REF_forces"] = mol_or_struct.properties["forces"][i]
    database_dict["train.extxyz"].append(mol_or_struct_copy)
    database_dict["test.extxyz"].append(mol_or_struct_copy)
    return database_dict

@job
def get_mace_relax_job(mlip_output, struct):
    mace_maker = ForceFieldRelaxMaker(
        force_field_name = MLFF.MACE,
        calculator_kwargs = {'model':f'{mlip_output["mlip_path"][0]}/MACE_compiled.model'},
        relax_kwargs = {'fmax':0.025})
    job_relax = mace_maker.make(struct)
    flow = Flow([job_relax,])
    return Response(replace=flow, output=job_relax.output)