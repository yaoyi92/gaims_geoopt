from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from atomate2.forcefields import MLFF
from jobflow import Flow, job, Response
import numpy as np

@job
def evaluate_max_force(forces):
    forces = np.array(forces)
    return np.max(np.sum(forces**2, axis=1)**0.5)

@job
def extract_mol_or_structure(mace_relax_output):
    if mace_relax_output.molecule is not None:
        return mace_relax_output.molecule
    else:
        return mace_relax_output.structure

@job
def add_structure_database(database_dict, mol_or_struct, forces, database_size_limit = 10):
    mol_or_struct_copy = mol_or_struct.copy()
    mol_or_struct_copy.properties["REF_energy"] = mol_or_struct.properties["energy"]
    mol_or_struct_copy.properties["REF_virial"] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for i in range(len(mol_or_struct)):
        mol_or_struct_copy.sites[i].properties["REF_forces"] = forces[i]
    database_dict["train.extxyz"].append(mol_or_struct_copy)
    database_dict["test.extxyz"].append(mol_or_struct_copy)
    while len(database_dict["train.extxyz"]) > database_size_limit:
        database_dict["train.extxyz"].pop(0)
    while len(database_dict["test.extxyz"]) > database_size_limit:
        database_dict["test.extxyz"].pop(0)
    return database_dict

@job
def get_mace_relax_job(mlip_output, struct, max_force_criteria, relax_calculator_kwargs):
    steps = 500
    if "max_steps" in relax_calculator_kwargs:
        steps = relax_calculator_kwargs["max_steps"]
        del relax_calculator_kwargs["max_steps"]
    calculator_kwargs = {'model':f'{mlip_output["mlip_path"][0]}/MACE_compiled.model'}
    calculator_kwargs.update(relax_calculator_kwargs)
    mace_maker = ForceFieldRelaxMaker(
        force_field_name = MLFF.MACE,
        steps=steps,
        calculator_kwargs = calculator_kwargs,
        relax_kwargs = {'fmax':max_force_criteria/10})
    job_relax = mace_maker.make(struct)
    flow = Flow([job_relax,])
    return Response(replace=flow, output=job_relax.output)
