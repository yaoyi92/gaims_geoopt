"""
Utility jobs for running GAIMS force-field relaxations in an active-learning setting.

Each function is wrapped with the ``@job`` decorator so that it can be scheduled
within a `jobflow.Flow`.  The typical workflow is:

1.  *evaluate_max_force* - compute the maximum atomic force after a relaxation.
2.  *extract_mol_or_structure* - obtain either the relaxed molecule or crystal
    structure from the relaxation output.
3.  *add_structure_database* - append the configuration to a running
    train/test EXTXYZ database, trimming it to a fixed size window.
4.  *get_mace_relax_job* - spawn the next MACE-based relaxation, using the
    updated potential.
"""


from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from atomate2.forcefields import MLFF
from jobflow import Flow, job, Response
import numpy as np

@job
def evaluate_max_force(forces, molecule):
    """Return the largest atomic force (eV/AA) after applying constraints.

    Parameters
    ----------
    forces : Sequence[Sequence[float]]
        Raw forces from the force-field relaxation (shape: ``(n_atoms, 3)``).
    molecule : :class:`pymatgen.core.Structure` or :class:`pymatgen.core.Molecule`
        Object used in the relaxation.  We convert it to an ASE *Atoms* object so
        we can apply its constraints.

    Returns
    -------
    float
        The maximum *magnitude* of the atomic forces after constraints have been
        projected out.
    """

    atoms = molecule.to_ase_atoms()
    forces = np.array(forces)
    for constraint in atoms.constraints:
        constraint.adjust_forces(atoms, forces)
    return np.max(np.sum(forces**2, axis=1)**0.5)

@job
def extract_mol_or_structure(mace_relax_output):
    """Extract the relaxed configuration (molecule **or** structure).

    The ``ForceFieldRelaxMaker`` returns either a ``molecule`` *or* ``structure``
    attribute depending on whether the input had lattice vectors.  This helper
    simply returns whichever is present so downstream jobs do not need to
    branch.
    """

    if mace_relax_output.molecule is not None:
        return mace_relax_output.molecule
    else:
        return mace_relax_output.structure

@job
def add_structure_database(database_dict, mol_or_struct, forces, database_size_limit = 10):
    """Append the configuration with reference data to an in-memory EXTXYZ db.

    The database is represented as a ``dict`` with two lists - ``"train.extxyz"``
    and ``"test.extxyz"`` - that mimic two on-disk XYZ files.  Both lists are
    pruned to ``database_size_limit`` entries (FIFO) to keep the total size
    bounded during active learning.

    Parameters
    ----------
    database_dict : dict[str, list]
        Running in-memory database with keys ``train.extxyz`` and ``test.extxyz``.
    mol_or_struct : Structure or Molecule
        Configuration to record.
    forces : Sequence[Sequence[float]]
        Reference forces (e.g. from first-principles) in eV/AA.
    database_size_limit : int, optional
        Maximum number of structures to retain in *each* list.

    Returns
    -------
    dict[str, list]
        The updated ``database_dict``.
    """

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
    """Create a *new* MACE relaxation job using the fine-tuned MLIP model.

    Parameters
    ----------
    mlip_output : dict
        Output of a previous MLIP training flow containing the ``mlip_path`` key
        that points to the directory where the compiled MACE model lives.
    struct : Structure or Molecule
        Atomic configuration to relax.
    max_force_criteria : float
        Target force threshold (eV/AA) to stop the relaxation.
    relax_calculator_kwargs : dict
        Extra keyword arguments forwarded to ASE's ``BFGS`` optimizer via
        ``Atomate2``.  If ``"max_steps"`` is supplied it will override the
        default value of *500*.

    Returns
    -------
    jobflow.Response
        A response that *replaces* the current job with a ``Flow`` containing the
        new relaxation, so that the parent flow continues seamlessly.
    """

    steps = 500
    if "max_steps" in relax_calculator_kwargs:
        steps = relax_calculator_kwargs["max_steps"]
        del relax_calculator_kwargs["max_steps"]
    calculator_kwargs = {'model':f'{mlip_output["mlip_path"][0]}/MACE_compiled.model'}
    calculator_kwargs.update(relax_calculator_kwargs)
    mace_maker = ForceFieldRelaxMaker(
        force_field_name = MLFF.MACE,
        relax_cell = False,
        steps=steps,
        calculator_kwargs = calculator_kwargs,
        relax_kwargs = {'fmax':max_force_criteria/10})
    job_relax = mace_maker.make(struct)
    flow = Flow([job_relax,])
    return Response(replace=flow, output=job_relax.output)
