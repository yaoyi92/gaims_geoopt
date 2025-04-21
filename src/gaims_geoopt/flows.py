"""MLIP-assisted geometry optimisation workflow
------------------------------------------------
This module defines two core components, both designed to be used within the
``jobflow`` ecosystem:

* ``check_convergence_and_next`` - a *recursive* job that decides whether the
  active-learning loop should continue.  If more data / optimisation steps are
  needed it:

  1. Fine-tune a MACE interatomic potential on the current
     in-memory database via ``machine_learning_fit``.
  2. Runs a MACE-based relaxation using that newly trained potential.
  3. Computes reference forces with a chosen *ab-initio* calculator (GFN2-xTB or
     FHI-aims) to evaluate error / convergence, updates the database, and calls
     itself again.

* ``MLIPAssistedGeoOptMaker`` - a convenience *Maker* that kicks off the first
  reference energy/force calculation, seeds the EXTXYZ database, and launches
  the recursive convergence job.

The overall logic can be visualised as:

.. code:: text

    ┌─> static reference calc (xTB / FHI-aims) ─┐
    │                                           ↓
    │  update in-memory EXTXYZ database         │
    │                                           ↓
    │  ML potential fit (MACE)                  │
    │                                           ↓
    │  ML relaxation (ASE + MACE)               │
    │                                           ↓
    └── check convergence & recurse ────────────┘

The loop stops when either *max_force* < *max_force_criteria*, the MLIP
relaxation is stuck (no movement in two consecutive steps), or the maximum
number of GAIMS geometry optimisation steps is reached.
"""



from dataclasses import dataclass
from atomate2.ase.jobs import GFNxTBStaticMaker
from jobflow import Flow, job, Response, Maker
from autoplex.fitting.common.jobs import machine_learning_fit
import logging
from gaims_geoopt.jobs import evaluate_max_force, add_structure_database, get_mace_relax_job, extract_mol_or_structure
from atomate2.aims.jobs.core import StaticMaker as AimsStaticMaker
from pymatgen.io.aims.sets.core import StaticSetGenerator
from pymatgen.core import Structure, Molecule

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------------------------------------------------------
#  Recursive convergence / continuation job
# -----------------------------------------------------------------------------

@job 
def check_convergence_and_next(struct, database_dict, last_dir, max_force, max_force_criteria, n_gaims_geoopt_steps, max_gaims_geoopt_steps, database_size_limit, n_mlip_relax_steps, machine_learning_fit_kwargs, relax_calculator_kwargs, calculator, calculator_kwargs):
    """Decide whether to *stop* or *continue* the active-learning geo-opt loop.

    Parameters
    ----------
    struct
        Current atomic configuration (output of the last MLIP relaxation).
    database_dict
        Rolling EXTXYZ database holding training and test configurations.
    last_dir
        Path to the directory containing the previous MACE model.  ``None``
        indicates that this is the *first* iteration.
    max_force
        Maximum atomic force (eV/AA) from the last reference calculation.
    max_force_criteria
        Target convergence threshold (eV/AA).
    n_gaims_geoopt_steps, max_gaims_geoopt_steps
        Current and maximum allowed GAIMS geometry optimisation steps.
    database_size_limit
        Soft limit on the number of structures kept in each EXTXYZ split.
    n_mlip_relax_steps
        Number of *ASE* optimisation steps taken in the last MLIP relaxation.
        A value of ``2`` with *no force improvement* is used as a heuristic for
        a *stuck* optimisation.
    machine_learning_fit_kwargs, relax_calculator_kwargs
        Keyword overrides passed on to downstream jobs.
    calculator, calculator_kwargs
        Choice of reference calculator (``"GFN2-xTB"`` or ``"aims"``) and its
        specific keyword arguments.
    """

    # ------------------------------------------------------------------
    # 1. Check termination criteria
    # ------------------------------------------------------------------

    if n_gaims_geoopt_steps >= max_gaims_geoopt_steps:
        logging.info(
            f"MLIP assisted Geometry Optimization stopped reach maximum Geoopt steps, with max_force: {max_force} > {max_force_criteria}, ML assisted steps: {n_mlip_relax_steps}, Geoopt steps: {n_gaims_geoopt_steps} "
        )
        return None
    if max_force < max_force_criteria or n_mlip_relax_steps == 2:
        if max_force < max_force_criteria:
            logging.info(
                    f"MLIP assisted Geometry Optimization Converged with max_force: {max_force} < {max_force_criteria}, ML assisted relax steps: {n_mlip_relax_steps}, Geoopt steps: {n_gaims_geoopt_steps}"
            )
        elif n_mlip_relax_steps == 2:
            logging.info(
                f"MLIP assisted Geometry Optimization stuck with ML relax not moving."
            )

        return None
    logging.info(
        f"MLIP assisted Geometry Optimization continues with max_force: {max_force} > {max_force_criteria}, ML assisted steps: {n_mlip_relax_steps}, Geoopt steps: {n_gaims_geoopt_steps} "
    )

    # ------------------------------------------------------------------
    # 2. Prepare kwargs for the next MACE fit
    # ------------------------------------------------------------------

    if last_dir is None:
        # First iteration – choose a small foundation model unless overridden.
        if "foundation_model" not in machine_learning_fit_kwargs:
            machine_learning_fit_kwargs["foundation_model"] = "small"
    else:
        # Warm‑start from the previous model.
        machine_learning_fit_kwargs["foundation_model"] = last_dir[0] + "/MACE.model"

    # Default hyper‑parameters for the *machine_learning_fit* helper.
    machine_learning_fit_kwargs_default = {
        "database_dir":None,
        "database_dict":database_dict,
        "run_fits_on_different_cluster":True,
        "name":"MACE",
        "mlip_type":"MACE",
        "ref_energy_name":"REF_energy",
        "ref_force_name":"REF_forces",
        "ref_virial_name":None,
        "species_list":None,
        "num_processes_fit":1,
        #"foundation_model":foundation_model,
        "multiheads_finetuning":False,
        "loss":"forces_only",
        "energy_weight" : 0.0,
        "forces_weight" : 1.0,
        "stress_weight" : 0.0,
        "E0s" : "average",
        "scaling" : "rms_forces_scaling",
        "batch_size" : 1,
        "max_num_epochs" : 500,
        "ema":True,
        "ema_decay" : 0.99,
        "swa":False,
        "start_swa":3000,
        "amsgrad":True,
        "default_dtype" : "float64",
        "keep_isolated_atoms":False,
        "lr" : 0.001,
        "patience" : 500,
        "device" : "cpu",
        "save_cpu" :True,
        "seed" : 3,
    }

    machine_learning_fit_kwargs_default.update(machine_learning_fit_kwargs)

    # ------------------------------------------------------------------
    # 3. Launch downstream jobs
    # ------------------------------------------------------------------
    # 3a. Fit / fine‑tune the MACE potential.
    job_macefit = machine_learning_fit(**machine_learning_fit_kwargs_default)

    # 3b. Use the fitted model for a force‑field relaxation.
    job_relax = get_mace_relax_job(job_macefit.output, struct, max_force_criteria, relax_calculator_kwargs)

    # 3c. High‑accuracy *reference* calculation and DB update depend on the
    #     chosen calculator.
    if calculator == "GFN2-xTB":
        # ---------------------
        # * Semi‑empirical GFN2‑xTB reference (molecules only)
        # ---------------------

        job_static = GFNxTBStaticMaker(
            calculator_kwargs={"method": "GFN2-xTB"},
        ).make(job_relax.output.output.molecule)
        job_max_force = evaluate_max_force(job_static.output.output.forces, job_relax.output.output.molecule )
        job_add_database = add_structure_database(database_dict, job_static.output.output.mol_or_struct, job_static.output.output.forces, database_size_limit)
        job_check_convergence_and_next = check_convergence_and_next(job_relax.output.output.molecule,
                                                                    job_add_database.output,
                                                                    job_macefit.output.mlip_path,
                                                                    job_max_force.output,
                                                                    max_force_criteria,
                                                                    n_gaims_geoopt_steps+1,
                                                                    max_gaims_geoopt_steps,
                                                                    database_size_limit,
                                                                    job_relax.output.output.n_steps,
                                                                    machine_learning_fit_kwargs,
                                                                    relax_calculator_kwargs,
                                                                    calculator,
                                                                    calculator_kwargs,
                                                                    )
        flow = Flow([job_macefit, job_relax, job_static, job_max_force, job_add_database, job_check_convergence_and_next])
    elif calculator == "aims":
        # ---------------------
        # * FHI‑aims reference calculation (molecule or periodic structure)
        # ---------------------

        job_mol_or_structure = extract_mol_or_structure(job_relax.output.output)
        job_static = AimsStaticMaker(
            input_set_generator=StaticSetGenerator(user_params=calculator_kwargs)
        ).make(job_mol_or_structure.output)
        job_max_force = evaluate_max_force(job_static.output.output.forces, job_mol_or_structure.output)
        job_add_database = add_structure_database(database_dict, job_static.output.output.structure, job_static.output.output.forces, database_size_limit)
        job_check_convergence_and_next = check_convergence_and_next(job_mol_or_structure.output,
                                                                    job_add_database.output,
                                                                    job_macefit.output.mlip_path,
                                                                    job_max_force.output,
                                                                    max_force_criteria, 
                                                                    n_gaims_geoopt_steps+1,
                                                                    max_gaims_geoopt_steps,
                                                                    database_size_limit,
                                                                    job_relax.output.output.n_steps,
                                                                    machine_learning_fit_kwargs,
                                                                    relax_calculator_kwargs,
                                                                    calculator,
                                                                    calculator_kwargs
                                                                    )
        flow = Flow([job_macefit, job_relax, job_mol_or_structure, job_static, job_max_force, job_add_database, job_check_convergence_and_next])
    return Response(replace=flow)


# -----------------------------------------------------------------------------
#  Maker that seeds the workflow
# -----------------------------------------------------------------------------

@dataclass    
class MLIPAssistedGeoOptMaker(Maker):
    """Launch a MLIP-assisted geometry optimisation from a *single* structure."""

    name: str = "MLIP assisted GeoOpt"

    def make(self, molecule, database_dict, max_force_criteria, max_gaims_geoopt_steps = 30, database_size_limit = 10, machine_learning_fit_kwargs={}, relax_calculator_kwargs={}, calculator = "GFN2-xTB", calculator_kwargs = {}):
        """Kick-off the optimisation by running the *first* reference calculation."""

        # ------------------------------------------------------------------
        # 1. Initial reference calculation and DB seeding
        # ------------------------------------------------------------------

        if calculator == "GFN2-xTB":
            # xTB only supports *molecules*, warn otherwise.

            if isinstance(molecule, Structure):
                logging.info(
                    f"Requesting a GFN2-xTB for periodic system which is not supported."
                )
                return None
            job_static = GFNxTBStaticMaker(
                calculator_kwargs={"method": "GFN2-xTB"},
            ).make(molecule)
            job_max_force = evaluate_max_force(job_static.output.output.forces, molecule)
            job_add_database = add_structure_database(database_dict, job_static.output.output.mol_or_struct, job_static.output.output.forces, database_size_limit)
            job_check_convergence_and_next = check_convergence_and_next(molecule,
                                                                        job_add_database.output,
                                                                        None,
                                                                        job_max_force.output,
                                                                        max_force_criteria, 
                                                                        0,
                                                                        max_gaims_geoopt_steps,
                                                                        database_size_limit,
                                                                        -1,
                                                                        machine_learning_fit_kwargs,
                                                                        relax_calculator_kwargs,
                                                                        calculator,
                                                                        calculator_kwargs
                                                                        )
        elif calculator == "aims":
            job_static = AimsStaticMaker(
                input_set_generator=StaticSetGenerator(user_params=calculator_kwargs)
            ).make(molecule)
            job_max_force = evaluate_max_force(job_static.output.output.forces, molecule)
            job_add_database = add_structure_database(database_dict, job_static.output.output.structure, job_static.output.output.forces, database_size_limit)
            job_check_convergence_and_next = check_convergence_and_next(molecule,
                                                                        job_add_database.output,
                                                                        None,
                                                                        job_max_force.output,
                                                                        max_force_criteria, 
                                                                        0,
                                                                        max_gaims_geoopt_steps,
                                                                        database_size_limit,
                                                                        -1,
                                                                        machine_learning_fit_kwargs,
                                                                        relax_calculator_kwargs,
                                                                        calculator,
                                                                        calculator_kwargs
                                                                        )
        # ------------------------------------------------------------------
        # 2. Assemble seed flow
        # ------------------------------------------------------------------

        jobs = [job_static, job_max_force, job_add_database, job_check_convergence_and_next]
        return Flow(jobs)
