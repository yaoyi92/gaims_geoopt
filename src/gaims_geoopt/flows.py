from dataclasses import dataclass
from atomate2.ase.jobs import GFNxTBStaticMaker
from jobflow import Flow, job, Response, Maker
from autoplex.fitting.common.jobs import machine_learning_fit
import logging
from gaims_geoopt.jobs import evaluate_max_force, add_structure_database, get_mace_relax_job
from atomate2.aims.jobs.core import StaticMaker as AimsStaticMaker
from pymatgen.io.aims.sets.core import StaticSetGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

@job 
def check_convergence_and_next(struct, database_dict, max_force, max_force_criteria, calculator, calculator_kwargs):
    if max_force < max_force_criteria:
        logging.info(
            f"MLIP assisted Geometry Optimization Converged with max_force: {max_force} < {max_force_criteria}"
        )

        return None
    logging.info(
        f"MLIP assisted Geometry Optimization continues with max_force: {max_force} > {max_force_criteria}"
    )
    job_macefit = machine_learning_fit(
                      database_dir=None,
                      database_dict=database_dict,
                      run_fits_on_different_cluster=True,
                      name="MACE",
                      mlip_type="MACE",
                      ref_energy_name="REF_energy",
                      ref_force_name="REF_forces",
                      ref_virial_name=None,
                      species_list=None,
                      num_processes_fit=1,
                      foundation_model="small",
                      multiheads_finetuning=False,
                      loss="ef",
                      energy_weight = 1.0,
                      forces_weight = 1.0,
                      stress_weight = 0.0,
                      E0s = "isolated",
                      scaling = "rms_forces_scaling",
                      batch_size = 2,
                      max_num_epochs = 1000,
                      ema=True,
                      ema_decay = 0.99,
                      amsgrad=True,
                      default_dtype = "float64",
                      keep_isolated_atoms=True,
                      lr = 0.01,
                      patience = 500,
                      device = "cpu",
                      save_cpu =True,
                      seed = 3,
                  )
    job_relax = get_mace_relax_job(job_macefit.output, struct, max_force_criteria)
    #job_static = GFNxTBStaticMaker(
    #    calculator_kwargs={"method": "GFN2-xTB"},
    #).make(job_relax.output.output.molecule)
    #job_max_force = evaluate_max_force(job_static.output.output.forces)
    if calculator == "GFN2-xTB":
        job_static = GFNxTBStaticMaker(
            calculator_kwargs={"method": "GFN2-xTB"},
        ).make(job_relax.output.output.molecule)
        job_max_force = evaluate_max_force(job_static.output.output.forces)
        job_add_database = add_structure_database(database_dict, job_static.output.output.mol_or_struct, job_static.output.output.forces)
        job_check_convergence_and_next = check_convergence_and_next(job_static.output.output.mol_or_struct,
                                                                    job_add_database.output,
                                                                    job_max_force.output,
                                                                    max_force_criteria,
                                                                    calculator,
                                                                    calculator_kwargs,
                                                                    )
    elif calculator == "aims":
        job_static = AimsStaticMaker(
            input_set_generator=StaticSetGenerator(user_params=calculator_kwargs)
        ).make(job_relax.output.output.molecule)
        job_max_force = evaluate_max_force(job_static.output.output.forces)
        job_add_database = add_structure_database(database_dict, job_static.output.output.structure, job_static.output.output.forces)
        job_check_convergence_and_next = check_convergence_and_next(job_static.output.output.structure,
                                                                    job_add_database.output,
                                                                    job_max_force.output,
                                                                    max_force_criteria, 
                                                                    calculator,
                                                                    calculator_kwargs
                                                                    )
    flow = Flow([job_macefit, job_relax, job_static, job_max_force, job_add_database, job_check_convergence_and_next])
    return Response(replace=flow)

@dataclass    
class MLIPAssistedGeoOptMaker(Maker):
    name: str = "MLIP assisted GeoOpt"
    def make(self, molecule, database_dict, max_force_criteria, calculator = "GFN2-xTB", calculator_kwargs = {}):
        if calculator == "GFN2-xTB":
            job_static = GFNxTBStaticMaker(
                calculator_kwargs={"method": "GFN2-xTB"},
            ).make(molecule)
            job_max_force = evaluate_max_force(job_static.output.output.forces)
            job_add_database = add_structure_database(database_dict, job_static.output.output.mol_or_struct, job_static.output.output.forces)
            job_check_convergence_and_next = check_convergence_and_next(job_static.output.output.mol_or_struct,
                                                                        job_add_database.output,
                                                                        job_max_force.output,
                                                                        max_force_criteria, 
                                                                        calculator,
                                                                        calculator_kwargs
                                                                        )
        elif calculator == "aims":
            job_static = AimsStaticMaker(
                input_set_generator=StaticSetGenerator(user_params=calculator_kwargs)
            ).make(molecule)
            job_max_force = evaluate_max_force(job_static.output.output.forces)
            job_add_database = add_structure_database(database_dict, job_static.output.output.structure, job_static.output.output.forces)
            job_check_convergence_and_next = check_convergence_and_next(job_static.output.output.structure,
                                                                        job_add_database.output,
                                                                        job_max_force.output,
                                                                        max_force_criteria, 
                                                                        calculator,
                                                                        calculator_kwargs
                                                                        )
        jobs = [job_static, job_max_force, job_add_database, job_check_convergence_and_next]
        return Flow(jobs)