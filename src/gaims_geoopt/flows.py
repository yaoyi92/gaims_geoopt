from dataclasses import dataclass
from atomate2.ase.jobs import GFNxTBStaticMaker
from jobflow import Flow, job, Response, Maker
from autoplex.fitting.common.jobs import machine_learning_fit
import logging
from gaims_geoopt.jobs import evaluate_max_force, add_structure_database, get_mace_relax_job, 
from atomate2.aims.jobs.core import StaticMaker as AimsStaticMaker
from pymatgen.io.aims.sets.core import StaticSetGenerator
from pymatgen.core import Structure, Molecule

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

@job 
def check_convergence_and_next(struct, database_dict, last_dir, max_force, max_force_criteria, n_gaims_geoopt_steps, max_gaims_geoopt_steps, database_size_limit, n_mlip_relax_steps, machine_learning_fit_kwargs, relax_calculator_kwargs, calculator, calculator_kwargs):
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
    if last_dir is None:
        #foundation_model = "small"
        if "foundation_model" not in machine_learning_fit_kwargs:
            machine_learning_fit_kwargs["foundation_model"] = "small"
    else:
        #foundation_model = last_dir[0] + "/MACE.model"
        machine_learning_fit_kwargs["foundation_model"] = last_dir[0] + "/MACE.model"
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
    job_macefit = machine_learning_fit(**machine_learning_fit_kwargs_default)
    job_relax = get_mace_relax_job(job_macefit.output, struct, max_force_criteria, relax_calculator_kwargs)
    #job_static = GFNxTBStaticMaker(
    #    calculator_kwargs={"method": "GFN2-xTB"},
    #).make(job_relax.output.output.molecule)
    #job_max_force = evaluate_max_force(job_static.output.output.forces)
    if calculator == "GFN2-xTB":
        job_static = GFNxTBStaticMaker(
            calculator_kwargs={"method": "GFN2-xTB"},
        ).make(job_relax.output.output.molecule)
        job_max_force = evaluate_max_force(job_static.output.output.forces)
        job_add_database = add_structure_database(database_dict, job_static.output.output.mol_or_struct, job_static.output.output.forces, database_size_limit)
        job_check_convergence_and_next = check_convergence_and_next(job_static.output.output.mol_or_struct,
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
        job_mol_or_structure = extract_mol_or_structure(job_relax.output.output)
        job_static = AimsStaticMaker(
            input_set_generator=StaticSetGenerator(user_params=calculator_kwargs)
        ).make(job_mol_or_structure.output)
        job_max_force = evaluate_max_force(job_static.output.output.forces)
        job_add_database = add_structure_database(database_dict, job_static.output.output.structure, job_static.output.output.forces, database_size_limit)
        job_check_convergence_and_next = check_convergence_and_next(job_static.output.output.structure,
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

@dataclass    
class MLIPAssistedGeoOptMaker(Maker):
    name: str = "MLIP assisted GeoOpt"
    def make(self, molecule, database_dict, max_force_criteria, max_gaims_geoopt_steps = 30, database_size_limit = 10, machine_learning_fit_kwargs={}, relax_calculator_kwargs={}, calculator = "GFN2-xTB", calculator_kwargs = {}):
        if calculator == "GFN2-xTB":
            if isinstance(molecule, Structure):
                logging.info(
                    f"Requesting a GFN2-xTB for periodic system which is not supported."
                )
                return None
            job_static = GFNxTBStaticMaker(
                calculator_kwargs={"method": "GFN2-xTB"},
            ).make(molecule)
            job_max_force = evaluate_max_force(job_static.output.output.forces)
            job_add_database = add_structure_database(database_dict, job_static.output.output.mol_or_struct, job_static.output.output.forces, database_size_limit)
            job_check_convergence_and_next = check_convergence_and_next(job_static.output.output.mol_or_struct,
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
            job_max_force = evaluate_max_force(job_static.output.output.forces)
            job_add_database = add_structure_database(database_dict, job_static.output.output.structure, job_static.output.output.forces, database_size_limit)
            job_check_convergence_and_next = check_convergence_and_next(job_static.output.output.structure,
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
        jobs = [job_static, job_max_force, job_add_database, job_check_convergence_and_next]
        return Flow(jobs)
