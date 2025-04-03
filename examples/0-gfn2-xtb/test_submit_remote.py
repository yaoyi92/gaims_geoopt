from pymatgen.core import Molecule
from jobflow import run_locally, job
from pymatgen.io.ase import AseAtomsAdaptor
import ase
from tblite.ase import TBLite
import numpy as np
from gaims_geoopt.flows import MLIPAssistedGeoOptMaker
from jobflow_remote import submit_flow, set_run_config

molecule = Molecule.from_str(
"""24
Properties=species:S:1:pos:R:3 pbc="F F F"
C        0.45285612       3.01426506      -2.32330513
C       -0.65326887       3.32036090      -1.70732403
O       -0.82637185       2.63587189      -0.55307001
C        0.26635113       1.80842292      -0.42423803
O        0.50404215       1.07012391       0.47730500
O        1.20395112       2.09938192      -1.54059100
H        0.99488211       3.42258286      -3.19497991
H       -1.42996383       3.95062685      -2.12583590
C        1.26803315       0.08491497       2.52308202
C        0.71784115      -0.92897701       3.20788789
O        0.64552510      -2.02956414       2.40690088
C        1.62233818      -1.82807600       1.46278298
O        2.07954812      -2.65240097       0.69964898
O        1.90033615      -0.53814703       1.52785194
H        1.60973608       0.97362196       3.01618791
H       -0.23638988      -0.89091206       3.98334789
C       -2.61711597      -1.50253499       0.01472599
C       -1.65134692      -2.33284712      -0.23200200
O       -0.75461686      -1.64041007      -1.12287402
C       -1.23930693      -0.42871904      -1.39447701
O       -0.76134789       0.30777997      -2.22421694
O       -2.37302589      -0.32890302      -0.69123900
H       -3.27383399      -1.43262005       0.79540098
H       -1.51131582      -3.31143308       0.29795000""",
fmt="xyz",
)
atoms = molecule.to_ase_atoms()

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




fl = MLIPAssistedGeoOptMaker().make(molecule, database_dict, 0.05,
                                    machine_learning_fit_kwargs={"foundation_model":"small", "device":"cpu", "default_dtype":"float32", "enable_cueq":False, "max_num_epochs":300},
                                    relax_calculator_kwargs={"device":"cuda", "enable_cueq":False})

resource = {"nodes": 1, "ntasks_per_node": 1, "cpus_per_task":20}

# Run relax job remotely
j_id = submit_flow(fl, worker="precision_tower_worker_mace", project="yiy_workstation", resources=resource)
print(fl)
print(j_id)


#flow_now = fl
#while True:
#    if flow_now is None:
#        break
#    for job in flow_now:
#        if job.name=="evaluate_max_force":
#            print(response[job.uuid][1].output)
#        if job.name=="get_mace_relax_job":
#            relax_job_uuid = response[job.uuid][1].replace[0].uuid
#            print(response[relax_job_uuid][1].output.output.n_steps, end = " ")
#    for job in flow_now:
#        if job.name=="check_convergence_and_next":
#            uuid_next = job.uuid
#    flow_now = response[uuid_next][1].replace
#    #print(flow_now)
