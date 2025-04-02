#!/usr/bin/env python

from jobflow_remote import submit_flow, get_jobstore
from jobflow_remote.cli.utils import get_job_controller, initialize_config_manager, get_config_manager
import json
initialize_config_manager()
jc = get_job_controller()
js = get_jobstore()
js.connect()
energies = []
max_forces = []
structures = []
for job_uuid in jc.get_flows_info(db_ids="365")[0].job_ids:
    job_info = jc.get_job_info(job_uuid)
    if job_info.name=="GFN-xTB static":
        try:
            energies.append(js.get_output(job_uuid)["output"]["energy"])
            structures.append(js.get_output(job_uuid)["output"]["molecule"])
        except:
            pass
    if job_info.name=="evaluate_max_force":
        try:
            max_forces.append(js.get_output(job_uuid))
        except:
            pass
data={"energies": energies, "max_forces": max_forces, "structures": structures}
with open('gaims_geoopt_result.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
