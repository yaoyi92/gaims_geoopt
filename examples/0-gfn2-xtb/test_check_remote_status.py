#!/usr/bin/env python

from jobflow_remote import submit_flow, get_jobstore
from jobflow_remote.cli.utils import get_job_controller, initialize_config_manager, get_config_manager
initialize_config_manager()
jc = get_job_controller()
js = get_jobstore()
js.connect()
for job_uuid in jc.get_flows_info(db_ids="311")[0].job_ids:
    job_info = jc.get_job_info(job_uuid)
    if job_info.name=="evaluate_max_force":
        print(js.get_output(job_uuid))
