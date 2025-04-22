# gaims_geoopt

**Machine‑Learning‑assisted Geometry Optimisation with FHI-aims, MACE, and `jobflow`**

`gaims_geoopt` orchestrates an *active‑learning* workflow that couples high‑accuracy
FHI‑aims reference calculations with fast MACE force‑field
relaxations.  The loop automatically fine‑tunes an interatomic
potential on‑the‑fly, expands its training database, and decides when the
geometry is converged – offering the accuracy of *ab‑initio* methods at a
fraction of the cost.

<br>

---

## Features

- **Hybrid optimisation loop**: GAIMS reference → database update → MACE fit →
  ML‑relax → *repeat until converged*.
- Supports GFN2‑xTB (**molecular**) *and* FHI-aims (**molecular/periodic**) as references.
- Runs anywhere `jobflow` can: local machine, HPC scheduler via
  [`jobflow_remote`](https://materialsproject.github.io/jobflow-remote/).
- Rolling in‑memory EXTXYZ database keeps the workflow lightweight.
- Highly configurable via keyword overrides – tweak training hyper‑parameters,
  convergence criteria, optimiser settings, etc.

---

> **Tip**   Use the supplied `environment.yml` for an exact, reproducible setup.

---

## Installation

```bash
# 1. Create & activate a clean environment (Python 3.11)
python3.11 -m venv gaims_geoopt_env
source gaims_geoopt_env/bin/activate

# 2. Install core + optional workflow managers
pip install .[workflow-managers]
```

### Extra steps

| Scenario            | What to do                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **Remote execution** (`jobflow_remote`) | 1. Add your cluster to `~/.jobflow_remote.toml`.<br>2. Launch the worker with `jfr runner &`. |
| **FHI‑aims**        | Make sure atomate2 FHI-aims works fine.|

---

## Quick Start

```python
from gaims_geoopt import MLIPAssistedGeoOptMaker
from jobflow import run_locally

fl = MLIPAssistedGeoOptMaker().make(molecule, database_dict, 0.05)
response = run_locally(fl, create_folders=True)
```

The optimiser will iterate until either `max_force_criteria` is met, the ML
relaxation stalls, or `max_gaims_geoopt_steps` is exceeded.

---

