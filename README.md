# AQUARIUS-main (for paper)

This directory is a clean paper-submission copy of the AQUARIUS model code.
It keeps only the core model implementation and environment files from the
original project.

Runtime caches, datasets, checkpoints, logs, experiment outputs, compiled
Python caches, and backup files are intentionally excluded.

## Contents

- `model/tgnn/aquarius_model.py`
- `model/tgnn/aquarius_model_enhanced.py`
- `model/tgnn/bert_whitening.py`
- `model/tgnn/bert_whitening_aspect.py`
- `model/tgnn/load_data.py`
- `model/tgnn/model_run.py`
- `model/tgnn/nlp_util.py`
- `model/tgnn/quadruple_model.py`
- `model/tgnn/quadruple_model_v2.py`
- `model/tgnn/rhgc.py`
- `model/tgnn/rhgc_aspect.py`
- `model/tgnn/rhg_data.py`
- `model/tgnn/util.py`
- `model/tgnn/requirements.txt`

## Directory Layout

```text
AQUARIUS-main(for paper)/
|-- README.md
`-- model/
    `-- tgnn/
        |-- aquarius_model.py
        |-- aquarius_model_enhanced.py
        |-- bert_whitening.py
        |-- bert_whitening_aspect.py
        |-- load_data.py
        |-- model_run.py
        |-- nlp_util.py
        |-- quadruple_model.py
        |-- quadruple_model_v2.py
        |-- requirements.txt
        |-- rhgc.py
        |-- rhgc_aspect.py
        |-- rhg_data.py
        `-- util.py
```

## Environment

Activate the conda environment used by the original AQUARIUS project, then
install the Python dependencies:

```bash
conda activate <your-aquarius-env>
pip install -r model/tgnn/requirements.txt
```

Base model resources are stored on the server under:

```text
/data/cxf2022/dl_project/00.model_base
```

Adjust model paths in scripts or configuration files if the runtime environment
differs from the original server layout.

## Usage

Run scripts from the project root:

```bash
cd "/data/cxf2022/dl_project/AQUARIUS-main(for paper)"
conda activate <your-aquarius-env>
python model/tgnn/model_run.py
```

Before reproducing experiments, prepare the required datasets and pretrained
model paths according to the original project settings.
