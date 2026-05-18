# AQUARIUS-main (for paper)

This directory is a clean paper-submission copy of the AQUARIUS model code.
It keeps only the core model implementation and environment files from the
original project.


## Directory Layout

```text
README.md
`model/
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
conda activate aquarius_env
pip install -rrequirements.txt
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
