# Multi-Check-Worthy Dataset (MultiCW)

This repository contains the code and datasets for the paper "MultiCW - A Large-Scale Balanced Benchmark Dataset for Training Robust Check-Worthiness Detection Models". The project introduces the Multi-Check-Worthy dataset (MultiCW), a benchmarking dataset of check-worthy claims that spans multiple languages, topics, and writing styles.

## Abstract

Large language models (LLMs) are beginning to reshape how media professionals verify information, yet automated support for detecting check-worthy claims — a key step in the fact-checking process — remains limited. We introduce the Multi-Check-Worthy (MultiCW) dataset, a multilingual benchmark for check-worthy claim detection spanning 16 languages, six topical domains, and two writing styles. It consists of 123,722 samples, evenly distributed between noisy (informal) and structured (formal) texts, with balanced representation of check-worthy and non-check-worthy classes across all languages. To probe robustness, we also introduce an equally balanced out-of-distribution evaluation set of 29,647 samples in 4 additional languages.
We benchmark three fine-tuned multilingual transformers against a diverse set of 14 commercial and open-source LLMs under zero-shot settings. Our findings show that fine-tuned models consistently outperform zero-shot LLMs on claim classification and show strong out-of-distribution generalization across languages, domains, and styles. MultiCW provides a rigorous multilingual resource for advancing automated fact-checking and enables systematic comparisons between fine-tuned models and cutting-edge LLMs on the check-worthy claim detection task.

## Project structure

- **Source-datasets**: Contains the datasets used to create the MultiCW dataset.

- **Final-dataset**: Contains the partial files used to compile the MultiCW dataset together with the final dataset and the train, validation, and test sets all in CSV format.

- **Tools**: Contains the detailed implementation of all the processes used in the notebooks.
 
- **1-MultiCW-dataset**: Notebook for setting up and exploring the MultiCW dataset.

- **2-Models-fine-tuning**: Notebook for fine-tuning XLM-RoBERTa and mDeBERTa models on the MultiCW dataset and their evaluation.

- **3-Models-fine-tuning-LESA**: Notebook for fine-tuning LESA model on the MultiCW dataset and its evaluation.

## Conda environment setup
There are three jupyter notebooks in this project for each of which we need to create a specific conda environment:
* MultiCW-dataset   : ***1-MultiCW-dataset notebook***
* MultiCW-fine-tune : ***2-Models-fine-tuning notebook***
* MultiCW-LESA      : ***3-Models-fine-tuning-LESA notebook***

To make the creation and setup of the conda environments as simple as possible, we have prepared the shell script to automate the process. 
To run the shell script simply run the following commands:

```bash
cd /<your-path>/MultiCW
chmod +x setup.sh
./setup.sh
```
When prompted, enter your path to your Conda installation, i.e.:
```bash
/home/yourname/miniconda3
```

### Manual installation of the conda environments
In case you want to install the conda environments manually, you can follow the following steps.  

#### MultiCW Dataset Notebook

```bash
conda create --name MultiCW-dataset python=3.10
conda activate MultiCW-dataset
pip install jupyterlab
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=MultiCW-dataset
conda install -c conda-forge gliner
conda install -n base -c conda-forge jupyterlab_widgets
jupyter labextension install js
```

#### MultiCW Fine-Tuning Notebook - mDeBERTa & xlm-RoBERTa models

```bash
conda create --name MultiCW-finetune python=3.10
conda activate MultiCW-finetune
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=MultiCW-finetune
conda install -n base -c conda-forge jupyterlab_widgets
pip install -r requirements-finetune.txt
python -m spacy download en_core_web_sm
```

#### MultiCW Fine-Tuning Notebook - LESA model

```bash
conda create --name MultiCW-lesa python=3.10
conda activate MultiCW-lesa
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=MultiCW-lesa
conda install -n base -c conda-forge jupyterlab_widgets
pip install -r requirements-lesa.txt
python -m spacy download en_core_web_sm
```

## References

* [CLEF2022-CheckThat! Lab dataset](https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab)
* [CLEF2023-CheckThat! Lab dataset](https://gitlab.com/checkthat_lab/clef2023-checkthat-lab)
* [LESA dataset (2021)](https://github.com/LCS2-IIITD/LESA-EACL-2021)
* [MultiClaim dataset](https://zenodo.org/records/7737983)
* [Ru22Fact dataset](https://paperswithcode.com/paper/ru22fact-optimizing-evidence-for-multilingual)