# Multi-Check-Worthy Dataset (MultiCW)

This repository contains the code and datasets for the paper "MultiCW - A Large-Scale Balanced Benchmark Dataset for Training Robust Check-Worthiness Detection Models". The project introduces the Multi-Check-Worthy dataset (MultiCW), a benchmarking dataset of check-worthy claims that spans multiple languages, topics, and writing styles.

## Abstract

Large language models (LLMs) are beginning to reshape how media professionals verify information, yet automated support for detecting check-worthy claims—a key step in the fact-checking process—remains limited. We introduce the Multi-Check-Worthy (MultiCW) dataset, a balanced multilingual benchmark for check-worthy claim detection spanning 16 languages, six topical domains, and two writing styles. It consists of 123,722 samples, evenly distributed between noisy (informal) and structured (formal) texts, with balanced representation of check-worthy and non-check-worthy classes across all languages. To probe robustness, we also introduce an equally balanced out-of-distribution evaluation set of 27,761 samples in 4 additional languages.

To provide baselines, we benchmark three common fine-tuned multilingual transformers against a diverse set of 15 commercial and open LLMs under zero-shot settings. Our findings show that fine-tuned models consistently outperform zero-shot LLMs on claim classification and show strong out-of-distribution generalization across languages, domains, and styles.

MultiCW provides a rigorous multilingual resource for advancing automated fact-checking and enables systematic comparisons between fine-tuned models and cutting-edge LLMs on the check-worthy claim detection task.

## Project structure

- **Source-datasets**: Contains the datasets used to create the MultiCW dataset.

- **Final-dataset**: Contains the partial files used to compile the MultiCW dataset together with the final dataset and the train, validation, and test sets all in CSV format.

- **Tools**: Contains the detailed implementation of all the processes used in the notebooks.
 
- **1-MultiCW-dataset**: A notebook for setting up and exploring the MultiCW dataset.

- **2-Models-fine-tuning**: A notebook for fine-tuning XLM-RoBERTa and mDeBERTa models on the MultiCW dataset and their evaluation.

- **3-Models-fine-tuning-LESA**: A notebook for fine-tuning LESA model on the MultiCW dataset and its evaluation.

- **4-Pilot-prompt-study**: A notebook containing the Pilot prompting study carried out on the small subset (100) of samples randomly selected from the original dataset.

- **5-Zero-shot evaluation**: A notebook allowing to access the zero-shot LLM evaluaton results. For details on the LLM evaluation process, see the ```llm_prompting.py``` file in the ```Tools``` folder.

## Fine-tuned Transformer models download
The models are available on the following location: [link](https://drive.google.com/drive/folders/1Pkn2OPlJeMz-0dkjb7dXFpp4SL8vdK0P)

To make the models accessible from the code, place them into the ```Models``` folder in the project root directory. 

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

## Acknowledge
This work was partially supported by the European Union under the Horizon Europe projects: \textit{vera.ai} (GA No. \href{https://doi.org/10.3030/101070093}{101070093}), and by \textit{AI-CODE} (GA No. \href{https://cordis.europa.eu/project/id/101135437}{101135437}); and by EU NextGenerationEU through the Recovery and Resilience Plan for Slovakia under the project No. 09I01-03-V04-00006.
