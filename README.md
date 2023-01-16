<font size="6"> RSNA-Screening-Mammography-Breast-Cancer-Detection-kaggle</font><br><br>

**This repository's aim is to contribute to the kaggle challenge RSNA-Screening-Mammography-Breast-Cancer-Detection** <br><br>

<font size="6"> Table of contents</font>

- [Introduction to the topic](#introduction-to-the-topic)
- [The data](#the-data)
- [Models](#models)
- [Results](#results)
- [Openings](#openings)
- [Usefull commands and installation](#usefull-commands-and-installation)
- [References](#references)

# Introduction to the topic

Input: données tabulaires + des screens de mamographie par patient

Output: patient positive au cancer ou non.

tabular data:

TODO Add an introduction and answer theses question @Reda:

- Qu’est ce que le cancer du sein:
  - Origine
  - Ses possibles impacts sur la santé (ablation, mort …)
  - Combien de femmes sont atteintes chaque année; combien de personnes meurent?
- Comment détecter le cancer du sein:
  - Mammographie —> Qu’est ce qu’une mammographie (introduction)?
  - Comment les médecins peuvent détecter des possibles tumeurs (important pour l’approche qu’on va avoir)?
  - Quels sont les suivis?
  - Peut-on détecter les différentes phases du cancer depuis une mammographies?
- Le challenge:
  - Quels sont les tenants et aboutissants?
  - De quoi est composé notre jeu de donnée?
  - Comment la donnée a été récoltée; que doit-on prédire; petite phrase explicative pour chaque feature?
  - Où est en la recherche en IA aujourd’hui dans le domaine médical et dans la détection de cancers du sein

# The data

TODO: Do a table of features with type of each feature and introduction of images' dataset.

# Models

TODO

# Results

TODO

# Openings

TODO

# Usefull commands and installation

To use the repository, you will need to add images dataset (in DCOM or PNG format) in a folder `kaggle_dataset` at the root of the project.

We advise you to use the docker configuration, but if one prefers to work without it, you will need to install requirements with `pip3 install -r requirements.txt`.

Be carefull, the repository is built to give the possibility to use a GPU. If you don't have a GPU on your computer, please comment lines below _deploy_ in the docker-compose file or it won't work.

## Build the container<br>

> `docker-compose build` <br>

## Up containers<br>

> `docker-compose up -d` <br>

## Stop containers<br>

> `docker-compose stop` <br>

## Open a shell inside the main container<br>

> `docker-compose exec rmbscd sh`

## Run jupyter lab from the container<br>

> `jupyter lab --ip 0.0.0.0 --allow-root`

## If some issues with jupyter lab or port already on used on 8888

> `lsof -i :8888` <br>
> `kill -9 <PID>` <br>
With PID being the process ID of python (for notebook)

## Run training model

> `python3 train.py --images_dir "../kaggle_dataset" --csv_file_path "../data/train.csv" --model "EfficientNet" --num_epochs 1`

# References

TODO
