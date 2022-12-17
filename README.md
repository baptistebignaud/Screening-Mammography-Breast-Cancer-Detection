<font size="6"> RSNA-Screening-Mammography-Breast-Cancer-Detection-kaggle</font><br><br>

**This repository's aim is to contribute to the kaggle challenge RSNA-Screening-Mammography-Breast-Cancer-Detection** <br><br>



<font size="6"> Table of contents</font>

-  [Introduction to the topic](#introduction-to-the-topic)
-  [The data](#the-data)
-  [Models](#models)
-  [Results](#results)
-  [Openings (ordered by increasing difficulty in our opinion)](#openings-ordered-by-increasing-difficulty-in-our-opinion)
-  [Usefull commands:](#usefull-commands)
-  [References](#references)

# Introduction to the topic

Input: données tabulaires + des screens de mamographie par patient 

Output: patient positive au cancer ou non.

tabular data: 

# The data

TODO

# Models

TODO

# Results
TODO

# Openings 

TODO


# Usefull commands:

Be carefull, the repository is built to give the possibility to use a GPU. If you don't have a GPU on your computer, please comment lines below _deploy_ in the docker-compose file or it won't work.

## Build the container<br>

> `docker-compose build ` <br>

## Up containers<br>

> `docker-compose up -d` <br>

## Stop containers<br>

> `docker-compose stop` <br>

## Open a shell inside the main container<br>

> `docker-compose exec rmbscd sh `

## Run jupyter lab from the container<br>

> `jupyter lab --ip 0.0.0.0 --allow-root`

# References

TODO