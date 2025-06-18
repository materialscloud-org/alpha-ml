# Alpha-ML

AlphaML was developed by the Ceriotti group at EPFL and the DiStasio group at Cornell. AlphaML uses a machine-learning framework based on comparisons of local atomic environments to predict the polarizabilities of molecules. This model was trained on the 7211 compounds in the QM7b dataset, containing up to 7 heavy atoms (C,N,O,S,Cl) as well as hydrogen. For these molecules, the polarizability was calculated using B3LYP-DFT, SCAN0-DFT and CCSD.

The compute module is handled via celery and redis. While the frontend website is built using the shiftml-pip2 branch of the [tools-barebones](https://github.com/materialscloud-org/tools-barebone) framework.

## Usage

This is a fully dockerised solution for use in deployment on materialscloud.org and requires the data.tar.gz file (which is availible on the MC S3 server).

With the zipped data file on the same level as docker-compose.yml:

```
docker compose build #Usually takes 5+ mins
docker compose up
```

where the landing page should be rendered on:

```
localhost:5050
```
