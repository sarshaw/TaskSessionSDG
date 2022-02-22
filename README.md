# TaskSessionSDG
This is the repository that contains the synthetic search session datasets and codes to generate synthetic data.

## Repository Organization
**Input** contains real search session logs with task state labels - Session Track 2014 and KDD 2019 user study data. The Intention study data is not included here for confidentiality purposes.
**Output** contains generated synthetic datasets.
**SDG.ipynb** contains the code to generate synthetic data.

## Synthetic Data Generation
First, to generate new data based on a real dataset, upload the real data into the Input folder. Then run the SDG.ipynb. Before running the CTGANs model, input the data file and specify the data schema.
