# NeuroQuake: Sparse SNN Models for Continual, Multi-Modal and Multi-Step Earthquake Forecasting

NeuroQuake investigates whether spiking neural networks (SNNs) can forecast earthquakes using continual, multi-modal and multi-step signals recorded in controlled laboratory fault experiments.  
The project uses the NEST simulator to construct a sparse 3D spiking reservoir that processes strain and acoustic-emission data encoded as spike trains.

## Features
- Preprocessing and spike encoding of strain and acoustic-emission signals  
- Automatic event detection and generation of multi-class targets  
- 3D sparse SNN reservoir simulated with NEST  
- Spike-based feature extraction and classification  
- Multi-step earthquake event forecasting and model evaluation

## Dataset
The dataset originates from **Cornell’s 0.76 m laboratory earthquake apparatus**.  
Only strain gauges and AE sensors are used in this implementation.  

## Citation
If you use this repository, please cite the project report:  
**NeuroQuake: Sparse SNN models for continual, multi-modal and multi-step earthquake forecasting** (ETH Zürich, 2024).
