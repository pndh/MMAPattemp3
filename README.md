# ST-problem baselines

Here, we reimplemented a number of different models for a task related to Spatial Transcriptomics (ST) - predicting gene expression from Whole Slide Images (WSIs)

Models implemented so far: 
* ST-Net
* HisToGene
* DeepPT
  
Models not yet implemented:
* Hist2ST
* EGN

# Installation guide

* Download data: Run "git clone https://github.com/almaan/her2st.git" in the "data" folder to clone the HER2ST dataset  
* To unzip files in ST-cnts, cd into "Net/data/her2st/data/ST-cnts/" and run "gunzip *.gz"  
* Run "pip install -r requirements.txt" to download all the necessary Python libraries (with the versions we are using)

Note: You may encounter installation issues. In such case, refer to the Guide for Installation or Compatibility issues (.txt file)

# Usage

For ST-Net and HisToGene:  
1. Train: Run ST_train.py and choose the model you wish to use. The model will conduct training and save checkpoints once completed.  
2. Test: Run ST_predict.py and choose the model you wish to use. Results printed are shown in metrics (printed in cmd line) and figures (imgs saved in the "figures" folder)  

For DeepPT:
(To be updated)

# References and links to source codes
[**ST-Net**](https://github.com/bryanhe/ST-Net)
> [**Integrating spatial gene expression and breast tumour morphology via deep learning**](https://rdcu.be/b46sX)  
  by Bryan He, Ludvig Bergenstråhle, Linnea Stenbeck, Abubakar Abid, Alma Andersson, Åke Borg, Jonas Maaskola, Joakim Lundeberg & James Zou.  
  <i>Nature Biomedical Engineering</i> (2020).

Since the code provided in their GitHub has been deprecated, we referred to the re-implemented versions from authors of HisToGene and Hist2ST to reproduce our own re-implemented version of ST-Net

[**HisToGene**](https://github.com/maxpmx/HisToGene)
> [**Leveraging information in spatial transcriptomics to predict super-resolution gene expression from histology images in tumors**](https://doi.org/10.1101/2021.11.28.470212)  
  by Minxing Pang, Kenong Su, Mingyao Li.  
  <i>biorxiv</i> (2021).

[**DeepPT**](https://zenodo.org/records/11125591)
> [**A deep-learning framework to predict cancer treatment response from histopathology images through imputed transcriptomics**](https://www.nature.com/articles/s43018-024-00793-2)  
  by Hoang, DT., Dinstag, G., Shulman, E.D. et al.  
  <i>Nature Cancer</i> (2024).

[**Hist2ST**](https://github.com/biomed-AI/Hist2ST)
> [**Spatial transcriptomics prediction from histology jointly through Transformer and graph neural networks**](https://doi.org/10.1093/bib/bbac297)  
 by Zeng Y, Wei Z, Yu W, Yin R, Yuan Y, Li B, Tang Z, Lu Y, Yang Y.  
  <i>Brief Bioinform</i> (2022).

[**EGN**](https://github.com/Yan98/EGN)
> [**Exemplar Guided Deep Neural Network for Spatial Transcriptomics Analysis of Gene Expression Prediction**](https://doi.org/10.48550/arXiv.2210.16721)  
 by Yan Yang, Md Zakir Hossain, Eric A Stone, Shafin Rahman.  
  <i>arxiv</i> (2022).
# MMAPattemp3
