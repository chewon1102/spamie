## SpaMIE My- Data Pipeline

=> Use RNA to impute missing protein layer 


This repo contains two scripts that together run a spatial multi-omics imputation experiment 
using the SpaMIE framework and then evaluate how well it worked. 

### 1. mydata_imputation_py = train the model and generate predictions 
The script trains a SpaMIE model to predict one omics modality from another for 
a spatial dataset, then saves the predicted vs true values to disk. 

Steps it performs: 

1. Load data - Read two AnnData (.h5ad) files: 
    adata_omics1: GSM8195494_A1_LN.h5ad (e.g. spatial transcriptomics)
adata_omics2: GSM8195498_A1_LN_Protein.h5ad 
(protein expression — the modality being predicted)

2. Build graphs - calls Sagegraph() on both modalities to construct for each 
    a spatial graph (based on physical location of spots/cells)
    a feature graph (based on similarity in expression space) 
    
3. Configure the model - creates a Sagewrapper model with 
    - task= 'prediction' - the model's job is to predict omics2 (protein) from omics1
    - in_feat/out_feat - input/output feature dimensions inferred from data
    - n_hidden=256, layers_num= 3 - network size/depth
    - weight = [0,0,2] - loss term weighing 
    (RNA reconstruction loss, protein reconstruction loss, cross modal prediction loss) 
    - epoch = 50 - training epochs (explicitly noted in the code as a low value 
    "for testing")
    - Other hyperparameters: learning rates(lr, lr2), activation(LeakyReLU),
    aggregation type(sagetype = "mean), 
    residual connection style(res_type = "res_add")
4. Train (model.fit) - trains the model on the constructed graphs and saves 
results to output_dir: 
    my_data_pred.csv - the models' predicted protein expression
    my_data_truth.csv = corresponding ground truth protein expression 
    
Output: Two CSV files written to /users/coh33/SpaMIE/results/mydata/. 

## mydata_eval.py - evaluate prediction accuracy 
This script loads the prediction output from step 1 and quantifies/visualizes 
how close the predictions are to the ground truth. 

1. Load results
2. Compute Pearson correlation - for each protein(column), computes the pcc between 
predicted and true values, then reports the mean pearson correlation across all proteins 
3. Compute RMSE 
4. Save metrics 
5. Scatter plot
    - Plots every predicted value against its true value (flattened across all
    proteins), with a dashed diagnoal (y=x) reference line, and saves the png 

6. Per-protein bar plot - computes pearson correlation separately for each protein 


mydata_imputation.py                    mydata_eval.py
─────────────────────                   ──────────────
Load h5ad data                          Load pred/truth CSVs
   │                                          │
Build spatial + feature graphs          Compute Pearson + RMSE
   │                                          │
Train SpaMIE model (prediction task)    Save metrics.txt
   │                                          │
Save my_data_pred.csv  ───────────────►  Plot scatter (pred vs true)
Save my_data_truth.csv ───────────────►  Plot per-protein bar chart



