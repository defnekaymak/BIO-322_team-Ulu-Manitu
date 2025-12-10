# BIO-322_team-Ulu-Manitu

# BIO_322 Miniproject – Decoding Neural Activity
Authors: Defne Kaymak and S. Ceren Erdoğan  
Course: BIO_322 – Biological data science II : machine learning 
Semester: Fall 2025

## Project Overview
The goal of this project is to decode mouse neural activity recorded during a whisker-based behavioral task. Each trial is labeled according to sensory stimulation, behavioral outcome (lick or no-lick), and task context (GO or NOGO). The input features consist of population neural activity averaged in 100 ms time bins across multiple brain areas.

Using this dataset, we train machine learning models to predict the trial type from neural activity alone. The notebook follows the structure required by the course:

1. Data inspection  
2. Preprocessing  
3. Linear baseline model  
4. Non-linear models  
5. Summary and conclusions  
6. Final training on full data and Kaggle submission

## Repository Structure
├── ml_project.ipynb # Final notebook with full workflow
├── submission.csv # Kaggle-formatted test predictions
├── requirements.txt # Python environment specification
└── README.md # Project description and instructions

## Methods Summary

### Data Inspection
We explored the dataset using various visualizations, including:
- distribution of mean firing rates per trial
- feature variances
- correlation matrices (subset of features)
- PCA projections of the feature space
- class distribution and class-conditional mean activity

These analyses provide insights into separability, redundancy, noise, and class imbalance.

### Preprocessing
A systematic preprocessing pipeline was applied:
1. Selection of neural activity features (metadata removed)  
2. Dropping all-NaN features  
3. Variance thresholding  
4. Correlation filtering  
5. Mean imputation  
6. Feature standardization  
7. PCA to 50 components  
8. Stratified train/validation split  

All preprocessing objects were stored and reused during test-set prediction to ensure reproducibility.

### Models Evaluated

#### Logistic Regression
- Multinomial classifier with tuned regularization parameter C  
- Achieved approximately 0.63 validation accuracy  

#### Random Forest
- Tuned using randomized search across depth, tree count, and split parameters  
- Achieved approximately 0.70 validation accuracy  

#### HistGradientBoosting (Best Model)
- Tuned across learning rate, depth, leaf nodes, and regularization  
- Achieved approximately 0.75 validation accuracy  
- Selected as the final model for test-set prediction

### Key Findings
- Non-linear models outperform the linear baseline, indicating that behavior-related neural activity is not linearly separable.  
- HistGradientBoosting achieved the best performance, suggesting boosted decision trees capture meaningful hierarchical interactions in neural data.  
- Frequent classes are predicted more accurately, while rare or overlapping trial types are more difficult due to class imbalance and similarity in neural representation.  
- Sensory stimulation conditions tend to be easier to decode than lick/no-lick decisions.

### Running the Notebook
1. Install dependencies:  
   `pip install -r requirements.txt`
2. Open the notebook:  
   `jupyter notebook ml_project_current_version.ipynb`
3. Run all cells sequentially.  
4. The final section generates `submission.csv`.

### Reproducibility
To ensure reproducibility, the project uses:
- a fixed random seed (RANDOM_STATE)  
- consistent preprocessing objects applied to both training and test data  
- Python environment specifications provided in requirements.txt  

Python version used:  
`Python 3.10.x`

## License
This repository is created for the EPFL BIO_322 miniproject and follows the course guidelines.

## Acknowledgements
Neural dataset provided by Parviz Ghaderi and Tam Nguyen, Petersen Lab (EPFL).  
Project structure and instructions provided by the BIO_322 teaching team.



