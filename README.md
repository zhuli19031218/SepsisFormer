# SepsisFormer and SMART
This repository contains the implementation of the paper
"Explainable AI-driven heterogeneity using coagulation–inflammatory markers improves prognosis prediction, risk stratification, and anticoagulant treatment effects for sepsis".
The code includes all major data, models, algorithms, and experimental settings described in the paper, enabling researchers to reproduce and verify our results. Enjoy~~~

Sepsis, a leading cause of hospital mortality, is characterized by substantial heterogeneity, hindering the development of effective and interpretable prognostic and stratification methods. To address this challenge, we developed an explainable prognostic model (SepsisFormer, a transformer-based deep neural network with an enhanced domain-adaptive generator) and an automated risk stratification tool (SMART, a scorecard consistent with medical knowledge). In a multi-center retrospective study of 12,408 sepsis patients, SepsisFormer achieved high predictive accuracy (AUC: 0.9301, sensitivity: 0.9346, and specificity: 0.8312). SMART (AUC: 0.7360) surpassed most established scoring systems. Based on SMART, four risk levels (mild, moderate, severe, dangerous) can be identified by using seven coagulation-inflammatory routine laboratory measurements and patient age, and the corresponding mortality is approximately 5%, 15%, 30%, and 50%, respectively. Meanwhile, two subphenotypes (CIS1 and CIS2) can be classified through unsupervised GMM, and CIS2 has a worse survival prognosis. Notably, patients with moderate/severe levels or CIS2 derive more significant benefits from anticoagulant treatment. In conclusion, explainable artificial intelligence (SepsisFormer) drives risk stratification and subphenotypic classification of sepsis, which help to guide anticoagulant treatment. Our work, therefore, offers a novel set of simple, real-time executable tools for sepsis heterogeneity, demonstrating considerable potential to significantly enhance sepsis clinical practice globally, particularly in resource-constrained healthcare settings.

NOTE: The Source Data for the 36 and 8 sepsis markers is provided in the "figure 2ab" folder, which contains the datasets used for Figures 2a, 2b, and 2e. The underlying code for these figures is identical; however, the input datasets differ: Figures 2a and 2b use 36 sepsis markers, while Figure 2e uses 8 sepsis markers.

# Code Guide

This repository contains comprehensive code and data for generating all figures presented in the manuscript. This documentation provides detailed step-by-step instructions to facilitate the reproduction of all results reported in the study.

## 📋 Table of Contents

- [Environment Requirements](#environment-requirements)
- [Project Structure](#project-structure)
- [Execution Procedures](#execution-procedures)
  - [Figure 2a2b: Multi-Model Comparative Analysis](#figure-2a2b-multi-model-comparative-analysis)
  - [Figure 2c: Network Analysis Visualization](#figure-2c-network-analysis-visualization)
  - [Figure 2d: Domain Adaptation Analysis](#figure-2d-domain-adaptation-analysis)
  - [Figure 2f: Clustering Visualization](#figure-2f-clustering-visualization)
  - [Figure 2g: SHAP Feature Importance Analysis](#figure-2g-shap-feature-importance-analysis)
  - [Figure 3c3h: Hierarchical Clustering Analysis](#figure-3c3h-hierarchical-clustering-analysis)
  - [Figure 3d: 3D Visualization](#figure-3d-3d-visualization)
  - [Figure 4a4c: Cox Proportional Hazards Model](#figure-4a4c-cox-proportional-hazards-model)
  - [Figure 4b: Radar Chart Analysis](#figure-4b-radar-chart-analysis)
  - [Figure 4d: Risk Ratio Analysis](#figure-4d-risk-ratio-analysis)

## 🔧 Environment Requirements

### Python Dependencies

```bash
pip install numpy pandas scikit-learn torch torchvision
pip install matplotlib seaborn plotly networkx
pip install shap shapely scipy statsmodels
pip install jupyter notebook
pip install igraph python-igraph
pip install lifelines
```

### Core Dependency Versions

- Python >= 3.7
- PyTorch >= 1.8.0
- scikit-learn >= 0.24.0
- pandas >= 1.3.0
- numpy >= 1.21.0

### Optional Software

- **GraphPad Prism**: For statistical chart generation in Figure 4a4c
- **Origin**: For clustering visualization charts in Figure 2f
- **R Language**: For statistical analysis (if using R scripts)

## 📁 Project Structure

```
figure/
├── 2a2b/                  # Figure 2a2b: Multi-model comparative analysis
│   ├── model2/            # Deep learning model code
│   │   ├── Transformer.py # Transformer model implementation
│   │   ├── model_LSTM.py  # LSTM model implementation
│   │   ├── model_GRU.py   # GRU model implementation
│   │   ├── model_gpt.py   # GPT model implementation
│   │   ├── train.py       # Training script
│   │   ├── main_mimic.py  # Main execution script
│   │   └── results/       # Model results
│   ├── data/              # Data files
│   └── ml_logs/           # Machine learning logs
├── 2c/                    # Figure 2c: Network analysis visualization
│   ├── data/              # Node and edge data
│   ├── log/               # Execution logs
│   └── video/             # Generated videos
├── 2d/                    # Figure 2d: Domain adaptation analysis
│   ├── models/            # Machine learning model code
│   ├── logs/              # Experimental results
│   ├── roc_domain_adaptation.ipynb
│   └── video/
├── 2f/                    # Figure 2f: Clustering visualization
│   ├── Chord Diagram/      # Chord diagram
│   ├── Heatmap/           # Heatmap
│   └── Radar Chart/       # Radar chart
├── 2g/                    # Figure 2g: SHAP analysis
│   ├── data/              # Data files
│   ├── logs/              # Model logs
│   ├── SepsisFormer_shap_8.py
│   ├── shap_8 (2).ipynb
│   └── video/
├── 3c3h/                  # Figure 3c3h: Hierarchical clustering
│   ├── data/              # Data files
│   ├── log/               # Clustering logs
│   └── video/             # Clustering videos
├── 3d/                    # Figure 3d: 3D visualization
│   ├── data/              # Data files
│   ├── 3d.ipynb           # 3D visualization code
│   └── video/
├── 4a4c/                  # Figure 4a4c: Cox model
│   ├── 4a/                # New scoring analysis
│   ├── 4c/                # Subphenotype analysis
│   └── *.ipynb            # Cox analysis code
├── 4b/                    # Figure 4b: Radar chart
│   ├── data/              # Radar chart data
│   └── video/
└── 4d/                    # Figure 4d: Risk ratio
    ├── 风险比.ipynb
    └── video/
```

## 🚀 Execution Procedures

### Figure 2a2b: Multi-Model Comparative Analysis

**Objective**: Compare the performance of different deep learning models (Transformer, LSTM, GRU, GPT) on sepsis prediction tasks

**Primary Files**:

- `2a2b/model2/main_mimic.py`: Main execution script
- `2a2b/model2/train.py`: Training and evaluation script
- `2a2b/model2/Transformer.py`: Transformer model implementation
- `2a2b/model2/model_LSTM.py`: LSTM model implementation
- `2a2b/model2/model_GRU.py`: GRU model implementation
- `2a2b/model2/model_gpt.py`: GPT model implementation
- `2a2b/data/SepsisFormer.py`: SepsisFormer model implementation

**Data Files**:

- `2a2b/data/8/`: Dataset with 8 features
- `2a2b/data/36/`: Dataset with 36 features

**Execution Steps**:

1. **Environment Preparation**:

   ```bash
   cd 2a2b/model2
   ```

2. **Train Transformer Model**:

   ```bash
   python main_mimic.py --model_name Transformer --factors 8 --lr 0.004 --epoch 50 --batch_size 3762
   ```

3. **Train LSTM Model**:

   ```bash
   python main_mimic.py --model_name Lstm --factors 8 --lr 0.004 --epoch 50 --batch_size 3762
   ```

4. **Train GRU Model**:

   ```bash
   python main_mimic.py --model_name GRU --factors 8 --lr 0.004 --epoch 50 --batch_size 3762
   ```

5. **Train GPT Model**:
   ```bash
   python main_mimic.py --model_name GPT --factors 8 --lr 0.004 --epoch 50 --batch_size 3762
   ```

**Model Parameter Specifications**:

- `--model_name`: Select model type (Transformer/Lstm/GRU/GPT)
- `--factors`: Number of input features (8 or 36)
- `--lr`: Learning rate
- `--epoch`: Number of training epochs
- `--batch_size`: Batch size
- `--pretrain`: Whether to use pre-trained models
- `--loadmodel`: Path to pre-trained model

**Model Architecture**:

- **Transformer**: Multi-head attention mechanism, 8-layer depth
- **LSTM**: Long Short-Term Memory network, 2-4 layers
- **GRU**: Gated Recurrent Unit, 2-4 layers
- **GPT**: Generative Pre-trained Transformer

**Output Results**:

- Model performance metrics (AUC, Accuracy, F1-score, MCC)
- Training logs and TensorBoard visualization
- ROC curve plots
- Model comparison result tables

**Expected Performance** (based on 8 features):

- **GRU**: AUC ~0.645, Accuracy ~0.611
- **LSTM**: AUC ~0.641, Accuracy ~0.611
- **Transformer**: AUC ~0.630, Accuracy ~0.600
- **GPT**: Performance pending evaluation

### Figure 2c: Network Analysis Visualization

**Objective**: Generate network relationship diagrams among sepsis patient features

**Generation Method**: Online generation using HiPlot website

**Data Files**:

- `2c/data/节点数据.csv`: Network node information
- `2c/data/连线数据-*.csv`: Edge data at different thresholds

**Execution Steps**:

1. Access HiPlot website (https://hiplot.com.cn/)
2. Upload node data and edge data files
3. Select network graph visualization type
4. Configure parameters:
   - Node label column: `media`
   - Node color column: `weight`
   - Node size column: `media.type`
   - Edge width column: `weight`
   - Layout style: Circular layout
5. Generate network analysis diagram

**Parameter Specifications**:

- Node label column: `media`
- Node color column: `weight`
- Node size column: `media.type`
- Edge width column: `weight`
- Layout style: Circular layout

**Output**: Network analysis video files

**Reference**: Please refer to the operation steps in `2c/video/Network_Igraph.mp4`

### Figure 2d: Domain Adaptation Analysis

**Objective**: Analyze domain adaptation effects between different data sources

**Primary Files**:

- `2d/models/domain_adaptation.py`: Domain adaptation algorithm implementation
- `2d/models/随机森林.py`: Random Forest model
- `2d/models/逻辑回归.py`: Logistic Regression model
- `2d/roc_domain_adaptation.ipynb`: ROC analysis

**Execution Steps**:

1. Prepare source and target domain data
2. Execute domain adaptation algorithm:
   ```bash
   cd 2d/models
   python domain_adaptation.py
   ```
3. Train machine learning models:
   ```bash
   python 随机森林.py
   python 逻辑回归.py
   python SVM.py
   python SGD.py
   ```
4. Execute ROC analysis:
   ```bash
   jupyter notebook roc_domain_adaptation.ipynb
   ```

**Domain Adaptation Methods**:

- `mean_teacher`: Mean teacher method
- `whitening`: Whitening method

**Output**: ROC curve plots and domain adaptation effect comparison

### Figure 2f: Clustering Visualization

**Objective**: Generate multiple visualization charts for clustering results

**Generation Method**: Generated using Origin software

**Content Included**:

- **Chord Diagram** (`Chord Diagram/`): Display relationships between clusters
- **Heatmap** (`Heatmap/`): Feature correlation heatmap
- **Radar Chart** (`Radar Chart/`): Cluster feature radar chart

**Data Files**:

- `2f/Chord Diagram/data/雷达图与和铉图数据.xlsx`
- `2f/Heatmap/data/36.xlsx`
- `2f/Radar Chart/*.oggu`: Origin project files

**Execution Steps**:

1. Open `.oggu` files using Origin software
2. Import corresponding data files
3. Generate chord diagram to display cluster relationships
4. Calculate feature correlations and generate heatmap
5. Create radar chart to display cluster features

**Output**: Chord diagram, heatmap, and radar chart video files

**Software Requirements**: Origin software

### Figure 2g: SHAP Feature Importance Analysis

**Objective**: Analyze feature importance of SepsisFormer model using SHAP method

**Primary Files**:

- `2g/SepsisFormer_shap_8.py`: SepsisFormer model implementation
- `2g/shap_8 (2).ipynb`: SHAP analysis code

**Execution Steps**:

1. Load pre-trained SepsisFormer model
2. Prepare test data
3. Execute SHAP analysis:
   ```bash
   cd 2g
   jupyter notebook "shap_8 (2).ipynb"
   ```

**Model Architecture**:

- Transformer-based architecture
- Multi-head attention mechanism
- Feed-forward neural network

**Output**: SHAP feature importance plots and interpretability analysis

### Figure 3c3h: Hierarchical Clustering Analysis

**Objective**: Perform hierarchical clustering analysis on sepsis patients

**Data Files**:

- `3c3h/data/mimic4_level_subphenotype_heparin.csv`

**Execution Steps**:

1. Load patient data
2. Execute hierarchical clustering algorithm
3. Generate clustering results at different levels
4. Visualize clustering dendrogram

**Clustering Methods**:

- Hierarchical Clustering
- Subphenotype Clustering

**Output**: Hierarchical clustering dendrogram and subphenotype analysis videos

### Figure 3d: 3D Visualization

**Objective**: Generate 3D visualization of sepsis patient features

**Primary Files**:

- `3d/3d.ipynb`: 3D visualization code
- `3d/a.html`: Interactive 3D chart

**Execution Steps**:

1. Load patient data
2. Execute 3D visualization:
   ```bash
   cd 3d
   jupyter notebook 3d.ipynb
   ```

**Visualization Types**:

- 3D scatter plots
- Interactive 3D charts
- Feature space visualization

**Output**: 3D visualization plots and interactive HTML files

### Figure 4a4c: Cox Proportional Hazards Model

**Objective**: Analyze prognostic prediction capabilities of different scoring systems and subphenotypes

**Generation Method**: Generated using GraphPad Prism software

**Primary Files**:

- `4a4c/cox比例_mimic4_heparin.ipynb`: Heparin-related Cox analysis
- `4a4c/cox比例_mimic4_8_subphenotype_level.ipynb`: Subphenotype Cox analysis
- `4a4c/4a/新评分.pzfx`: GraphPad Prism project file
- `4a4c/4c/亚表型.pzfx`: GraphPad Prism project file

**Data Files**:

- `4a4c/score_hierarchy_mimic4_*.csv`: Scoring data at different levels

**Execution Steps**:

1. Execute Cox analysis to obtain data:

   ```bash
   cd 4a4c
   jupyter notebook cox比例_mimic4_heparin.ipynb
   jupyter notebook cox比例_mimic4_8_subphenotype_level.ipynb
   ```

2. Use GraphPad Prism software:
   - Open `4a4c/4a/新评分.pzfx` file
   - Open `4a4c/4c/亚表型.pzfx` file
   - Import analysis result data
   - Generate statistical charts

**Statistical Methods**:

- Cox proportional hazards model
- Survival analysis
- Risk stratification

**Output**: Cox regression result plots and statistical reports

**Software Requirements**: GraphPad Prism (https://www.graphpad.com/features)

### Figure 4b: Radar Chart Analysis

**Objective**: Generate radar charts for multi-dimensional features

**Data Files**:

- `4b/data/雷达图_环状注释11.24.R`: R script data

**Execution Steps**:

1. Execute radar chart script using R language
2. Generate circular annotation radar chart

**Output**: Radar chart PDF files and videos

### Figure 4d: Risk Ratio Analysis

**Objective**: Calculate and visualize risk ratios for different factors

**Primary Files**:

- `4d/风险比.ipynb`: Risk ratio calculation code

**Execution Steps**:

1. Calculate risk ratios for various factors
2. Generate risk ratio forest plot:
   ```bash
   cd 4d
   jupyter notebook 风险比.ipynb
   ```

**Output**: Risk ratio forest plot and statistical reports

## 📊 Result Files

Each figure folder contains the following types of output files:

- **Video Files** (`video/`): Dynamic visualization results
- **Log Files** (`log/`): Execution logs and intermediate results
- **Data Files** (`data/`): Raw and preprocessed data

## 🔍 Important Notes

1. **Data Paths**: Ensure all data file paths are correct
2. **Dependency Versions**: Recommend using specified dependency versions
3. **Computational Resources**: Some analyses (e.g., SHAP) require substantial computational resources
4. **Memory Requirements**: Large datasets may require sufficient memory

## 📞 Technical Support

If encountering issues during execution, please check:

1. Python environment and dependency package versions
2. Data file paths and formats
3. Error messages in execution logs


## Figure Legends
Figure 2ab, ROC curves illustrating the prognostic prediction ability of SepsisFormer using 36 sepsis predictors is superior to the based models on the dataset MIMIC-III (p<0.01, except SepsisFormer vs. LSTM p=0.08,) and MIMIC-IV (p<0.01, except SepsisFormer vs. GRU p=0.41). All p-values by DeLong's test. 
Figure 2c, Network diagram displaying correlations among predictors, where a line connection indicates a statistically significant difference among predictors (p<0.05, Pearson’s correlation coefficients with two-sided significance testing). 
Figure 2d, ROC curves of prognostic prediction performance based on coagulation-inflammatory markers from local ICU data, comparing MMID-SMOTE with various domain-adaptive approaches and a no-adaptation baseline. 
Figure 2f, Cluster-informed EHR variable-level explainability. The chord diagram and the radar plot illustrate the varying contributions among the five categories. 
Figure 2g, Model-level explainability. The SHAP value-based contributions of the coagulation-inflammatory predictors to SepsisFormer reflect the importance of the features. The decision plot visualizes the contribution of coagulation-inflammatory predictors to individual patient predictions. Each line represents a patient, showing the cumulative SHAP values of each predictor from the bottom to the top of the plot. The Sankey diagram shows the cumulative overlap ordering of 8 predictors in the 1~8 transformer neural network. 
Figure  3c, Violin diagrams showing the significant difference in the distributions of coagulation-inflammatory markers between CIS1 and CIS2, as determined by Mann-Whitney U tests. 
Figure 3d, ROC curves illustrating the SMART and established scoring systems for the local ICU. 
Figure 3h, Violin diagrams illustrating the difference in distribution of coagulation-inflammatory markers among mild, moderate, severe, and dangerous risk levels, as determined by Mann–Whitney U tests. 
Figure 4a, c, Kaplan‒Meier survival curves illustrating the cumulative probability of 28-day survival for patient subgroups with distinct subphenotypes (CIS1 and CIS2) or risk levels (mild, moderate, severe, and dangerous) in the heparin treatment and control subgroups. Log-rank tests for survival analysis. 
Figure 4b, Radar plot highlighting different degrees of severity across five categories among the four risk levels in the heparin treatment and control subgroups. 
Figure 4d, Simplified diagram demonstrating the division of septic patients into eight subgroups of subphenotypes combined with risk stratification (CIS1_Mild, CIS1_Moderate, CIS1_Severe, CIS1_Dangerous, CIS2_Mild, CIS2_Moderate, CIS2_Severe, and CIS2_Dangerous). The HR values (95% CI) and p values (Log-rank tests) for the subphenotypes of patients illustrate the differences in the benefits of heparin treatment among these subgroups.

