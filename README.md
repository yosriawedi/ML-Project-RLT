# Complete CRISP-DM ML Pipeline with RLT Methodology

**A comprehensive machine learning project implementing Reinforcement Learning Trees (RLT) across 8 datasets**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production--Ready-green.svg)]()

---

## 🎯 Project Overview

This project implements a **complete CRISP-DM methodology** workflow integrating **Reinforcement Learning Trees (RLT)** concepts from [Zhu et al. (2015)](). The pipeline demonstrates systematic data science best practices from business understanding to deployment, with a focus on comparing RLT-style models against classical baselines.

### Key Achievements
- ✅ **8 datasets** analyzed end-to-end (classification & regression)
- ✅ **50% RLT win rate** (4/8 datasets improved)
- ✅ **22-41% feature reduction** on high-dimensional datasets
- ✅ **Production-ready pipeline** with save/load functionality
- ✅ **Comprehensive documentation** (130+ page report)

---

## 📁 Project Structure

```
(No subject)/
│
├── README.md                           # This file
├── CRISP_DM_REPORT.md                  # Complete 130-page report
├── requirements.txt                    # Python dependencies
├── pipeline_model.py                   # Production-ready ML pipeline
├── RLT_ML_Pipeline.ipynb              # Jupyter Notebook walkthrough
│
├── step1_business_understanding.py     # CRISP-DM Step 1
├── step2_data_understanding.py         # CRISP-DM Step 2 (EDA)
├── step3_data_preparation.py           # CRISP-DM Step 3 (RLT VI & Muting)
├── step4_modeling.py                   # CRISP-DM Step 4 (Baseline vs RLT)
├── step5_evaluation.py                 # CRISP-DM Step 5 (Evaluation)
│
├── prepared_data/                      # Preprocessed datasets
│   ├── *_X.npy                        # Muted features (RLT)
│   ├── *_X_full.npy                   # Full features (Baseline)
│   ├── *_y.npy                        # Targets
│   └── *_VI.csv                       # Variable importance scores
│
├── models/                             # Trained models
│   ├── *_best_model.pkl               # Best models per dataset
│   ├── *_results.csv                  # Per-dataset results
│   └── ALL_RESULTS.csv                # Consolidated results
│
├── evaluation/                         # Evaluation outputs
│   ├── *_confusion_matrix.png         # Confusion matrices
│   ├── *_roc_curve.png                # ROC curves
│   ├── *_residuals.png                # Residual plots
│   └── evaluation_results.csv         # Final metrics
│
└── visualizations/                     # EDA visualizations
    ├── *_target_distribution.png
    ├── *_correlation_heatmap.png
    └── *_feature_distributions.png
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository (or download files)
cd "C:\Users\DELL\Downloads\(No subject)"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Execute all CRISP-DM steps sequentially
python step1_business_understanding.py
python step2_data_understanding.py
python step3_data_preparation.py
python step4_modeling.py
python step5_evaluation.py
```

### 3. Use Production Pipeline

```python
from pipeline_model import RLTMLPipeline
import pandas as pd

# Initialize pipeline
pipeline = RLTMLPipeline(problem_type='classification', vi_threshold=0.01)

# Load and preprocess data
df = pd.read_csv('your_data.csv')
X, y = pipeline.preprocess(df, target_col='target', fit=True)

# Train with RLT variable muting
model = pipeline.train(X, y, apply_muting=True)

# Make predictions
predictions = pipeline.predict(X_new)

# Save model
pipeline.save_model('my_model.pkl')

# Load model
loaded_pipeline = RLTMLPipeline.load_model('my_model.pkl')
```

### 4. Explore Jupyter Notebook

```bash
jupyter notebook RLT_ML_Pipeline.ipynb
```

---

## 📊 Results Summary

### Baseline vs RLT Performance

| Dataset | Baseline Best | RLT Best | Improvement | Winner |
|---------|---------------|----------|-------------|--------|
| **BostonHousing** | 0.8847 | 0.8938 | +1.03% | **RLT 🏆** |
| WineQuality_Red | 0.5961 | 0.5961 | 0.00% | TIE |
| **WineQuality_White** | 0.5529 | 0.5544 | +0.27% | **RLT 🏆** |
| Sonar | 0.8561 | 0.8466 | -1.11% | BASELINE |
| **Parkinsons** | 0.9333 | 0.9385 | +0.55% | **RLT 🏆** |
| WDBC Breast Cancer | 0.9614 | 0.9579 | -0.36% | BASELINE |
| AutoMPG | 0.8712 | 0.8712 | 0.00% | TIE |
| **SchoolData** | 0.6850 | 0.7050 | +2.92% | **RLT 🏆** |

**Win Rate:** 50% (4 out of 8 datasets improved)

### RLT Feature Reduction

| Dataset | Original Features | Muted Features | Reduction % |
|---------|-------------------|----------------|-------------|
| Sonar | 60 | 18 | 30.0% |
| Parkinsons | 22 | 9 | 40.9% |
| WDBC | 30 | 12 | 40.0% |
| SchoolData | 35 | 10 | 28.6% |

---

## 🧠 RLT Methodology

### What is RLT?

**Reinforcement Learning Trees** (Zhu et al., 2015) is a tree-based method designed for high-dimensional sparse settings. Key innovations:

1. **Variable Importance-Driven Splitting**
   - Chooses variables with greatest future improvement
   - Not just immediate marginal effect

2. **Variable Muting**
   - Progressively eliminates noise variables
   - Prevents noise features at terminal nodes

3. **Sparsity Focus**
   - Assumes p₁ << p (few strong variables among many)
   - Addresses Random Forest weakness in high dimensions

### Implementation

Our implementation includes:
- ✅ **Global Variable Importance:** Ensemble-based (RF + ET + Statistical tests)
- ✅ **Variable Muting:** Threshold-based feature elimination
- ⚠️ **Partial Look-Ahead:** Basic implementation (full version in future work)
- ⚠️ **Linear Combination Splits:** Not yet implemented

---

## 📈 When to Use RLT

### ✅ RLT Excels When:
- High-dimensional datasets (p > 20)
- Sparse signal structure (few strong variables)
- Presence of noise/irrelevant features
- Multicollinear features

### ⚠️ RLT Underperforms When:
- Low-dimensional datasets (p < 10)
- All features carry signal
- Small sample sizes (n < 200)
- Strong feature interactions required

---

## 🎓 Datasets Used

| # | Dataset | Type | Samples | Features | Domain |
|---|---------|------|---------|----------|--------|
| 1 | BostonHousing | Regression | 506 | 13 | Real Estate |
| 2 | WineQuality_Red | Classification | 1,599 | 11 | Food & Beverage |
| 3 | WineQuality_White | Classification | 4,898 | 11 | Food & Beverage |
| 4 | Sonar | Binary Classification | 208 | 60 | Signal Processing |
| 5 | Parkinsons | Binary Classification | 195 | 22 | Healthcare |
| 6 | WDBC | Binary Classification | 569 | 30 | Healthcare/Oncology |
| 7 | AutoMPG | Regression | 398 | 7 | Automotive |
| 8 | SchoolData | Classification | 200 | 36 | Education |

---

## 🔧 Technical Stack

- **Python:** 3.10+
- **ML Framework:** scikit-learn 1.2+
- **Gradient Boosting:** XGBoost 1.7+
- **Visualization:** Matplotlib, Seaborn
- **Notebook:** Jupyter
- **Document Processing:** python-docx, PyPDF2, pdfplumber

---

## 📚 Documentation

### Main Documents
- **[CRISP_DM_REPORT.md](CRISP_DM_REPORT.md):** Complete 130-page report
  - Business understanding
  - Data exploration (EDA)
  - Preprocessing & RLT variable muting
  - Modeling results (baseline vs RLT)
  - Evaluation metrics & visualizations
  - Deployment recommendations

- **[RLT_ML_Pipeline.ipynb](RLT_ML_Pipeline.ipynb):** Interactive notebook
  - Step-by-step walkthrough
  - Visualizations
  - Code examples
  - Results analysis

### API Documentation

#### `RLTMLPipeline` Class

```python
class RLTMLPipeline:
    """Production ML Pipeline with RLT methodology"""
    
    def __init__(self, problem_type='classification', vi_threshold=0.01):
        """Initialize pipeline"""
        
    def preprocess(self, df, target_col, fit=True):
        """Preprocess data: missing values, encoding, scaling"""
        
    def compute_variable_importance(self, X, y):
        """Compute RLT-style variable importance"""
        
    def apply_variable_muting(self, X):
        """Apply RLT variable muting (feature selection)"""
        
    def train(self, X, y, apply_muting=True):
        """Train model with optional RLT muting"""
        
    def predict(self, X):
        """Make predictions on new data"""
        
    def save_model(self, filepath):
        """Save trained pipeline"""
        
    @staticmethod
    def load_model(filepath):
        """Load trained pipeline"""
```

---

## 🚀 Deployment

### Production Checklist

- [x] Model training complete
- [x] Cross-validation performed
- [x] Test set evaluation done
- [x] Feature importance documented
- [x] Pipeline save/load tested
- [x] Error handling implemented
- [ ] REST API endpoint (future work)
- [ ] Docker containerization (future work)
- [ ] Monitoring dashboard (future work)

### Deployment-Ready Models

**HIGH Priority (Ready for Production):**
- ✅ **Parkinsons:** 94.9% accuracy, ROC-AUC=0.971
- ✅ **WDBC Breast Cancer:** 96.5% accuracy, ROC-AUC=0.994
- ✅ **BostonHousing:** R²=0.904, MAPE=10.3%

**MEDIUM Priority (Pilot Testing):**
- ⚠️ **SchoolData:** 72.5% accuracy (+2.92% with RLT)
- ⚠️ **AutoMPG:** R²=0.873, MAPE=8.3%

**LOW Priority (Needs Improvement):**
- ❌ **Wine Quality:** 55-60% accuracy (collect more data)
- ❌ **Sonar:** RLT underperformed (revisit features)

---

## 🔮 Future Work

### Short-Term (1-3 months)
- [ ] Implement full look-ahead behavior
- [ ] Test linear combination splits
- [ ] Adaptive muting thresholds
- [ ] Ensemble stacking (baseline + RLT)

### Medium-Term (3-6 months)
- [ ] REST API deployment (Flask/FastAPI)
- [ ] Docker containerization
- [ ] MLOps pipeline (monitoring, retraining)
- [ ] Feature engineering enhancements

### Long-Term (6-12 months)
- [ ] Custom tree implementation (full RLT)
- [ ] Benchmark on 50+ datasets
- [ ] Comparison to AutoML frameworks
- [ ] Medical AI validation (FDA compliance)

---

## 📖 References

1. **Zhu et al. (2015):** "Reinforcement Learning Trees" - Original RLT paper
2. **Breiman (2001):** "Random Forests" - Machine Learning 45(1), 5-32
3. **Chapman et al. (2000):** "CRISP-DM 1.0 Step-by-step data mining guide"
4. **scikit-learn Documentation:** https://scikit-learn.org

---

## 👥 Contributing

Contributions are welcome! Areas for contribution:
- Full RLT implementation (look-ahead, linear splits)
- Additional datasets and benchmarks
- Hyperparameter optimization
- Documentation improvements
- Bug fixes and testing

---

## 📧 Contact

**Project:** CRISP-DM ML Pipeline with RLT  
**Authors:** Dhia Romdhane, Yosri Awedi, Baha Saadoui, Nour Rajhi, Bouguerra Taha, Oumaima Nacef  
**Course:** Machine Learning Project  
**Date:** December 2025  
**Status:** ✅ Production-Ready

---

## 📄 License

This project is for educational and research purposes.

---

## 🎉 Acknowledgments

I would like to thank:
- **Zhu et al.** for the original RLT methodology that inspired this work
- **Breiman et al.** for their groundbreaking Random Forests algorithm
- **scikit-learn community** for providing excellent ML tools
- **UCI Machine Learning Repository** for making quality datasets available

---

**⭐ If you find this project useful, please star the repository!**
