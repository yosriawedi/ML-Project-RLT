# 🌲 RLT Extra Trees: Étude Comparative Complète
## Reinforcement Learning Trees - Analyse Multi-Modèles

**Author:** Dhia Romdhane  
**Date:** December 2025  
**Repository:** https://github.com/yosriawedi/ML-Project-RLT

---

## 🎯 Objectif

Comparer **RLT-ExtraTrees** (Reinforcement Learning Trees) contre 7 autres modèles de référence sur n'importe quel dataset uploadé.

### Modèles Comparés (8 au total):

1. **RLT-ExtraTrees** - RLT avec Variable Importance + Muting
2. **RF** - Random Forest classique  
3. **RF-√p** - Random Forest avec mtry = √p
4. **RF-log(p)** - Random Forest avec mtry = log(p)
5. **ExtraTrees** - Extra Trees standard
6. **BART/AdaBoost** - Bayesian/Adaptive Boosting
7. **LASSO** - Régression LASSO
8. **XGBoost** - Gradient Boosting

---

## 📊 Méthodologie

### CRISP-DM (Cross-Industry Standard Process for Data Mining)

1. **Business Understanding** - Définition du problème
2. **Data Understanding** - Analyse exploratoire (EDA)
3. **Data Preparation** - Preprocessing + Feature Engineering
4. **Modeling** - Entraînement des 8 modèles
5. **Evaluation** - Comparaison analytique
6. **Deployment** - Sauvegarde des résultats

### Hyperparamètres FIXES

Tous les modèles utilisent les **mêmes hyperparamètres** (fixés avant modélisation) pour une comparaison équitable:

```python
# Configuration globale
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Tree-based models
n_estimators = 100
max_depth = None
min_samples_split = 2
min_samples_leaf = 1

# RLT
VI_THRESHOLD = 0.01
VI_EXTRA_TREES_WEIGHT = 0.5
VI_STAT_WEIGHT = 0.5
```

---

## 🚀 Comment Utiliser

### Option 1: Google Colab (RECOMMANDÉ)

1. **Ouvrir Google Colab**: https://colab.research.google.com/

2. **Créer un nouveau notebook**

3. **Copier le contenu de `RLT_Complete_Analysis.py`** dans Colab

4. **Découper en cellules**:
   - Chercher les lignes avec `# ===... CELLULE X`
   - Créer une nouvelle cellule pour chaque section

5. **Exécuter cellule par cellule** (Shift+Enter)

6. **Upload votre dataset CSV** quand demandé (Cellule 4)

7. **Attendre les résultats** et visualisations

### Option 2: Utiliser le fichier Python directement

```python
# Dans Colab, créez une cellule et exécutez:
!wget https://raw.githubusercontent.com/yosriawedi/ML-Project-RLT/main/RLT_Complete_Analysis.py

# Puis copiez-collez le contenu dans des cellules
```

---

## 📁 Format de Dataset Attendu

### Structure du CSV:

```
feature1, feature2, feature3, ..., target
1.2,      3.4,      5.6,      ..., 0
2.3,      4.5,      6.7,      ..., 1
...
```

### Règles:

- ✅ **Format**: CSV avec header
- ✅ **Dernière colonne**: Target (variable à prédire)
- ✅ **Autres colonnes**: Features
- ✅ **Valeurs manquantes**: Acceptées (seront traitées automatiquement)
- ✅ **Variables catégorielles**: Acceptées (seront encodées)

### Exemples de datasets compatibles:

- Iris
- Boston Housing
- Breast Cancer
- Wine Quality
- Diabetes
- Titanic
- N'importe quel dataset classification/régression!

---

## 📊 Ce Que Vous Obtenez

### 1. Data Understanding (EDA)

- Statistiques descriptives
- Distribution du target
- Valeurs manquantes
- Doublons
- Matrice de corrélation
- Distribution des features

### 2. Data Preparation

- Nettoyage automatique
- Encoding catégorielles
- Scaling (StandardScaler)
- Split train/test (80/20)

### 3. RLT Variable Importance

- Calcul VI avec Extra Trees (50%) + Tests Statistiques (50%)
- Ranking de toutes les features
- Variable Muting (seuil = 0.01)
- Visualisation des top features

### 4. Modélisation

- Entraînement de 8 modèles
- Métriques pour chaque modèle:
  - **Classification**: Accuracy, Precision, Recall, F1-Score
  - **Régression**: R², RMSE, MAE
- Temps d'entraînement

### 5. Comparaison Analytique

- Tableau complet des résultats
- Ranking des modèles
- Visualisations comparatives
- Analyse de la performance de RLT
- Conclusion avec recommandations

### 6. Résultats Sauvegardés

- Fichier CSV avec tous les résultats
- Téléchargement automatique

---

## 🎯 Exemple de Sortie

```
=============================================================================
📊 COMPARAISON ANALYTIQUE DES RÉSULTATS
=============================================================================

📋 Tableau Complet des Résultats:

Model              Features  Train_Accuracy  Test_Accuracy  Precision  Recall  F1_Score  Train_Time
-----------------  --------  --------------  -------------  ---------  ------  --------  ----------
1. RLT-ExtraTrees  15        0.9876          0.9543         0.9534     0.9543  0.9538    2.34
2. RF              25        0.9923          0.9487         0.9481     0.9487  0.9484    3.12
3. RF-√p           25        0.9845          0.9456         0.9449     0.9456  0.9452    2.98
4. RF-log(p)       25        0.9834          0.9423         0.9418     0.9423  0.9420    2.87
5. ExtraTrees      25        0.9912          0.9398         0.9392     0.9398  0.9395    3.45
6. LASSO           25        0.8567          0.8234         0.8229     0.8234  0.8231    0.45
7. XGBoost         25        0.9901          0.9512         0.9507     0.9512  0.9509    4.23
8. AdaBoost        25        0.9234          0.8987         0.8982     0.8987  0.8984    2.67

🏆 MEILLEUR MODÈLE:
   - Nom: 1. RLT-ExtraTrees
   - Test Accuracy: 0.9543
   - Features: 15 (40% réduction!)
   - Temps: 2.34s

🌲 RLT-ExtraTrees:
   - Position: #1 / 8
   - Test Accuracy: 0.9543

🔍 ANALYSE RLT:
   ✅ RLT est MEILLEUR que les autres modèles
   📈 Amélioration: +0.59%
   🚀 Avec 40% moins de features!

💡 CONCLUSION:
   RLT-ExtraTrees obtient les meilleures performances avec 0.9543
   et utilise seulement 15/25 features (60% des features originales)
```

---

## 📈 Interprétation des Résultats

### Si RLT Gagne:

✅ **RLT est efficace** pour ce dataset  
✅ **Variable Importance** a bien identifié les features importantes  
✅ **Variable Muting** a éliminé le bruit sans perdre d'information  
✅ **Réduction de features** = Modèle plus rapide et interprétable

### Si RLT Perd:

⚠️ **Toutes les features sont importantes** - pas de bruit à éliminer  
⚠️ **Dataset trop petit** - VI pas assez fiable  
⚠️ **Features faiblement corrélées** - Muting trop agressif  

→ Essayez d'ajuster `VI_THRESHOLD` (actuellement 0.01)

---

## ⚙️ Personnalisation

Vous pouvez modifier les hyperparamètres dans **CELLULE 3**:

```python
# Changer le seuil de muting
VI_THRESHOLD = 0.01  # Plus bas = garde plus de features

# Changer les poids de VI
VI_EXTRA_TREES_WEIGHT = 0.5  # 0 à 1
VI_STAT_WEIGHT = 0.5          # 0 à 1 (total = 1)

# Changer le nombre d'arbres
TREE_CONFIG['n_estimators'] = 100  # Plus = mieux mais plus lent

# Changer le test size
TEST_SIZE = 0.2  # 20% test, 80% train
```

---

## 🔧 Dépendances

Toutes installées automatiquement dans Colab:

```python
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
scipy
```

---

## 📚 Références

1. **Zhu, R., Zeng, D., & Kosorok, M. R. (2015)**  
   "Reinforcement Learning Trees"  
   *Journal of the American Statistical Association*, 110(512), 1770-1784.

2. **Breiman, L. (2001)**  
   "Random Forests"  
   *Machine Learning*, 45(1), 5-32.

3. **CRISP-DM (2000)**  
   "Cross-Industry Standard Process for Data Mining"

---

## 📞 Contact

**Author:** Dhia Romdhane  
**Repository:** https://github.com/yosriawedi/ML-Project-RLT  
**Issues:** https://github.com/yosriawedi/ML-Project-RLT/issues

---

## 📝 License

Ce projet est à usage académique.

---

## 🎉 Changelog

### Version 1.0 (December 2025)
- ✅ Upload de dataset CSV
- ✅ Data Understanding (CRISP-DM)
- ✅ Data Preparation (CRISP-DM)
- ✅ RLT Variable Importance (Extra Trees + Statistical)
- ✅ 8 modèles comparés
- ✅ Hyperparamètres fixes
- ✅ Comparaison analytique complète
- ✅ Visualisations
- ✅ Sauvegarde résultats CSV

---

**🚀 Prêt à commencer? Uploadez votre dataset et lancez l'analyse!**
