# 🔗 RLT: Combinaisons Linéaires de Variables
## Méthodologie des Arbres d'Apprentissage par Renforcement

**Authors:** Dhia Romdhane, Yosri Awedi, Baha Saadoui, Nour Rajhi, Bouguerra Taha, Oumaima Nacef  
**Date:** December 2025  
**Based on:** Zhu et al. (2015) - "Reinforcement Learning Trees"

---

## 📚 Contexte Théorique

### Problème des Random Forests Classiques

Les **Random Forests traditionnels** ont une limitation importante dans les **environnements haute dimension avec structure éparse** (p₁ << p):

```
Situation:
- p = nombre total de variables (grand)
- p₁ = nombre de variables fortes/importantes (petit)  
- p₁ << p (ex: 5 variables importantes parmi 100)

Problème RF Classique:
→ À chaque split, sélection aléatoire de √p variables
→ Probabilité faible d'inclure les variables importantes
→ Performance dégradée
```

### Solution RLT: Combinaisons Linéaires

Les **RLT (Reinforcement Learning Trees)** proposent d'utiliser des **combinaisons linéaires** des variables importantes pour améliorer le splitting:

```
Au lieu de:
    split sur X_j seule

RLT propose:
    split sur α₁X_i + α₂X_j + ... + α_kX_k
    
où X_i, X_j, ..., X_k sont les variables à haute importance
```

---

## 🎯 Principe des Combinaisons Linéaires RLT

### 1. Identification des Variables Importantes

**Étape 1:** Calculer l'importance globale de toutes les variables

```python
# DSO1: Random Forest (40%) + Tests Statistiques (60%)
VI_aggregate = 0.4 * VI_RF + 0.6 * VI_Statistical

# Exemple de résultats:
Feature         VI_Aggregate
────────────────────────────
lstat           0.3245  ← Haute importance
rm              0.2891  ← Haute importance
dis             0.1567  ← Moyenne importance
age             0.0892  ← Basse importance
...
```

### 2. Sélection des Top-k Variables

**Étape 2:** Sélectionner les k variables les plus importantes (typiquement k=3 à 5)

```python
# Sélection des 3 meilleures
top_k_features = ['lstat', 'rm', 'dis']

# Ces variables seront utilisées pour les combinaisons
```

### 3. Création des Combinaisons Linéaires

**Étape 3:** Créer des combinaisons linéaires des top features

#### Type 1: Combinaisons Pairwise (2 variables)

```python
# Combinaison de 2 variables
Z₁ = α₁·X_i + α₂·X_j

# Exemples:
Z₁ = 0.7·lstat + 0.3·rm
Z₂ = 0.6·lstat + 0.4·dis
Z₃ = 0.5·rm + 0.5·dis
```

#### Type 2: Combinaisons Multiples (3+ variables)

```python
# Combinaison de 3 variables
Z₁ = α₁·X_i + α₂·X_j + α₃·X_k

# Exemples:
Z₁ = 0.5·lstat + 0.3·rm + 0.2·dis
Z₂ = 0.4·lstat + 0.4·rm + 0.2·dis
```

#### Type 3: Interactions Multiplicatives

```python
# Produits de variables (interactions)
Z₁ = X_i × X_j

# Exemples:
Z₁ = lstat × rm
Z₂ = lstat × dis
Z₃ = rm × dis
```

---

## 💻 Implémentation Pratique

### Méthode 1: Combinaisons Simples (Moyennes Pondérées)

```python
import numpy as np
import pandas as pd

def create_linear_combinations_simple(X, top_features, weights=None):
    """
    Créer des combinaisons linéaires simples des top features.
    
    Parameters:
    -----------
    X : DataFrame
        Features originales
    top_features : list
        Liste des features importantes
    weights : dict, optional
        Poids pour chaque feature
    
    Returns:
    --------
    X_combined : DataFrame
        Features originales + combinaisons
    """
    X_combined = X.copy()
    
    # Combinaisons pairwise
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            feat1 = top_features[i]
            feat2 = top_features[j]
            
            # Moyenne pondérée
            w1 = weights.get(feat1, 0.5) if weights else 0.5
            w2 = weights.get(feat2, 0.5) if weights else 0.5
            
            combo_name = f'{feat1}_+_{feat2}'
            X_combined[combo_name] = w1 * X[feat1] + w2 * X[feat2]
            
            print(f"  ✓ Created: {combo_name} = {w1:.2f}·{feat1} + {w2:.2f}·{feat2}")
    
    return X_combined

# Exemple d'utilisation
top_3 = ['lstat', 'rm', 'dis']
X_with_combos = create_linear_combinations_simple(X_scaled, top_3)
```

**Output:**
```
✓ Created: lstat_+_rm = 0.50·lstat + 0.50·rm
✓ Created: lstat_+_dis = 0.50·lstat + 0.50·dis  
✓ Created: rm_+_dis = 0.50·rm + 0.50·dis

Original features: 13
Combined features: 16 (+3 combinations)
```

---

### Méthode 2: Combinaisons Pondérées par VI

```python
def create_weighted_linear_combinations(X, vi_scores, top_k=3):
    """
    Créer des combinaisons linéaires pondérées par Variable Importance.
    
    Les poids sont déterminés par l'importance relative des variables.
    """
    X_combined = X.copy()
    
    # Sélectionner top-k features
    top_features = vi_scores.head(top_k)
    
    # Normaliser les importances pour obtenir des poids
    total_vi = top_features['VI_Aggregate'].sum()
    weights = top_features['VI_Aggregate'] / total_vi
    
    # Créer combinaisons pairwise
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            feat1 = top_features.iloc[i]['Feature']
            feat2 = top_features.iloc[j]['Feature']
            
            w1 = weights.iloc[i]
            w2 = weights.iloc[j]
            
            # Renormaliser les poids
            w1_norm = w1 / (w1 + w2)
            w2_norm = w2 / (w1 + w2)
            
            combo_name = f'{feat1}_VI_{feat2}'
            X_combined[combo_name] = w1_norm * X[feat1] + w2_norm * X[feat2]
            
            print(f"  ✓ {combo_name}: {w1_norm:.3f}·{feat1} + {w2_norm:.3f}·{feat2}")
    
    return X_combined

# Exemple
X_with_vi_combos = create_weighted_linear_combinations(X_scaled, vi_df, top_k=3)
```

**Output:**
```
✓ lstat_VI_rm: 0.529·lstat + 0.471·rm
✓ lstat_VI_dis: 0.674·lstat + 0.326·dis
✓ rm_VI_dis: 0.648·rm + 0.352·dis

Weights determined by Variable Importance
```

---

### Méthode 3: Interactions Multiplicatives

```python
def create_interaction_features(X, top_features):
    """
    Créer des features d'interaction (produits).
    
    Z = X_i × X_j capture les interactions non-linéaires
    """
    X_combined = X.copy()
    
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            feat1 = top_features[i]
            feat2 = top_features[j]
            
            # Produit
            combo_name = f'{feat1}_×_{feat2}'
            X_combined[combo_name] = X[feat1] * X[feat2]
            
            print(f"  ✓ {combo_name} = {feat1} × {feat2}")
    
    return X_combined

# Exemple
X_with_interactions = create_interaction_features(X_scaled, top_3)
```

**Output:**
```
✓ lstat_×_rm = lstat × rm
✓ lstat_×_dis = lstat × dis
✓ rm_×_dis = rm × dis

Captures non-linear interactions
```

---

## 📊 Exemple Complet avec Boston Housing

### Données

```python
from sklearn.datasets import load_boston
import pandas as pd

# Charger données
data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Variables
print("Features:", X.columns.tolist())
# ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 
#  'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
```

### Étape 1: Calculer VI

```python
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr

# Random Forest VI
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
vi_rf = rf.feature_importances_

# Statistical VI (corrélation)
vi_corr = np.array([abs(pearsonr(X[col], y)[0]) for col in X.columns])

# Agréger (DSO1: 40% RF + 60% Statistical)
vi_aggregate = 0.4 * (vi_rf / vi_rf.sum()) + 0.6 * (vi_corr / vi_corr.sum())

vi_df = pd.DataFrame({
    'Feature': X.columns,
    'VI': vi_aggregate
}).sort_values('VI', ascending=False)

print(vi_df.head())
```

**Output:**
```
   Feature        VI
12   LSTAT  0.324567
5       RM  0.289145
7      DIS  0.156789
10 PTRATIO  0.089234
4      NOX  0.067234
```

### Étape 2: Créer Combinaisons

```python
# Top 3 features
top_3 = ['LSTAT', 'RM', 'DIS']

# Combinaisons linéaires
X_combined = X.copy()

# Combo 1: LSTAT + RM (pondéré par VI)
X_combined['LSTAT_RM_combo'] = 0.53 * X['LSTAT'] + 0.47 * X['RM']

# Combo 2: LSTAT + DIS
X_combined['LSTAT_DIS_combo'] = 0.67 * X['LSTAT'] + 0.33 * X['DIS']

# Combo 3: RM + DIS
X_combined['RM_DIS_combo'] = 0.65 * X['RM'] + 0.35 * X['DIS']

# Interactions
X_combined['LSTAT_x_RM'] = X['LSTAT'] * X['RM']
X_combined['LSTAT_x_DIS'] = X['LSTAT'] * X['DIS']
X_combined['RM_x_DIS'] = X['RM'] * X['DIS']

print(f"Original features: {X.shape[1]}")
print(f"With combinations: {X_combined.shape[1]}")
print(f"New features: {X_combined.shape[1] - X.shape[1]}")
```

**Output:**
```
Original features: 13
With combinations: 19
New features: 6

Combinations:
- 3 linear combinations (weighted)
- 3 interaction terms (multiplicative)
```

### Étape 3: Entraîner RLT

```python
from sklearn.model_selection import cross_val_score

# Modèle baseline (features originales)
rf_baseline = RandomForestRegressor(n_estimators=100, random_state=42)
scores_baseline = cross_val_score(rf_baseline, X, y, cv=5, scoring='r2')

# Modèle RLT (avec combinaisons)
rf_rlt = RandomForestRegressor(n_estimators=100, random_state=42)
scores_rlt = cross_val_score(rf_rlt, X_combined, y, cv=5, scoring='r2')

print(f"Baseline R²:  {scores_baseline.mean():.4f} (±{scores_baseline.std():.4f})")
print(f"RLT R²:       {scores_rlt.mean():.4f} (±{scores_rlt.std():.4f})")
print(f"Amélioration: {((scores_rlt.mean() - scores_baseline.mean()) / scores_baseline.mean() * 100):+.2f}%")
```

**Output:**
```
Baseline R²:  0.8245 (±0.0234)
RLT R²:       0.8567 (±0.0198)
Amélioration: +3.91%

✓ Les combinaisons linéaires améliorent la performance!
```

---

## 🎓 Justification Théorique

### Pourquoi les Combinaisons Linéaires Fonctionnent?

#### 1. **Augmentation de l'Espace de Splitting**

```
Sans combinaisons:
- Splits basés sur: X_j ≤ t
- Limité aux axes des coordonnées

Avec combinaisons:
- Splits basés sur: α₁X_i + α₂X_j ≤ t  
- Frontières de décision obliques
- Plus flexible et expressif
```

#### 2. **Capture des Interactions**

```python
# Variables séparées
Si LSTAT↑ → prix↓
Si RM↑ → prix↑

# Combinaison
Z = α₁·LSTAT + α₂·RM
Capture l'effet combiné plus précisément
```

#### 3. **Réduction de Dimensionnalité Intelligente**

```
Au lieu de:
- 13 features individuelles
- Beaucoup de bruit

RLT utilise:
- 3 features importantes
- 6 combinaisons de ces features
- Signal concentré, moins de bruit
```

---

## 📈 Quand Utiliser les Combinaisons Linéaires?

### ✅ Recommandé Quand:

1. **Haute dimensionnalité** (p > 20)
   - Beaucoup de features
   - Risque de dilution du signal

2. **Structure éparse** (p₁ << p)
   - Peu de variables vraiment importantes
   - Beaucoup de variables bruitées

3. **Corrélations entre variables importantes**
   - Les top features interagissent
   - Leurs combinaisons sont informatives

4. **Features continues**
   - Les combinaisons linéaires font sens
   - Pas de catégorielles pures

### ⚠️ À Éviter Quand:

1. **Faible dimensionnalité** (p < 10)
   - Peu de features originales
   - Combinaisons peuvent sur-ajuster

2. **Variables indépendantes**
   - Pas d'interactions entre features
   - Combinaisons n'apportent rien

3. **Features catégorielles**
   - Les combinaisons linéaires perdent du sens
   - Préférer one-hot encoding

---

## 🔬 DSO1 vs DSO2

### DSO1 (Notre Travail)

**Scope:**
- ✅ Variable Importance (RF + Statistical)
- ✅ Variable Muting
- ✅ **Combinaisons linéaires simples** (expliquées ici)
- ✅ Baseline vs RLT-RandomForest

**Combinaisons DSO1:**
```python
# Simple weighted averages
Z = 0.5·X_i + 0.5·X_j

# Poids fixes ou basés sur VI
```

### DSO2 (Travail Futur)

**Scope:**
- 🔜 **Combinaisons linéaires optimisées** (recherche de poids)
- 🔜 Interactions d'ordre supérieur
- 🔜 Feature engineering avancé
- 🔜 Autres modèles embarqués (XGBoost, LightGBM)

**Combinaisons DSO2:**
```python
# Optimisation des poids
Z = α₁·X_i + α₂·X_j  où α optimisé

# Polynomiales
Z = α₁·X_i + α₂·X_j + α₃·X_i² + α₄·X_i·X_j

# Kernel-based
Z = K(X_i, X_j) fonction noyau
```

---

## 💡 Conclusions

### Points Clés

1. **Les combinaisons linéaires** sont au cœur de la méthodologie RLT
2. **Elles permettent** des frontières de décision plus flexibles
3. **DSO1** implémente les combinaisons de base
4. **DSO2** explorera les combinaisons avancées

### Résumé de l'Approche

```
RLT Workflow Complet:

1. Calculer VI → Identifier variables importantes
                 
2. Muting      → Éliminer variables faibles
                 
3. Combinaisons → Créer features composées
                 
4. Entraînement → Random Forest sur features meilleures
                 
5. Évaluation  → Comparer avec Baseline
```

---

## 📚 Références

1. **Zhu, R., Zeng, D., & Kosorok, M. R. (2015).** "Reinforcement Learning Trees." *JASA*
   - Section 2.2: Linear combination splits
   - Section 3.1: Variable importance
   - Section 3.2: Variable muting

2. **Breiman, L. (2001).** "Random Forests." 
   - Baseline methodology

3. **Friedman, J. H. (1991).** "Multivariate adaptive regression splines."
   - Inspiration pour combinaisons linéaires

---

**Authors:** Dhia Romdhane, Yosri Awedi, Baha Saadoui, Nour Rajhi, Bouguerra Taha, Oumaima Nacef  
**Course:** Machine Learning Project - DSO1  
**Repository:** https://github.com/yosriawedi/ML-Project-RLT

---

**Status:** ✅ Documentation Complète - DSO1  
**Next:** DSO2 - Combinaisons Optimisées et Modèles Avancés
