# DeiT — PyTorch Implementation

### Data-efficient Image Transformers & Knowledge Distillation

Implémentation pédagogique de **DeiT-Tiny Distilled (Data-efficient Image Transformer)** avec **Python et PyTorch**, réalisée dans le cadre d'un projet universitaire de Machine Learning / Deep Learning.

Ce projet s'appuie sur l'article scientifique :

**Training data-efficient image transformers & distillation through attention** — Hugo Touvron et al., ICML 2021.

[Consulter l'article scientifique](https://arxiv.org/abs/2012.12877)

> **Important :** ce projet implémente les mécanismes essentiels de DeiT-Tiny Distilled à des fins pédagogiques. Il ne reproduit ni le protocole complet d'entraînement ni les performances rapportées dans l'article.

---

## 1. Objectif du projet

L'objectif est de comprendre et d'illustrer :

- Le découpage des images en patches et le **Patch Embedding**.
- Les **CLS token**, **Distillation token** et **Position Embedding**.
- La **Multi-Head Self-Attention** et les blocs Transformer.
- Les connexions résiduelles, la normalisation, le MLP et le **Stochastic Depth (DropPath)**.
- La distillation de connaissances **Teacher–Student**.
- Les fonctions de perte de classification et de distillation (**Hard** et **Soft**).
- L'utilisation de `torch.nn.Module` et de la méthode `forward()`.
- La rétropropagation, l'optimisation et la notion de **batch** et d'**epoch**.

## 2. Architecture du Student : DeiT-Tiny Distilled

Le Student est défini par la classe `DeiTTinyDistilled(nn.Module)` dans `deit_model.py`.

| Paramètre | Valeur |
|---|---|
| Taille des images | 224 × 224 pixels |
| Canaux | 3 (RGB) |
| Taille des patches | 16 × 16 pixels |
| Nombre de patches | 196 |
| Dimension des embeddings | 192 |
| Nombre de têtes d'attention | 3 |
| Blocs Transformer | 12 |
| Tokens supplémentaires | CLS et DIST |
| Nombre total de tokens | 198 |
| Classes de sortie | 1 000 |
| Têtes de classification | `head` (CLS) et `head_dist` (DIST) |
| Nombre de paramètres du Student | 5 910 800 |

### Fonctionnement

```text
Image RGB (3 × 224 × 224)
           |
           v
Patch Embedding (Conv2d, patch 16 × 16)
           |
           v
196 Patch Tokens (dimension 192)
           |
           v
Ajout CLS Token + DIST Token
           |
           v
Position Embedding (198 tokens)
           |
           v
Transformer Encoder (12 blocs)
   - Multi-Head Self-Attention
   - LayerNorm et connexions résiduelles
   - MLP et DropPath
           |
           v
       LayerNorm
           |
      +----+----+
      |         |
      v         v
   CLS Head  DIST Head
      |         |
      v         v
 (B, 1000)  (B, 1000)
```

Le Student renvoie deux ensembles de **logits** : `cls_logits` et `dist_logits`, chacun de forme `(B, 1000)`, où `B` est la taille du batch.

## 3. Knowledge Distillation

DeiT utilise un apprentissage **Teacher–Student** :

- **Teacher :** `TeacherModel(nn.Module)`, basé sur **RegNetY-16GF**, avec les poids pré-entraînés `IMAGENET1K_V2` de `torchvision`. Ses paramètres sont gelés et il reste en mode évaluation.
- **Student :** `DeiTTinyDistilled(nn.Module)`, un Vision Transformer qui apprend à partir des étiquettes fournies et des prédictions du Teacher.

La fonction `DistillationLoss(nn.Module)` implémente deux variantes :

- **Hard Distillation :** la tête DIST apprend à prédire la classe choisie par le Teacher (`argmax`).
- **Soft Distillation :** la tête DIST apprend à rapprocher sa distribution de probabilités de celle du Teacher, à l'aide de la divergence KL et d'une température `tau`.

La fonction objective est :

```text
L_total = (1 - alpha) × L_CLS + alpha × L_DIST
```

Dans le test exécuté, `distillation_type="hard"` et `alpha=0.5` :

```text
L_total = 0.5 × L_CLS + 0.5 × L_DIST
```

La perte de classification et la perte de Hard Distillation utilisent `CrossEntropyLoss`.

## 4. Structure du projet

```text
projet_ml/
├── deit_model.py     # Architecture du Student DeiT-Tiny Distilled
├── train.py          # Teacher, distillation, optimisation, batches et epochs
├── requirements.txt  # Dépendances Python
├── README.md         # Documentation du projet
└── .gitignore        # Fichiers ignorés par Git (si présent)
```

| Fichier | Description |
|---|---|
| `deit_model.py` | Patch Embedding, tokens, attention, MLP, blocs Transformer, DropPath et Student |
| `train.py` | Teacher RegNetY-16GF, fonction de perte, test de rétropropagation et boucle pédagogique |
| `requirements.txt` | Dépendances nécessaires |
| `README.md` | Présentation et instructions d'exécution |

## 5. Installation

### Cloner le dépôt

```bash
git clone https://github.com/YaiciAnis/DeiT-PyTorch-Implementation.git
cd DeiT-PyTorch-Implementation
```

> Si le dépôt GitHub porte un autre nom ou si le projet n'y a pas encore été publié, adaptez cette commande à l'URL réelle du dépôt.

### Créer et activer un environnement virtuel

**Windows (PowerShell) :**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / WSL2 :**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Installer les dépendances

```bash
python -m pip install -r requirements.txt
```

La première exécution de `train.py` peut télécharger automatiquement les poids pré-entraînés du Teacher si ceux-ci ne sont pas déjà en cache. Le modèle Teacher est volumineux.

## 6. Exécution

### Tester l'architecture du Student

```bash
python deit_model.py
```

Ce test vérifie notamment les dimensions des tenseurs, les 12 blocs Transformer, le nombre de paramètres et le fonctionnement de DropPath.

### Tester la distillation et la boucle d'apprentissage

```bash
python train.py
```

Le script effectue :

1. Un **test réel sur un batch fictif d'une image** : propagation avant du Student et du Teacher, calcul de la perte, `backward()` et `optimizer.step()`.
2. Une vérification : les poids du Student changent, tandis que ceux du Teacher restent inchangés.
3. Une **démonstration sans entraînement** de deux epochs et de quatre batches au total.

Extrait d'une exécution validée sur CPU :

```text
Appareil utilisé : cpu
Images : torch.Size([1, 3, 224, 224])
CLS : torch.Size([1, 1000])
DIST : torch.Size([1, 1000])
Teacher : torch.Size([1, 1000])

===== HARD DISTILLATION =====
Loss CLS : 6.761744022369385
Loss DIST : 5.48687219619751
Loss totale : 6.124308109283447

===== RÉTROPROPAGATION =====
Gradients CLS : calculés
Gradients DIST : calculés
Gradients Teacher : aucun

===== VÉRIFICATION DES PARAMÈTRES =====
Poids Student modifiés : True
Poids Teacher inchangés : True

Tests des étapes 9 à 11 réussis !

===== ÉTAPE 12 : BATCHES ET EPOCHS =====
Nombre d'images : 8
Taille d'un batch : 4
Nombre de batches : 2

========== EPOCH 1/2 ==========
Batch 1/2 | Images : (4, 3, 224, 224) | Labels : (4,)
Batch 2/2 | Images : (4, 3, 224, 224) | Labels : (4,)

========== EPOCH 2/2 ==========
Batch 1/2 | Images : (4, 3, 224, 224) | Labels : (4,)
Batch 2/2 | Images : (4, 3, 224, 224) | Labels : (4,)

Test de l'étape 12 réussi !
```

Les valeurs de perte peuvent varier selon l'environnement. Les lignes des epochs ci-dessus sont abrégées : le programme affiche également les six opérations prévues pour chaque batch.

## 7. Batches, epochs et optimisation

Le script distingue **une itération réellement exécutée** et **une démonstration de la structure de l'entraînement**.

| Élément | Configuration |
|---|---|
| Données | Images et étiquettes fictives générées aléatoirement |
| Dimensions des images | 3 × 224 × 224 |
| Nombre de classes | 1 000 |
| Test réel de rétropropagation | 1 image, 1 itération |
| Dataset de démonstration | 8 images |
| Batch size de démonstration | 4 |
| Batches par epoch | 2 |
| Nombre d'epochs de démonstration | 2 |
| Itérations affichées | 4 (non exécutées) |
| Optimiseur | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.05 |
| Fonction objective utilisée | Cross-Entropy + Hard Distillation |

### Étapes d'une itération d'apprentissage

1. Charger un batch d'images et ses étiquettes.
2. Calculer `cls_logits` et `dist_logits` avec le Student.
3. Calculer `teacher_logits` sans gradients.
4. Calculer la perte totale.
5. Réinitialiser les gradients avec `optimizer.zero_grad()`.
6. Effectuer la rétropropagation avec `loss_total.backward()`.
7. Mettre à jour les paramètres du Student avec `optimizer.step()`.
8. Répéter pour les batches et les epochs suivants.

La fonction `boucle_apprentissage(...)` propose deux modes :

- `demonstration=True` (**utilisé par défaut**) : affiche le déroulement des batches et des epochs sans exécuter les calculs d'apprentissage.
- `demonstration=False` : exécute la boucle PyTorch, avec calcul des pertes, gradients et mises à jour du Student. Ce mode n'est pas nécessaire pour le rendu demandé.

**Aucun dataset réel n'est téléchargé ni utilisé pour l'entraînement du Student.** Les pertes affichées ne constituent pas une mesure de performance ou de précision du modèle.

## 8. Technologies

- Python 3
- PyTorch (`torch`)
- Torchvision (`torchvision`)
- NumPy (`numpy`, déclaré dans les dépendances)
- VS Code
- Windows / PowerShell ou Linux / WSL2
- Git et GitHub

## 9. Cadre pédagogique

Le projet répond aux objectifs de présentation suivants :

- Présenter l'architecture du modèle et la technique de distillation.
- Présenter les classes PyTorch héritant de `nn.Module`.
- Expliquer la fonction objective.
- Montrer le fonctionnement de `forward()`, `backward()` et de l'optimiseur.
- Illustrer la différence entre un **batch**, une **itération** et une **epoch**.

Il ne prétend pas reproduire les performances expérimentales de l'article original.
