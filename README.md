# DeiT — PyTorch Implementation

### Data-efficient Image Transformers & Knowledge Distillation

Implémentation pédagogique de **DeiT (Data-efficient Image Transformer)** avec **Python et PyTorch**, réalisée dans le cadre d'un projet universitaire de Deep Learning.

Ce projet s'appuie sur l'article scientifique :

**Training data-efficient image transformers & distillation through attention** — Hugo Touvron et al., ICML 2021.

[Consulter l'article scientifique](https://arxiv.org/abs/2012.12877)

---

## 1. Objectif du projet

L'objectif est de comprendre et d'illustrer les principaux mécanismes de DeiT :

- Architecture Vision Transformer (ViT).
- Découpage des images en patches.
- Patch Embedding et Position Embedding.
- Mécanisme de Self-Attention.
- CLS token et Distillation token.
- Distillation de connaissances Teacher–Student.
- Fonction de perte de classification et de distillation.
- Implémentation avec `torch.nn.Module`.
- Boucle d'apprentissage avec batches et epochs.

**Important :** cette implémentation est une version simplifiée de DeiT. Elle sert à expliquer le fonctionnement du modèle et ne vise pas à reproduire les performances de l'article original.

## 2. Architecture de MiniDeiT

Le modèle Student est implémenté dans la classe `MiniDeiT(nn.Module)`.

| Paramètre | Valeur |
|---|---|
| Taille des images | 32 × 32 pixels |
| Canaux | 3 (RGB) |
| Taille des patches | 8 × 8 |
| Nombre de patches | 16 |
| Dimension des embeddings | 96 |
| Nombre de têtes d'attention | 3 |
| Couches Transformer | 2 |
| Classes de sortie | 10 |

### Fonctionnement

```text
Image RGB (3 × 32 × 32)
          |
          v
Patch Embedding (Conv2d)
          |
          v
16 Patch Tokens (dimension 96)
          |
          v
Ajout CLS Token + DIST Token
          |
          v
Position Embedding
          |
          v
Transformer Encoder (2 couches)
          |
          v
      LayerNorm
          |
     +----+----+
     |         |
     v         v
 CLS Head   DIST Head
     |         |
     v         v
Classification  Distillation
```

## 3. Knowledge Distillation

DeiT utilise un mécanisme de distillation de connaissances avec deux modèles :

- **Teacher :** réseau CNN qui fournit des prédictions.
- **Student :** réseau Transformer qui apprend à partir des vraies étiquettes et des prédictions du Teacher.

Dans notre implémentation, le Teacher est représenté par `SimpleTeacher(nn.Module)`, un petit CNN non pré-entraîné.

La fonction de perte utilisée correspond à une version pédagogique de la *Hard Distillation* :

**Loss totale = 0.5 × Loss classification + 0.5 × Loss distillation**

Les deux pertes utilisent la fonction `CrossEntropyLoss`.

## 4. Structure du projet

```text
DeiT-PyTorch-Implementation/
├── deit_pedagogique.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Description des fichiers

| Fichier | Description |
|---|---|
| `deit_pedagogique.py` | Architecture MiniDeiT, Teacher, loss et boucle d'apprentissage |
| `requirements.txt` | Dépendances Python |
| `README.md` | Documentation du projet |
| `.gitignore` | Exclusion des environnements virtuels et fichiers temporaires |

## 5. Installation

### Cloner le projet

```bash
git clone https://github.com/YaiciAnis/DeiT-PyTorch-Implementation.git
cd DeiT-PyTorch-Implementation
```

### Créer un environnement virtuel

Sous Linux ou WSL2 :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Installer les dépendances

```bash
python -m pip install -r requirements.txt
```

Le projet utilise principalement **PyTorch**.

## 6. Exécution

Pour exécuter le programme :

```bash
python deit_pedagogique.py
```

Exemple de résultat :

```text
Epoch 1/2 | loss du dernier batch = 1.3165
Epoch 2/2 | loss du dernier batch = 0.4650
Forme des probabilités finales : (2, 10)
```

Les valeurs peuvent varier selon l'environnement d'exécution.

## 7. Processus d'apprentissage

Le programme utilise des données fictives afin de démontrer les étapes d'entraînement :

| Élément | Configuration |
|---|---|
| Nombre d'images | 20 |
| Dimensions | 3 × 32 × 32 |
| Nombre de classes | 10 |
| Batch size | 5 |
| Nombre d'epochs | 2 |
| Optimiseur | AdamW |
| Learning rate | 0.001 |
| Fonction de perte | Cross-Entropy + Hard Distillation |

### Étapes de l'apprentissage

1. Chargement d'un batch d'images.
2. Prédiction du Student.
3. Prédiction du Teacher sans calcul des gradients.
4. Calcul de la fonction de perte.
5. Rétropropagation avec `loss.backward()`.
6. Mise à jour des paramètres avec `optimizer.step()`.
7. Répétition pour chaque batch et chaque epoch.

Les données sont générées aléatoirement et le Teacher n'est pas pré-entraîné. Les résultats ne constituent donc pas une évaluation de la qualité des prédictions.

## 8. Technologies

- Python 3
- PyTorch
- VS Code
- Ubuntu / WSL2
- Git et GitHub

