# 🤖 DeiT — Explication Complète du Code `Untitled-1.py`
### Data-efficient Image Transformers — Implémentation PyTorch Pédagogique

> **Basé sur le papier :** *Training data-efficient image transformers & distillation through attention*
> Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa — **ICML 2021**
> 📄 [Lien arXiv](https://arxiv.org/abs/2012.12877)

---

## 📋 Table des Matières

1. [🧠 Concepts Clés à Comprendre](#-concepts-clés-à-comprendre)
2. [🏗️ Architecture Globale](#️-architecture-globale)
3. [📦 Module par Module](#-module-par-module)
   - [1. DropPath — Stochastic Depth](#1-droppath--stochastic-depth)
   - [2. PatchEmbed](#2-patchembed)
   - [3. Attention Multi-Head](#3-attention-multi-head)
   - [4. MLP — Feed-Forward Network](#4-mlp--feed-forward-network)
   - [5. Block — Transformer Block](#5-block--transformer-block)
   - [6. DeiT — Modèle Student Complet](#6-deit--modèle-student-complet)
   - [7. SimpleTeacherCNN](#7-simpleteachercnn)
   - [8. LabelSmoothingCrossEntropy](#8-labelsmoothingcrossentropy)
   - [9. Fonctions de Distillation](#9-fonctions-de-distillation)
   - [10. ModelEMA](#10-modelema)
   - [11. train_one_epoch](#11-train_one_epoch)
   - [12. evaluate](#12-evaluate)
   - [13. demo](#13-demo)
4. [🔁 Flux d'entraînement complet](#-flux-dentraînement-complet)
5. [📐 Dimensions des Tenseurs](#-dimensions-des-tenseurs)
6. [🔄 Ce qui a changé vs la version précédente](#-ce-qui-a-changé-vs-la-version-précédente)
7. [📖 Glossaire](#-glossaire)
8. [⚙️ Installation & Exécution](#️-installation--exécution)

---

## 🧠 Concepts Clés à Comprendre

### Vision Transformer (ViT) — La base de DeiT

L'idée centrale : **traiter une image comme une séquence de tokens**, exactement comme des mots en NLP.

```
Image 224x224
      │
      ▼  Découpage
Patches 16x16 (196 au total)
      │
      ▼  Projection linéaire (Conv2d)
Séquence de vecteurs [B, 196, D]
      │
      ▼
+ CLS token + DIST token (tokens spéciaux)
      │
      ▼
Transformer → Self-Attention sur tous les tokens
      │
      ▼
2 têtes de classification
```

### Pourquoi DeiT et pas ViT ?

| Aspect | ViT | DeiT |
|--------|-----|------|
| Données nécessaires | 300M+ images | 1.2M images (ImageNet seul) |
| Token spécial | CLS seulement | CLS **+** DIST (distillation) |
| Teacher | Aucun | CNN pré-entraîné (ex: RegNet) |
| Technique clé | — | **Knowledge Distillation** |

---

## 🏗️ Architecture Globale

```
┌──────────────────────────────────────────────────────────────────┐
│                        MODÈLE DeiT                               │
│                                                                  │
│  Image RGB [B, 3, 224, 224]                                      │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────┐                                             │
│  │  PatchEmbed     │  Conv2d(stride=16) → [B, 196, D]           │
│  └────────┬────────┘                                             │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────────────────────────┐                        │
│  │  Concat CLS token + DIST token       │  [B, 198, D]          │
│  │  + Positional Embeddings             │                        │
│  └────────────────┬─────────────────────┘                        │
│                   │                                              │
│                   ▼                                              │
│  ┌─────────────────────────────────┐                             │
│  │     Transformer Block x depth   │                             │
│  │  ┌───────────────────────────┐  │                             │
│  │  │  LayerNorm → Attention    │  │                             │
│  │  │  + Residual + DropPath    │  │  x depth blocs             │
│  │  │  LayerNorm → MLP          │  │                             │
│  │  │  + Residual + DropPath    │  │                             │
│  │  └───────────────────────────┘  │                             │
│  └────────────────┬────────────────┘                             │
│                   │                                              │
│                   ▼                                              │
│            LayerNorm finale                                      │
│                   │                                              │
│         ┌─────────┴──────────┐                                   │
│         ▼                    ▼                                   │
│    x[:,0] CLS          x[:,1] DIST                              │
│         │                    │                                   │
│         ▼                    ▼                                   │
│   class_head           dist_head                                 │
│   Linear(D→C)          Linear(D→C)                               │
│         │                    │                                   │
│         └──────┬─────────────┘                                   │
│                ▼                                                 │
│   (class_logits + dist_logits) / 2 → softmax                    │
│                ▼                                                 │
│         Probabilités finales [B, C]                              │
└──────────────────────────────────────────────────────────────────┘
```

> **Nouveauté vs ancienne version :** `predict_proba` fait maintenant la moyenne des **logits bruts** avant softmax, et non plus la moyenne des softmax séparés. Résultat identique en pratique, mais plus propre mathématiquement.

---

## 📦 Module par Module

---

### 1. `DropPath` — Stochastic Depth

**Lignes 12–29**

```python
class DropPath(nn.Module):
    """Ignore aleatoirement une branche residuelle, par image, au training."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob doit etre dans [0, 1[.")
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x                         # ← En évaluation : rien ne change
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )).floor()                           # → 0 ou 1 par image
        return x * mask / keep_prob          # Masquage + mise à l'échelle
```

**Rôle :** Désactiver aléatoirement des **blocs entiers** pendant l'entraînement, image par image.

#### Changements vs ancienne version

| Avant | Après |
|-------|-------|
| Pas de validation des paramètres | `ValueError` si `drop_prob` hors `[0, 1[` |
| `x / keep_prob * random_tensor` | `x * mask / keep_prob` — ordre plus clair |
| Variable nommée `random_tensor` | Variable nommée `mask` — plus lisible |

#### Visualisation du mécanisme

```
Batch de 4 images, drop_prob=0.25 :

Image 1 : mask=1 → bloc actif,   sortie = x / keep_prob
Image 2 : mask=0 → bloc ignoré,  sortie = 0
Image 3 : mask=1 → bloc actif,   sortie = x / keep_prob
Image 4 : mask=0 → bloc ignoré,  sortie = 0

La division par keep_prob (0.75) compense pour que l'espérance reste correcte.
```

#### Pourquoi masquer par image et pas par batch ?

```
shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      = (B,          1,          1,          1)   pour un tenseur 4D

→ Chaque image a son propre masque 0 ou 1.
→ Dans un même batch, certaines images passent par le bloc, d'autres non.
→ Régularisation plus fine et plus efficace.
```

---

### 2. `PatchEmbed` — Découpage en Patches

**Lignes 34–59**

```python
class PatchEmbed(nn.Module):
    """Image [B, 3, 224, 224] -> sequence [B, 196, D], si patch_size=16."""

    def __init__(self, image_size=224, patch_size=16,
                 in_channels=3, embed_dim=384):
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size != 0:
            raise ValueError("image_size doit etre positif et divisible par patch_size.")
        self.image_size  = image_size
        self.patch_size  = patch_size
        self.grid_size   = image_size // patch_size
        self.num_patches = self.grid_size ** 2          # ← 14² = 196
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError("Forme attendue : [batch, canaux, hauteur, largeur].")
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(...)
        x = self.proj(images)                    # [B, D, 14, 14]
        return x.flatten(2).transpose(1, 2)      # [B, 196, D]
```

**Rôle :** Découper l'image en N patches carrés et les projeter en vecteurs de dimension D.

#### Changements vs ancienne version

| Avant | Après |
|-------|-------|
| `assert` pour les validations | `ValueError` avec messages explicites |
| `self.num_patches = grid_size * grid_size` | `self.num_patches = grid_size ** 2` |
| Pas de vérification dans `forward` | Validation de la forme des images à l'entrée |
| `self.proj(x)`, `x.flatten(2)`, `x.transpose(1,2)` (3 lignes) | Chaîné en une seule ligne de retour |

#### Visualisation du découpage

```
Image 224×224 (3 canaux RGB)
┌─────┬─────┬─────┬─────┬───┬─────┐
│ P1  │ P2  │ P3  │ P4  │...│ P14 │  ← rangée 1
├─────┼─────┼─────┼─────┼───┼─────┤
│ P15 │     │     │     │...│ P28 │  ← rangée 2
├─────┼─────┼─────┼─────┼───┼─────┤
│ ... │                         ...│
├─────┼─────┼─────┼─────┼───┼─────┤
│     │     │     │     │...│ P196│  ← rangée 14
└─────┴─────┴─────┴─────┴───┴─────┘

Chaque patch 16×16 = 768 valeurs brutes
          │
          ▼  Conv2d (kernel=16, stride=16) — projection linéaire apprise
          │
    Vecteur de taille D=384

num_patches = (224/16)² = 14² = 196
```

#### Pourquoi `resolution fixe` dans cette version ?

```python
if images.shape[-2:] != (self.image_size, self.image_size):
    raise ValueError("Cette version utilise une resolution fixe.")
```

L'ancienne version avait une méthode `interpolate_pos_encoding` pour les images de taille différente (ex: 384×384). Cette version **supprime cette complexité** et impose une résolution fixe à l'entrée. Plus simple, moins de bugs potentiels.

---

### 3. `Attention` — Multi-Head Self-Attention

**Lignes 64–88**

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("dim doit etre divisible par num_heads > 0.")
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5    # 1/√d_head
        self.qkv       = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores     = (q @ k.transpose(-2, -1)) * self.scale
        attention  = self.attn_drop(scores.softmax(dim=-1))
        x          = attention @ v
        x          = x.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(x))
```

**Rôle :** Permettre à chaque token de "regarder" tous les autres et de s'intéresser à eux de manière pondérée.

#### Changements vs ancienne version

| Avant | Après |
|-------|-------|
| `self.qkv = nn.Linear(dim, dim * 3, ...)` | `nn.Linear(dim, 3 * dim, ...)` — même résultat, ordre conventionnel |
| `attn = ... ; attn = attn.softmax() ; attn = self.attn_drop(attn)` (3 lignes) | `attention = self.attn_drop(scores.softmax(dim=-1))` — chaîné |
| `self.proj(x) ; self.proj_drop(x)` séparés | `return self.proj_drop(self.proj(x))` — une ligne |
| `self.dim` stocké comme attribut | Attribut `self.dim` supprimé (non utilisé) |
| Variable `attn` | Variable `attention` — nom plus explicite |

#### La formule mathématique

```
                          Q × K^T
Attention(Q, K, V)  =  softmax(────────────) × V
                               √(d_head)

Où d_head = D / num_heads = 384 / 6 = 64
```

#### Transformation des dimensions dans `forward`

```
x         : [B, 198, 384]
qkv(x)    : [B, 198, 1152]      (3 × 384 = 1152)
reshape   : [B, 198, 3, 6, 64]
permute   : [3, B, 6, 198, 64]
q,k,v     : [B, 6, 198, 64]    (chacun)

q @ k^T   : [B, 6, 198, 198]   ← matrice d'attention
× scale   : [B, 6, 198, 198]
softmax   : [B, 6, 198, 198]

attn @ v  : [B, 6, 198, 64]
transpose + reshape : [B, 198, 384]
proj      : [B, 198, 384]
```

#### Q, K, V — La métaphore

```
Chaque token pose une question (Q), affiche une étiquette (K),
et porte une information (V).

Score de compatibilité = Q · K^T / √d
                       = "à quel point ce token m'intéresse-t-il ?"

softmax(scores) → poids d'attention (somme = 1)
poids × V       → information agrégée
```

---

### 4. `MLP` — Feed-Forward Network

**Lignes 93–105**

```python
class MLP(nn.Module):
    """Applique D -> hidden_dim -> D independamment a chaque token."""

    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1  = nn.Linear(dim, hidden_dim)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))
```

**Rôle :** Traiter chaque token **indépendamment** après l'attention. Ajoute de la non-linéarité.

#### Changements vs ancienne version

| Avant | Après |
|-------|-------|
| `__init__(self, in_features, hidden_features=None, out_features=None, ...)` | `__init__(self, dim, hidden_dim, ...)` — signature simplifiée |
| `out_features = out_features or in_features` | Supprimé : `dim_entrée == dim_sortie` toujours |
| `hidden_features = hidden_features or in_features * 4` | Le calcul fait à l'extérieur (`int(dim * mlp_ratio)` dans `Block`) |
| 5 lignes dans `forward` | 2 lignes — opérations chaînées |

```
Flux dans le MLP :

Entrée  [B, N, 384]
    │
    ▼ fc1 : Linear(384 → 1536)   (ratio 4 : calculé dans Block)
    │
    ▼ GELU + Dropout
    │
    ▼ fc2 : Linear(1536 → 384)
    │
    ▼ Dropout
    │
Sortie  [B, N, 384]
```

---

### 5. `Block` — Transformer Block

**Lignes 110–124**

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1      = nn.LayerNorm(dim)
        self.attn       = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path1 = DropPath(drop_path)
        self.norm2      = nn.LayerNorm(dim)
        self.mlp        = MLP(dim, int(dim * mlp_ratio), drop)  # ← hidden_dim calculé ici
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
```

**Rôle :** Combiner Attention + MLP avec connexions résiduelles et LayerNorm pré-appliquée.

#### Architecture d'un bloc

```
Entrée x  [B, N, D]
    │
    ├─────────────────────────────────────┐
    │                                     │  connexion résiduelle
    ▼                                     │
LayerNorm (norm1)                         │
    │                                     │
    ▼                                     │
Multi-Head Attention                      │
    │                                     │
    ▼                                     │
DropPath1 ────────────────────── (+) ◄───┘
                                   │
                             x actualisé
                                   │
    ├─────────────────────────────────────┐
    │                                     │  connexion résiduelle
    ▼                                     │
LayerNorm (norm2)                         │
    │                                     │
    ▼                                     │
MLP(dim, dim×4)                          │
    │                                     │
    ▼                                     │
DropPath2 ────────────────────── (+) ◄───┘
                                   │
                             Sortie x  [B, N, D]
```

#### Changements vs ancienne version

| Avant | Après |
|-------|-------|
| `Attention(dim, num_heads=num_heads, ...)` | `Attention(dim, num_heads, qkv_bias, attn_drop, drop)` — arguments positionnels |
| `MLP(in_features=dim, hidden_features=hidden_dim, ...)` | `MLP(dim, int(dim * mlp_ratio), drop)` — signature plus courte |
| Variable `hidden_dim` déclarée séparément | Calculée directement dans l'appel |

---

### 6. `DeiT` — Modèle Student Complet

**Lignes 129–196**

```python
class DeiT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=384, depth=12, num_heads=6,
                 mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path_rate=0.1):
        super().__init__()
        if depth < 1 or num_classes < 2:
            raise ValueError("Utiliser au moins un bloc et deux classes.")
        ...
        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, ..., drop_path=rates[i])
            for i in range(depth)
        ])
```

#### Changements majeurs vs ancienne version

| Avant | Après |
|-------|-------|
| Pas de validation des paramètres | `ValueError` si `depth < 1` ou `num_classes < 2` |
| `dpr = torch.linspace(...)` | `rates = torch.linspace(...)` — nom plus court |
| `self.image_size` utilisé dans `interpolate_pos_encoding` | Utilisé uniquement dans `PatchEmbed` maintenant |
| Méthode `interpolate_pos_encoding` présente | **Supprimée** — résolution fixe uniquement |

#### `_init_weights` — Initialisation

```python
def _init_weights(self):
    for parameter in (self.cls_token, self.dist_token, self.pos_embed):
        nn.init.trunc_normal_(parameter, std=0.02)
    for module in self.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
```

| Type de paramètre | Initialisation |
|-------------------|----------------|
| CLS, DIST, pos_embed | `trunc_normal_(std=0.02)` |
| Poids Linear et Conv2d | `trunc_normal_(std=0.02)` |
| Biais Linear et Conv2d | `zeros_` |
| Poids LayerNorm | `ones_` (neutre → ne change rien au départ) |
| Biais LayerNorm | `zeros_` |

#### `forward_features` — Le cœur du modèle

```python
def forward_features(self, images):
    batch_size = images.shape[0]
    patches    = self.patch_embed(images)                  # [B, 196, D]
    cls        = self.cls_token.expand(batch_size, -1, -1) # [B, 1, D]
    dist       = self.dist_token.expand(batch_size, -1, -1)# [B, 1, D]
    x          = torch.cat([cls, dist, patches], dim=1)    # [B, 198, D]
    x          = self.pos_drop(x + self.pos_embed)         # + position
    for block in self.blocks:
        x = block(x)
    x = self.norm(x)
    return x[:, 0], x[:, 1]   # CLS final, DIST final
```

#### `predict_proba` — Inférence (version modifiée)

```python
@torch.no_grad()
def predict_proba(self, images):
    """Appeler model.eval() avant cette methode : moyenne des logits puis softmax."""
    class_logits, dist_logits = self(images)
    final_logits = (class_logits + dist_logits) / 2.0
    return final_logits.softmax(dim=1)
```

**Changement important :**

```
Ancienne version :
  class_probs  = softmax(class_logits)
  dist_probs   = softmax(dist_logits)
  final        = (class_probs + dist_probs) / 2     ← moyenne des softmax

Nouvelle version :
  final_logits = (class_logits + dist_logits) / 2   ← moyenne des logits
  final        = softmax(final_logits)               ← softmax unique

Les deux sont équivalents pour argmax, mais la version logits est plus correcte
mathématiquement et plus efficace (un seul softmax au lieu de deux).

Aussi : @torch.no_grad() est maintenant décoré directement sur la méthode
        → pas besoin du bloc "with torch.no_grad():" à l'extérieur.
```

#### Les Tokens Spéciaux

```
Séquence d'entrée au Transformer :

Position : [  0  ] [  1  ] [  2  ] [  3  ] ... [  197  ]
Token     : [ CLS ] [DIST ] [ P1  ] [ P2  ] ... [  P196  ]
              │       │
              │       └── État final → dist_head → imite le Teacher
              └────────── État final → class_head → classification réelle

Via l'attention : CLS et DIST "voient" tous les 196 patches → info globale.
```

#### Positional Embedding

```
Problème : le Transformer ne sait pas quelle patch est où dans l'image.

Solution : ajouter un vecteur de position appris à chaque token.

pos_embed : [1, 198, D]  (un vecteur par position)
x         : [B, 198, D]

x + pos_embed → broadcasting automatique sur la dimension batch
```

#### Stochastic Depth Progressif

```python
rates = torch.linspace(0, drop_path_rate, depth).tolist()
# depth=12, drop_path_rate=0.1
# → [0.0, 0.009, ..., 0.09, 0.1]
```

```
Bloc  1 : drop_prob = 0.000  ← jamais ignoré (features bas niveau)
Bloc  6 : drop_prob = 0.045
Bloc 12 : drop_prob = 0.100  ← ignoré 10% du temps (features haut niveau)

Intuition : les derniers blocs sont plus spécialisés → plus de régularisation.
```

---

### 7. `SimpleTeacherCNN`

**Lignes 201–212**

```python
class SimpleTeacherCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, images):
        return self.classifier(self.features(images))
```

**Rôle :** Teacher illustratif. Dans le vrai DeiT, c'est un **RegNet-Y-16GF** pré-entraîné (~84% top-1 ImageNet).

#### Changements vs ancienne version

| Avant | Après |
|-------|-------|
| `nn.Conv2d(3, 32, 3, padding=1)` | `nn.Conv2d(3, 32, kernel_size=3, padding=1)` — argument nommé |
| `nn.AdaptiveAvgPool2d((1, 1))` | `nn.AdaptiveAvgPool2d(1)` — tuple inutile |
| `forward` avec deux lignes | `return self.classifier(self.features(images))` — une ligne |

```
Image [B, 3, 224, 224]
    │
    ▼ Conv2d(3→32) + ReLU
    │
    ▼ Conv2d(32→64) + ReLU
    │
    ▼ AdaptiveAvgPool2d(1) → [B, 64, 1, 1]
    │
    ▼ Flatten → [B, 64]
    │
    ▼ Linear(64→num_classes)
    │
teacher_logits [B, num_classes]
```

> **Gelé dans la démo :** `teacher.requires_grad_(False)` — aucun gradient calculé côté teacher.

---

### 8. `LabelSmoothingCrossEntropy`

**Lignes 217–231**

```python
class LabelSmoothingCrossEntropy(nn.Module):
    """Cible lissee : (1 - epsilon) * one_hot(y) + epsilon / K."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing doit etre dans [0, 1[.")
        self.smoothing = smoothing

    def forward(self, logits, target):
        log_probs    = F.log_softmax(logits, dim=1)
        nll          = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        uniform_loss = -log_probs.mean(dim=1)
        return ((1.0 - self.smoothing) * nll
                + self.smoothing * uniform_loss).mean()
```

**Rôle :** Remplacer la cible one-hot par une distribution légèrement lissée pour éviter la surconfiance.

#### Changements vs ancienne version

| Avant | Après |
|-------|-------|
| Calcul via `true_dist.fill_()` et `scatter_()` | Calcul via `gather` sur `log_probs` — plus efficace |
| Utilisation de `torch.no_grad()` à l'intérieur | Pas nécessaire — opérations sur `log_probs` seulement |
| Validation absente | `ValueError` si `smoothing` hors `[0, 1[` |

#### Explication du calcul

```python
log_probs    = F.log_softmax(logits, dim=1)          # log(p_i) pour toutes les classes
nll          = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
                                                       # = -log(p_vrai) : NLL classique
uniform_loss = -log_probs.mean(dim=1)                # = -moyenne(log(p_i)) : entropie uniforme
```

```
Loss = (1 - ε) × NLL  +  ε × entropie_uniforme

Avec ε=0.1 :
  - La vraie classe contribue à 90%
  - Les autres classes contribuent à 10% uniformément

Effet :
  SANS smoothing : target = [0, 0, 1, 0, 0]  → modèle vise 100% sur classe 3
  AVEC smoothing : modèle vise ~90% sur classe 3, ~2.5% sur les autres
```

---

### 9. Fonctions de Distillation

**Lignes 236–256**

#### Hard Distillation

```python
def hard_distillation_loss(class_logits, dist_logits, true_labels,
                           teacher_logits, smoothing=0.1):
    """L = 0.5 * CE_lissee(CLS, y) + 0.5 * CE(DIST, argmax(teacher))."""
    loss_cls      = LabelSmoothingCrossEntropy(smoothing)(class_logits, true_labels)
    teacher_labels = teacher_logits.detach().argmax(dim=1)   # ← .detach() explicite
    loss_dist     = F.cross_entropy(dist_logits, teacher_labels)
    return 0.5 * loss_cls + 0.5 * loss_dist
```

#### Changement clé : `.detach()` explicite

```python
# Avant :
teacher_labels = teacher_logits.argmax(dim=1)

# Après :
teacher_labels = teacher_logits.detach().argmax(dim=1)
```

`.detach()` coupe explicitement le graphe de calcul du teacher. Même si le teacher est sous `torch.no_grad()` dans la boucle, c'est une bonne pratique défensive.

```
Hard Distillation :

Teacher logits : [0.1, 9.2, 1.3, 0.4]
                        ↓ argmax
teacher_label  :        1              (classe "gagnante")

Branche CLS  : CE_lissée(class_logits, y_vrai)
Branche DIST : CE(dist_logits, 1)

Loss = 0.5 × CE_CLS + 0.5 × CE_DIST
```

#### Soft Distillation

```python
def soft_distillation_loss(class_logits, dist_logits, true_labels,
                           teacher_logits, tau=3.0, lam=0.1, smoothing=0.1):
    """L = (1-lam) * CE_lissee(CLS, y) + lam * tau^2 * KL(teacher || DIST)."""
    if tau <= 0.0 or not 0.0 <= lam <= 1.0:
        raise ValueError("Utiliser tau > 0 et lam dans [0, 1].")
    loss_cls          = LabelSmoothingCrossEntropy(smoothing)(class_logits, true_labels)
    student_log_probs = F.log_softmax(dist_logits / tau, dim=1)
    teacher_probs     = F.softmax(teacher_logits.detach() / tau, dim=1)  # ← .detach()
    loss_kl           = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * tau**2
    return (1.0 - lam) * loss_cls + lam * loss_kl
```

```
Température τ :
  τ=1 (normal)  : softmax piquée → peu d'info sur les similarités
  τ=3 (adoucie) : softmax lissée → plus d'info transférée

Avec τ=3 :
  Teacher logits  : [0.1, 9.2, 1.3, 0.4]
  Teacher /τ      : [0.03, 3.07, 0.43, 0.13]
  softmax(/τ)     : [0.08, 0.68, 0.17, 0.07]   ← distribution "douce"

Loss = (1-λ) × CE_CLS  +  λ × τ² × KL(student/τ || teacher/τ)
```

#### Comparaison Hard vs Soft

| Critère | Hard | Soft |
|---------|------|------|
| Info transmise | Classe argmax | Distribution complète |
| Hyperparamètres | `smoothing` | `tau`, `lam`, `smoothing` |
| Résultat papier | **Meilleur** | Légèrement inférieur |
| Loss utilisée | Cross-Entropy | Divergence KL |

---

### 10. `ModelEMA` — Exponential Moving Average

**Lignes 261–280**

```python
class ModelEMA:
    """Copie lissee du student ; aucune backpropagation dans cette copie."""

    def __init__(self, model, decay=0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay doit etre dans [0, 1[.")
        self.ema_model = copy.deepcopy(model).eval()
        self.ema_model.requires_grad_(False)   # ← sur le modèle entier, plus efficace
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        student_state = model.state_dict()
        for name, ema_value in self.ema_model.state_dict().items():
            student_value = student_state[name].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(self.decay).add_(student_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(student_value)   # ← entiers copiés directement
```

#### Changements vs ancienne version

| Avant | Après |
|-------|-------|
| `decay=0.9999` | `decay=0.999` — valeur légèrement différente dans la démo |
| `for p in ...: p.requires_grad_(False)` | `self.ema_model.requires_grad_(False)` — plus direct |
| Mise à jour via `ema_state[key].copy_(...)` | `ema_value.mul_(decay).add_(student_value, alpha=...)` — in-place |
| Pas de traitement des tenseurs entiers | Vérification `is_floating_point()` → entiers copiés directement |

#### Formule et visualisation

```
θ_ema ← decay × θ_ema + (1 - decay) × θ_student

Avec decay=0.999 :
  θ_ema = 0.999 × θ_ema + 0.001 × θ_nouveau
  → Bouge très lentement (0.1% vers les nouveaux poids par batch)

Poids student : /\/\/\/\/\/   (fluctue avec chaque batch)
Poids EMA     : ──────────    (courbe lisse et stable)

En évaluation → EMA donne de meilleures performances que le student brut.
```

#### Pourquoi séparer les tenseurs flottants des entiers ?

```python
if ema_value.is_floating_point():
    ema_value.mul_(self.decay).add_(...)   # lissage EMA
else:
    ema_value.copy_(student_value)          # copie directe
```

Certains modules PyTorch (ex: `BatchNorm`) stockent des compteurs entiers (`num_batches_tracked`). Faire une moyenne exponentielle sur des entiers n'a pas de sens → on les copie directement.

---

### 11. `train_one_epoch` — Boucle d'Entraînement

**Lignes 285–323**

```python
def train_one_epoch(student, teacher, dataloader, optimizer, device,
                    use_soft_distillation=False, tau=3.0, lam=0.1,
                    smoothing=0.1, ema=None):
    student.train()
    teacher.eval()
    total_loss   = 0.0
    total_images = 0              # ← nouveau compteur

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)   # ← plus efficace que zero_grad()

        class_logits, dist_logits = student(images)
        with torch.no_grad():
            teacher_logits = teacher(images)

        if use_soft_distillation:
            loss = soft_distillation_loss(...)
        else:
            loss = hard_distillation_loss(...)

        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(student)

        total_loss   += loss.item() * labels.size(0)   # ← pondéré par taille du batch
        total_images += labels.size(0)

    if total_images == 0:
        raise ValueError("Le dataloader d'entrainement est vide.")
    return total_loss / total_images   # ← loss par image (pas par batch)
```

#### Changements vs ancienne version

| Avant | Après |
|-------|-------|
| `optimizer.zero_grad()` | `optimizer.zero_grad(set_to_none=True)` — libère la mémoire GPU |
| `total_loss += loss.item()` | `total_loss += loss.item() * labels.size(0)` — pondéré |
| `return total_loss / len(dataloader)` | `return total_loss / total_images` — moyenne par image |
| Pas de vérification dataloader vide | `ValueError` si `total_images == 0` |

#### Pourquoi `set_to_none=True` ?

```
zero_grad()            : met les .grad à 0 (alloue de la mémoire)
zero_grad(set_to_none=True) : supprime les .grad (libère de la mémoire GPU)

→ Plus efficace, recommandé par la documentation PyTorch.
```

#### Pourquoi pondérer par `labels.size(0)` ?

```
Ancienne formule : loss_par_batch / nb_batches
  → Si le dernier batch est plus petit (ex: 2 images au lieu de 4),
    il a autant de poids qu'un batch complet → biais !

Nouvelle formule : (Σ loss_i × taille_batch_i) / nb_images_totales
  → Chaque IMAGE a le même poids → calcul correct.
```

---

### 12. `evaluate` — Évaluation

**Lignes 328–341**

```python
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total, correct = 0, 0
    for images, labels in dataloader:
        images      = images.to(device)
        labels      = labels.to(device)
        predictions = model.predict_proba(images).argmax(dim=1)
        total      += labels.size(0)
        correct    += (predictions == labels).sum().item()
    if total == 0:
        raise ValueError("Le dataloader d'evaluation est vide.")
    return correct / total
```

#### Changements vs ancienne version

| Avant | Après |
|-------|-------|
| `probs = ...; preds = probs.argmax(...)` (2 lignes) | `predictions = model.predict_proba(images).argmax(dim=1)` (1 ligne) |
| Variable nommée `preds` | Variable nommée `predictions` — plus explicite |
| Pas de vérification dataloader vide | `ValueError` si `total == 0` |

---

### 13. `demo` — Démonstration Complète

**Lignes 346–403**

```python
def demo(num_epochs=10, use_soft_distillation=False):
    """Donnees fictives et teacher aleatoire ; aucune phase de fine-tuning."""
    if num_epochs < 1:
        raise ValueError("Utiliser au moins une epoque.")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ...
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    eval_loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    for epoch in range(num_epochs):
        loss = train_one_epoch(...)
        print(f"Epoch {epoch + 1}/{num_epochs} | loss = {loss:.4f}")

    acc     = evaluate(student, eval_loader, device)
    acc_ema = evaluate(ema.ema_model, eval_loader, device)
    ...
    print("Classes predites :", probabilities.argmax(dim=1).cpu().tolist())
```

#### Changements vs ancienne version

| Avant | Après |
|-------|-------|
| `demo()` sans paramètres | `demo(num_epochs=10, use_soft_distillation=False)` — configurable |
| `for epoch in range(2)` (2 epochs fixes) | `for epoch in range(num_epochs)` — 10 par défaut |
| `"cuda" if ... else "cpu"` | `torch.device("cuda" if ... else "cpu")` — objet `device` |
| `import` dans le code principal | `from torch.utils.data import DataLoader, TensorDataset` en haut du fichier |
| Pas de vérification `num_epochs` | `ValueError` si `num_epochs < 1` |
| Pas de `num_workers` | `num_workers=0` explicite (important sur Windows) |
| Un seul dataloader | `train_loader` et `eval_loader` séparés (meilleures pratiques) |
| `ema.ema_model.to(device)` | `ema.ema_model` directement (déjà sur le bon device) |
| Affiche seulement la forme des proba | Affiche aussi les classes prédites |

#### Flux de la démo

```
1. torch.manual_seed(42)     ← reproductibilité
2. Créer student DeiT (config réduite : 4 blocs, embed=192)
3. Créer teacher CNN (aléatoire, non pré-entraîné) + geler
4. Créer AdamW + EMA
5. Générer 20 images fictives + labels fictifs
6. Créer train_loader + eval_loader
7. Pour chaque epoch :
     → train_one_epoch (student apprend, EMA se met à jour)
     → afficher la loss
8. evaluate(student)      → accuracy student
9. evaluate(ema.ema_model) → accuracy EMA
10. predict_proba(images[:2]) → proba et classes prédites
```

---

## 🔁 Flux d'Entraînement Complet

```
┌────────────────────────────────────────────────────────────────────┐
│                    BOUCLE D'ENTRAÎNEMENT DEIT                       │
│                                                                     │
│  Pour chaque epoch :                                               │
│    Pour chaque batch (images, labels) :                            │
│                                                                     │
│         images [B, 3, 224, 224]                                    │
│           │                                                        │
│     ┌─────┴────────────┐                                           │
│     ▼                  ▼                                           │
│  Student             Teacher (gelé, no_grad)                       │
│  (DeiT)                 │                                          │
│     │                   │                                          │
│  CLS    DIST         teacher_logits                               │
│  logits logits           │                                         │
│     │      │             │                                         │
│     └──────┴─────────────┘                                         │
│                  │                                                  │
│          Hard ou Soft Distillation Loss                            │
│                  │                                                  │
│      loss.backward()  → gradients ∂L/∂θ                           │
│      optimizer.step() → θ ← θ - lr × grad (AdamW)                │
│      ema.update()     → θ_ema ← decay×θ_ema + (1-decay)×θ        │
│                                                                     │
│  Fin epoch → evaluate(student) → evaluate(ema_model)              │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Dimensions des Tenseurs

```
Notations :
  B = batch size       (ex: 4)
  N = nb patches       (196 pour image 224×224, patch_size=16)
  D = embed_dim        (384 pour DeiT-S, 192 dans la démo)
  H = num_heads        (6 pour DeiT-S, 3 dans la démo)
  d = head_dim         (D / H = 64)
  C = num_classes      (1000 ImageNet, 10 dans la démo)

Étape                              │ Forme
───────────────────────────────────┼──────────────────────
Image brute (entrée)               │ [B,  3,   224, 224]
Après PatchEmbed (Conv2d + flatten)│ [B,  196, D  ]
+ CLS token                        │ [B,  197, D  ]
+ DIST token                       │ [B,  198, D  ]
+ pos_embed (addition)             │ [B,  198, D  ]
───────────────────────────────────┼──────────────────────
Dans Attention :                   │
  qkv projection                   │ [B,  198, 3D ]
  reshape                          │ [B,  198, 3, H, d]
  permute                          │ [3,  B,   H, 198, d]
  q ou k ou v                      │ [B,  H,   198, d]
  scores QK^T                      │ [B,  H,   198, 198]
  attention (softmax)               │ [B,  H,   198, 198]
  attn × V                         │ [B,  H,   198, d]
  après reshape + proj             │ [B,  198, D  ]
───────────────────────────────────┼──────────────────────
Dans MLP :                         │
  fc1 (D → 4D)                     │ [B,  198, 4D ]
  fc2 (4D → D)                     │ [B,  198, D  ]
───────────────────────────────────┼──────────────────────
Après depth blocs + LayerNorm      │ [B,  198, D  ]
CLS final x[:,0]                   │ [B,  D  ]
DIST final x[:,1]                  │ [B,  D  ]
class_logits                       │ [B,  C  ]
dist_logits                        │ [B,  C  ]
(class_logits + dist_logits) / 2   │ [B,  C  ]
softmax → proba finale             │ [B,  C  ]
```

---

## 🔄 Ce qui a changé vs la version précédente

### Résumé des modifications principales

| Module | Changement |
|--------|-----------|
| **En-tête** | Suppression du docstring global + de `import math` |
| **Imports** | Ajout de `from torch.utils.data import DataLoader, TensorDataset` |
| **Toutes les classes** | Ajout de validations `ValueError` sur les paramètres |
| **`DropPath`** | Masquage avec variable `mask` (plus clair que `random_tensor`) |
| **`PatchEmbed`** | Validation de la forme dans `forward`, résolution fixe, suppression de `interpolate_pos_encoding` |
| **`Attention`** | Nom de variables amélioré (`scores`, `attention`), code chaîné |
| **`MLP`** | Signature simplifiée (`dim, hidden_dim`), `forward` en 2 lignes |
| **`Block`** | Appels simplifiés vers `Attention` et `MLP` |
| **`DeiT._init_weights`** | Boucle plus propre sur les tokens spéciaux |
| **`DeiT.forward_features`** | Noms de variables plus clairs, code condensé |
| **`DeiT.predict_proba`** | Moyenne sur les **logits** (pas les softmax), `@torch.no_grad()` |
| **`ModelEMA`** | Traitement des tenseurs entiers, mise à jour in-place, `requires_grad_` sur le modèle |
| **`train_one_epoch`** | `set_to_none=True`, loss pondérée par image, vérification dataloader vide |
| **`evaluate`** | Vérification dataloader vide, code condensé |
| **`demo`** | Paramétrable (`num_epochs`, `use_soft_distillation`), 2 dataloaders séparés, 10 epochs |

---

## 📖 Glossaire

| Terme | Définition |
|-------|------------|
| **Token** | Unité de base d'une séquence : un patch en vision, un mot en NLP |
| **Embedding** | Représentation d'un token sous forme de vecteur dense de dimension D |
| **CLS Token** | Token spécial "Classificateur". Son état final encode l'info globale. Hérité de BERT |
| **DIST Token** | Token de distillation, introduit par DeiT. Son état final imite le teacher |
| **Patch** | Morceau carré d'image (ex: 16×16 pixels). L'image est découpée en N patches |
| **Self-Attention** | Chaque token regarde tous les autres tokens et s'intéresse à eux proportionnellement |
| **Q, K, V** | Query, Key, Value — les 3 composantes calculées par projection linéaire dans l'attention |
| **Scaled Dot-Product** | `QK^T / √d` — la formule de score de compatibilité entre tokens |
| **Head (tête)** | Sous-espace de l'attention. Multi-head = plusieurs attentions parallèles |
| **MLP/FFN** | Multi-Layer Perceptron / Feed-Forward Network — traitement non-linéaire par token |
| **GELU** | Gaussian Error Linear Unit — activation douce, meilleure que ReLU pour les Transformers |
| **LayerNorm** | Normalisation sur la dimension D. Stabilise l'entraînement. Utilisée en Pre-Norm ici |
| **Pre-Norm** | LayerNorm appliquée AVANT le module (`x + f(LayerNorm(x))`). Plus stable que Post-Norm |
| **Connexion Résiduelle** | `x = x + f(x)`. Permet au gradient de circuler → évite sa disparition |
| **DropPath** | Stochastic Depth — désactive des blocs entiers aléatoirement par image |
| **Stochastic Depth** | Autre nom de DropPath — profondeur effective du réseau varie à chaque batch |
| **Knowledge Distillation** | Entraîner un petit modèle (student) à imiter un grand modèle (teacher) |
| **Hard Distillation** | Le student imite la classe argmax du teacher (label dur) |
| **Soft Distillation** | Le student imite la distribution softmax du teacher (label doux) |
| **Température τ** | Adoucit une distribution softmax. τ > 1 → distribution plus uniforme |
| **Label Smoothing** | Remplace le label one-hot par une distribution lissée. Réduit la surconfiance |
| **NLL** | Negative Log-Likelihood — perte classique de classification : `-log(p_vrai)` |
| **Divergence KL** | Kullback-Leibler — mesure la différence entre deux distributions. Utilisée en soft distillation |
| **AdamW** | Adam avec weight decay découplé. Standard pour entraîner les Transformers |
| **Weight Decay** | Pénalité L2 sur les poids — régularisation contre le surapprentissage |
| **EMA** | Exponential Moving Average — copie lissée des poids qui évolue lentement |
| **Positional Embedding** | Vecteur appris ajouté à chaque token pour encoder sa position dans la séquence |
| **embed_dim (D)** | Dimension de l'espace vectoriel. DeiT-Ti: 192, DeiT-S: 384, DeiT-B: 768 |
| **depth** | Nombre de blocs Transformer empilés. DeiT utilise 12 |
| **mlp_ratio** | Facteur d'expansion du MLP (D → ratio×D → D). Standard: 4 |
| **`set_to_none=True`** | Libère la mémoire GPU en supprimant les gradients plutôt qu'en les mettant à 0 |
| **`is_floating_point()`** | Vérifie si un tenseur est de type flottant (float16/32/64). Entiers → copie directe dans EMA |
| **`@torch.no_grad()`** | Décorateur qui désactive le calcul de gradients dans la méthode décorée |
| **ImageNet** | Dataset de 1.2M images, 1000 classes. Benchmark standard |
| **ViT** | Vision Transformer — précurseur de DeiT, nécessite 300M images |
| **RegNet** | Architecture CNN de Facebook Research. Teacher recommandé dans le papier DeiT |
| **top-1 accuracy** | % d'images où la classe prédite (argmax) correspond à la vraie classe |

---

## ⚙️ Installation & Exécution

### Prérequis

```
Python >= 3.8
PyTorch >= 1.10
```

### Installation

```bash
git clone https://github.com/YaiciAnis/DeiT-PyTorch-Implementation.git
cd DeiT-PyTorch-Implementation
pip install -r requirements.txt
```

### Lancer la démo

```bash
python Untitled-1.py
```

#### Résultat attendu

```
Appareil : cpu
Distillation : hard
Epoch  1/10 | loss = X.XXXX
Epoch  2/10 | loss = X.XXXX
...
Epoch 10/10 | loss = X.XXXX
Accuracy student (entrainement fictif) = X.XXXX
Accuracy EMA (entrainement fictif) = X.XXXX
Forme des probabilites : (2, 10)
Classes predites : [X, X]
Note : donnees fictives et teacher non pre-entraine ; scores illustratifs.
Fin de la demonstration, sans fine-tuning.
```

### Tester la soft distillation

```python
# Modifier la dernière ligne de Untitled-1.py :
demo(num_epochs=10, use_soft_distillation=True)
```

### Variantes DeiT du papier original (ImageNet 1K)

| Modèle | embed_dim | depth | num_heads | Params | Top-1 |
|--------|-----------|-------|-----------|--------|-------|
| DeiT-Ti | 192 | 12 | 3 | 5M | 72.2% |
| DeiT-S  | 384 | 12 | 6 | 22M | 79.8% |
| DeiT-B  | 768 | 12 | 12 | 86M | 81.8% |
| DeiT-B ↑384 | 768 | 12 | 12 | 86M | 83.1% |

---

## 📁 Structure du Projet

```
DeiT-PyTorch-Implementation/
├── Untitled-1.py          ← Implémentation complète DeiT (408 lignes)
├── deit_pedagogique.py    ← Version simplifiée / mini
├── explication.md         ← Ce fichier d'explication
├── deit (1).pdf           ← Article scientifique original
├── requirements.txt       ← Dépendances Python
├── README.md              ← Documentation générale
└── .gitignore
```

---

## 🔗 Références

- 📄 **Papier DeiT** : [arxiv.org/abs/2012.12877](https://arxiv.org/abs/2012.12877)
- 📄 **Papier ViT** : [arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
- 📄 **Attention Is All You Need** : [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- 📄 **ResNet (connexions résiduelles)** : [arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
- 🐙 **GitHub officiel DeiT** : [github.com/facebookresearch/deit](https://github.com/facebookresearch/deit)

---

*Documentation réalisée dans le cadre d'un projet universitaire — M2 Machine Learning, DATASCAL*
