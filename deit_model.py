
import torch
import torch.nn as nn


# ============================================================
# ÉTAPE 1 — PATCH EMBEDDING
# ============================================================
# Image : (B, 3, 224, 224)
# Patch : 16 x 16
# Nombre de patches : 196
# Dimension d'embedding : 192
# Sortie : (B, 196, 192)
# ============================================================

# Les valeurs par défaut correspondent à ImageNet (224, patch 16).
# Pour CIFAR-10 : img_size=32 et patch_size=4 (soit 8 x 8 = 64
# patches, chacun de 3 x 4 x 4 = 48 valeurs).

class PatchEmbedding(nn.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=192
    ):
        super().__init__()

        assert img_size % patch_size == 0

        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, images):

        # Exemple ImageNet :
        # (B, 3, 224, 224) -> (B, 192, 14, 14)
        x = self.proj(images)

        # (B, 192, 14, 14) -> (B, 192, 196)
        x = x.flatten(2)

        # (B, 192, 196) -> (B, 196, 192)
        x = x.transpose(1, 2)

        return x


# ============================================================
# ÉTAPE 2 — CLS TOKEN, DIST TOKEN ET POSITION
# ============================================================
# 196 patches + CLS + DIST = 198 tokens.
# Chaque token possède 192 caractéristiques.
# ============================================================

class TokenEmbedding(nn.Module):

    def __init__(self, num_patches=196, embed_dim=192):
        super().__init__()

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.dist_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # Un embedding de position par token :
        # num_patches + CLS + DIST.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 2, embed_dim)
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):

        batch_size = x.shape[0]

        cls = self.cls_token.expand(batch_size, -1, -1)
        dist = self.dist_token.expand(batch_size, -1, -1)

        # Ajouter CLS et DIST au début.
        x = torch.cat([cls, dist, x], dim=1)

        # Ajouter les embeddings de position.
        x = x + self.pos_embed

        return x


# ============================================================
# ÉTAPE 3 — MULTI-HEAD SELF-ATTENTION
# ============================================================
# Dimension : 192
# Nombre de têtes : 3
# Dimension par tête : 64
#
# Attention(Q,K,V) = Softmax(QK^T / sqrt(d_k))V
# ============================================================

class Attention(nn.Module):

    def __init__(self, embed_dim=192, num_heads=3):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        B, N, C = x.shape

        # Produire Q, K et V.
        qkv = self.qkv(x)

        # Séparer les têtes d'attention.
        qkv = qkv.reshape(
            B, N, 3, self.num_heads, self.head_dim
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        # Calculer les scores d'attention.
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # Normaliser les scores.
        attention = scores.softmax(dim=-1)

        # Appliquer l'attention aux valeurs V.
        x = attention @ v

        # Réunir les têtes.
        x = x.transpose(1, 2).reshape(B, N, C)

        return self.proj(x)


# ============================================================
# ÉTAPE 4 — MLP
# ============================================================
# 192 -> 768 -> 192
# Activation : GELU
# ============================================================

class MLP(nn.Module):

    def __init__(self, embed_dim=192, mlp_ratio=4):
        super().__init__()

        hidden_dim = embed_dim * mlp_ratio

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


# ============================================================
# ÉTAPE 8 — STOCHASTIC DEPTH (DROPPATH)
# ============================================================
# Cette classe est placée avant TransformerBlock parce que
# Python doit connaître DropPath lorsqu'il construit un bloc.
#
# Pendant l'entraînement :
# - Certaines branches résiduelles sont désactivées.
# - Les branches conservées sont mises à l'échelle.
#
# Pendant l'évaluation :
# - Aucune branche n'est désactivée.
#
# Les dimensions restent identiques.
# ============================================================

class DropPath(nn.Module):

    def __init__(self, drop_prob=0.0):
        super().__init__()

        assert 0.0 <= drop_prob < 1.0
        self.drop_prob = drop_prob

    def forward(self, x):

        # Ne rien supprimer pendant l'évaluation.
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob

        # Un masque par image du batch.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device
        )

        binary_mask = random_tensor.floor()

        return x * binary_mask / keep_prob


# ============================================================
# ÉTAPE 5 — BLOC TRANSFORMER
# ============================================================
# x = x + DropPath(Attention(LayerNorm(x)))
# x = x + DropPath(MLP(LayerNorm(x)))
#
# DropPath est maintenant défini dans ce même fichier.
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim=192,
        num_heads=3,
        mlp_ratio=4,
        drop_path=0.0
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)

        self.drop_path = DropPath(drop_path)

    def forward(self, x):

        # Branche Self-Attention.
        x = x + self.drop_path(
            self.attn(self.norm1(x))
        )

        # Branche MLP.
        x = x + self.drop_path(
            self.mlp(self.norm2(x))
        )

        return x


# ============================================================
# ÉTAPE 6 — ENCODEUR TRANSFORMER
# ============================================================
# 12 blocs indépendants.
# DropPath progressif entre 0 et 0.1.
# ============================================================

class TransformerEncoder(nn.Module):

    def __init__(
        self,
        embed_dim=192,
        num_heads=3,
        mlp_ratio=4,
        depth=12,
        drop_path_rate=0.1
    ):
        super().__init__()

        rates = torch.linspace(
            0,
            drop_path_rate,
            depth
        ).tolist()

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=rates[i]
            )
            for i in range(depth)
        ])

    def forward(self, x):

        for block in self.blocks:
            x = block(x)

        return x


# ============================================================
# ÉTAPE 7 — MODÈLE DEIT-TINY DISTILLÉ
# ============================================================
# Image -> Patches -> CLS/DIST -> 12 Transformers
#       -> LayerNorm -> Deux têtes de classification
# ============================================================

class DeiTTinyDistilled(nn.Module):

    # Par défaut : DeiT-Tiny pour ImageNet (224, patch 16, 1000 classes).
    # Pour CIFAR-10 :
    #   DeiTTinyDistilled(img_size=32, patch_size=4, num_classes=10)

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_classes=1000,
        embed_dim=192,
        depth=12,
        num_heads=3,
        drop_path_rate=0.1
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )

        self.token_embed = TokenEmbedding(
            num_patches=self.patch_embed.num_patches,
            embed_dim=embed_dim
        )

        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4,
            depth=depth,
            drop_path_rate=drop_path_rate
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Tête de classification CLS.
        self.head = nn.Linear(embed_dim, num_classes)

        # Tête de distillation DIST.
        self.head_dist = nn.Linear(embed_dim, num_classes)

        # Initialisation de toutes les couches (voir ci-dessous).
        self.apply(self._init_weights)

    # --------------------------------------------------------
    # INITIALISATION (section 6 du papier)
    # --------------------------------------------------------
    # Le papier explique que les Transformers sont sensibles
    # à l'initialisation et que certaines options ne convergent
    # pas. Ils initialisent les poids avec une loi NORMALE
    # TRONQUÉE (Hanin & Rolnick, 2018).
    #
    # L'écart-type 0.02 n'est pas donné dans le papier : c'est
    # la convention de la bibliothèque timm, sur laquelle DeiT
    # s'appuie.
    # --------------------------------------------------------

    @staticmethod
    def _init_weights(module):

        if isinstance(module, nn.Linear):

            nn.init.trunc_normal_(module.weight, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):

            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):

        # Extraire les patches.
        x = self.patch_embed(images)

        # Ajouter les tokens et les positions.
        x = self.token_embed(x)

        # Appliquer les 12 blocs Transformer.
        x = self.encoder(x)

        # Normalisation finale.
        x = self.norm(x)

        # CLS : position 0.
        cls_token = x[:, 0]

        # DIST : position 1.
        dist_token = x[:, 1]

        # Deux têtes indépendantes.
        cls_logits = self.head(cls_token)
        dist_logits = self.head_dist(dist_token)

        return cls_logits, dist_logits

    # --------------------------------------------------------
    # PRÉDICTION FINALE : FUSION TARDIVE (section 4 du papier)
    # --------------------------------------------------------
    # À l'inférence, les deux têtes savent prédire la classe.
    # La méthode de référence du papier est la "fusion tardive" :
    # on additionne les SORTIES SOFTMAX des deux classifieurs.
    # Cette méthode ne s'entraîne pas, elle sert à l'évaluation.
    # Appeler model.eval() avant, pour désactiver DropPath.
    # --------------------------------------------------------

    @torch.no_grad()
    def predict(self, images):

        cls_logits, dist_logits = self(images)

        # Chaque softmax donne une distribution (somme = 1).
        cls_probs = cls_logits.softmax(dim=-1)
        dist_probs = dist_logits.softmax(dim=-1)

        # (B, num_classes), somme = 2. La classe prédite
        # est l'argmax de ce résultat.
        return cls_probs + dist_probs


# ============================================================
# TESTS DU STUDENT — ÉTAPES 1 À 8
# ============================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    model = DeiTTinyDistilled(
        num_classes=1000,
        drop_path_rate=0.1
    )

    model.eval()

    images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():

        patches = model.patch_embed(images)
        tokens = model.token_embed(patches)
        encoder_output = model.encoder(tokens)

        cls_logits, dist_logits = model(images)

    print("Images :", images.shape)
    print("Après Patch Embedding :", patches.shape)
    print("Après CLS + DIST + Position :", tokens.shape)
    print("Nombre de blocs Transformer :", len(model.encoder.blocks))
    print("Après les 12 blocs Transformer :", encoder_output.shape)
    print("Sortie CLS :", cls_logits.shape)
    print("Sortie DIST :", dist_logits.shape)

    print(
        "DropPath premier bloc :",
        model.encoder.blocks[0].drop_path.drop_prob
    )

    print(
        "DropPath dernier bloc :",
        model.encoder.blocks[-1].drop_path.drop_prob
    )

    num_parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("Nombre de paramètres :", num_parameters)

    # Vérifications automatiques.
    assert patches.shape == (2, 196, 192)
    assert tokens.shape == (2, 198, 192)
    assert encoder_output.shape == (2, 198, 192)

    assert len(model.encoder.blocks) == 12
    assert cls_logits.shape == (2, 1000)
    assert dist_logits.shape == (2, 1000)

    assert num_parameters == 5910800

    # Vérifier le fonctionnement du DropPath.
    drop_path = DropPath(drop_prob=0.5)

    test_tensor = torch.ones(100, 198, 192)

    drop_path.train()

    with torch.no_grad():
        train_output = drop_path(test_tensor)

    drop_path.eval()

    with torch.no_grad():
        eval_output = drop_path(test_tensor)

    dropped = (
        train_output[:, 0, 0] == 0
    ).sum().item()

    print("Branches désactivées :", dropped)

    assert train_output.shape == test_tensor.shape
    assert torch.equal(eval_output, test_tensor)
    assert 0 < dropped < 100

    # Vérifier la fusion tardive (somme de deux softmax = 2).
    with torch.no_grad():
        probs = model.predict(images)

    predicted_classes = probs.argmax(dim=-1)

    print("Fusion tardive :", probs.shape)
    print("Classes prédites :", predicted_classes.tolist())

    assert probs.shape == (2, 1000)
    assert torch.allclose(
        probs.sum(dim=-1),
        torch.full((2,), 2.0),
        atol=1e-4
    )
    assert predicted_classes.shape == (2,)

    # Vérifier l'initialisation normale tronquée (std = 0.02).
    qkv_weight = model.encoder.blocks[0].attn.qkv.weight

    print("Écart-type des poids qkv :", qkv_weight.std().item())

    assert abs(qkv_weight.std().item() - 0.02) < 0.002
    assert torch.count_nonzero(model.head.bias) == 0
    assert torch.count_nonzero(model.head_dist.bias) == 0
    assert torch.all(model.norm.weight == 1.0)

    # Vérifier la configuration CIFAR-10 :
    # images 32 x 32, patchs 4 x 4, 10 classes.
    cifar_model = DeiTTinyDistilled(
        img_size=32,
        patch_size=4,
        num_classes=10
    )

    cifar_model.eval()

    cifar_images = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        c_cls, c_dist = cifar_model(cifar_images)
        c_probs = cifar_model.predict(cifar_images)

    print("CIFAR-10 : patches :", cifar_model.patch_embed.num_patches)
    print("CIFAR-10 : tokens  :", cifar_model.token_embed.pos_embed.shape[1])

    assert cifar_model.patch_embed.num_patches == 64
    assert cifar_model.token_embed.pos_embed.shape == (1, 66, 192)
    assert c_cls.shape == (2, 10)
    assert c_dist.shape == (2, 10)
    assert c_probs.shape == (2, 10)

    print("Tests des étapes 1 à 8 réussis !")
