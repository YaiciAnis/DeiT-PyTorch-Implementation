
import math


# ============================================================
# ÉTAPE 4 — LEARNING RATE : MISE À L'ÉCHELLE, WARMUP, COSINUS
# ============================================================
# Section 6 du papier (Regularization & Optimizers) :
#
#   lr_mis_a_echelle = lr / 512 * taille_du_batch
#
# Le papier a testé trois learning rates (5e-4, 3e-4, 5e-5) et
# trois weight decay (0.03, 0.04, 0.05), et utilise AdamW.
#
# ATTENTION : le papier (la version fournie) ne détaille pas le
# schedule. Le warmup linéaire suivi d'une décroissance cosinus
# est la pratique standard de la bibliothèque timm / du code
# officiel de DeiT. Les valeurs par défaut ci-dessous (lr de base
# 5e-4, lr minimal 1e-5) suivent cette convention et sont
# des paramètres modifiables.
# ============================================================


def lr_mis_a_echelle(base_lr=5e-4, batch_size=1024, reference=512):

    # Plus le batch est grand, plus on peut se permettre
    # un learning rate grand (le gradient est moins bruité).
    return base_lr * batch_size / reference


# ============================================================
# FACTEUR MULTIPLICATIF DU LEARNING RATE À L'ITÉRATION "step"
# ============================================================
# 1) Warmup : le facteur monte linéairement de presque 0 à 1.
#    On évite ainsi des mises à jour brutales au début, quand
#    les poids sont aléatoires (les Transformers y sont sensibles).
#
# 2) Décroissance cosinus : le facteur descend de 1 à min_ratio
#    en suivant une demi-cosinusoïde (douce au début et à la fin).
#
#    facteur = min_ratio + (1 - min_ratio) * 0.5 * (1 + cos(pi * p))
#    où p va de 0 à 1 pendant la phase de décroissance.
# ============================================================

def facteur_lr(step, warmup_steps, total_steps, min_ratio=0.0):

    assert 0 <= warmup_steps < total_steps
    assert 0.0 <= min_ratio <= 1.0

    # Phase de warmup.
    if step < warmup_steps:
        return (step + 1) / warmup_steps

    # Phase de décroissance cosinus.
    progression = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progression = min(1.0, progression)

    cosinus = 0.5 * (1.0 + math.cos(math.pi * progression))

    return min_ratio + (1.0 - min_ratio) * cosinus


# ============================================================
# SCHEDULER PYTORCH
# ============================================================
# À appeler UNE FOIS PAR ITÉRATION (donc après chaque batch,
# pas après chaque epoch) : scheduler.step()
# ============================================================

def creer_scheduler(optimizer, total_steps, warmup_steps, min_lr=1e-5):

    from torch.optim.lr_scheduler import LambdaLR

    base_lr = optimizer.param_groups[0]["lr"]
    min_ratio = min_lr / base_lr

    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: facteur_lr(
            step, warmup_steps, total_steps, min_ratio
        )
    )


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":

    # --- Mise à l'échelle -----------------------------------
    assert math.isclose(lr_mis_a_echelle(5e-4, 512), 5e-4)
    assert math.isclose(lr_mis_a_echelle(5e-4, 1024), 1e-3)
    assert math.isclose(lr_mis_a_echelle(5e-4, 128), 1.25e-4)

    print("lr pour batch 1024 :", lr_mis_a_echelle(5e-4, 1024))

    # --- Forme du schedule ----------------------------------
    total, warmup, min_ratio = 1000, 100, 0.01

    facteurs = [
        facteur_lr(t, warmup, total, min_ratio)
        for t in range(total)
    ]

    # Début : très petit, puis monte.
    assert facteurs[0] < 0.02
    assert all(
        facteurs[i] < facteurs[i + 1] for i in range(warmup - 1)
    )

    # Fin du warmup : on atteint 1.
    assert math.isclose(facteurs[warmup - 1], 1.0)
    assert math.isclose(facteurs[warmup], 1.0)

    # Ensuite : décroissance, jusqu'à (presque) min_ratio.
    assert all(
        facteurs[i] >= facteurs[i + 1]
        for i in range(warmup, total - 1)
    )
    assert abs(facteurs[-1] - min_ratio) < 1e-3

    # Au milieu de la décroissance : environ à mi-chemin.
    milieu = warmup + (total - warmup) // 2
    assert abs(facteurs[milieu] - (1 + min_ratio) / 2) < 0.01

    print("Facteur au début       :", facteurs[0])
    print("Facteur fin du warmup  :", facteurs[warmup])
    print("Facteur au milieu      :", facteurs[milieu])
    print("Facteur à la fin       :", facteurs[-1])

    # --- Test avec un vrai optimiseur PyTorch ---------------
    try:
        import torch

        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([param], lr=1e-3)
        scheduler = creer_scheduler(
            optimizer, total_steps=total, warmup_steps=warmup, min_lr=1e-5
        )

        lrs = []

        for _ in range(total):
            optimizer.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        assert max(lrs) <= 1e-3 + 1e-12
        assert abs(lrs[warmup] - 1e-3) < 1e-9
        assert abs(lrs[-1] - 1e-5) < 2e-6

        print("Test avec PyTorch réussi")

    except ImportError:
        print("PyTorch absent : test du scheduler ignoré")

    print("Tests du learning rate réussis !")
