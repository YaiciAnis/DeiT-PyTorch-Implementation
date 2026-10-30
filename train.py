
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

# pyrefly: ignore[missing-import]
from torchvision.models import (
    regnet_y_16gf,
    RegNet_Y_16GF_Weights
)

# Importer le Student depuis deit_model.py.
from deit_model import DeiTTinyDistilled


# ============================================================
# ÉTAPE 9 — TEACHER CNN : REGNETY-16GF
# ============================================================
# Le Teacher est un CNN pré-entraîné sur ImageNet-1k.
#
# Son rôle :
# - Recevoir une image.
# - Produire 1000 scores de classification.
# - Fournir des connaissances au Student.
#
# Pendant la distillation :
# - Les paramètres du Teacher restent gelés.
# - Seul le Student apprend.
#
# Entrée : (B, 3, 224, 224)
# Sortie : (B, 1000)
# ============================================================

class TeacherModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Charger les poids pré-entraînés ImageNet.
        weights = RegNet_Y_16GF_Weights.IMAGENET1K_V2

        self.model = regnet_y_16gf(weights=weights)

        # Geler tous les paramètres du Teacher.
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        # Placer le Teacher en mode évaluation.
        self.model.eval()

    def train(self, mode=True):

        # Le Teacher reste toujours en mode évaluation.
        super().train(False)
        return self

    def forward(self, images):

        # Aucun gradient n'est calculé pour le Teacher.
        with torch.no_grad():
            logits = self.model(images)

        return logits


# ============================================================
# ÉTAPE 10 — FONCTION DE PERTE DE DISTILLATION
# ============================================================
#
# Le Student possède deux sorties :
#
# 1. cls_logits  : (B, 1000)
# 2. dist_logits : (B, 1000)
#
# Le Teacher possède une sortie :
#
# 3. teacher_logits : (B, 1000)
#
# Les vraies étiquettes :
#
# 4. labels : (B,)
#
# Fonction de perte totale :
#
# L_total = (1 - alpha) * L_CLS + alpha * L_DIST
#
# Deux méthodes de distillation :
#
# HARD :
# Le Student apprend la classe prédite par le Teacher.
#
# SOFT :
# Le Student apprend la distribution de probabilités
# produite par le Teacher.
# ============================================================

class DistillationLoss(nn.Module):

    def __init__(
        self,
        distillation_type="hard",
        alpha=0.5,
        tau=1.0
    ):
        super().__init__()

        assert distillation_type in ["hard", "soft"]
        assert 0.0 <= alpha <= 1.0
        assert tau > 0.0

        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

        # Fonction de perte Cross-Entropy.
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        cls_logits,
        dist_logits,
        teacher_logits,
        labels
    ):

        # ----------------------------------------------------
        # ÉTAPE 10.1 — PERTE DE CLASSIFICATION CLS
        # ----------------------------------------------------
        # Comparer la prédiction CLS du Student
        # avec les vraies étiquettes.
        # ----------------------------------------------------

        loss_cls = self.cross_entropy(
            cls_logits,
            labels
        )

        # Empêcher toute propagation vers le Teacher.
        teacher_logits = teacher_logits.detach()

        # ----------------------------------------------------
        # ÉTAPE 10.2 — HARD DISTILLATION
        # ----------------------------------------------------

        if self.distillation_type == "hard":

            # Récupérer la classe prédite par le Teacher.
            teacher_labels = teacher_logits.argmax(dim=1)

            # Comparer DIST aux classes du Teacher.
            loss_dist = self.cross_entropy(
                dist_logits,
                teacher_labels
            )

        # ----------------------------------------------------
        # ÉTAPE 10.3 — SOFT DISTILLATION
        # ----------------------------------------------------

        else:

            # Calculer les log-probabilités du Student.
            student_log_probs = F.log_softmax(
                dist_logits / self.tau,
                dim=1
            )

            # Calculer les probabilités du Teacher.
            teacher_probs = F.softmax(
                teacher_logits / self.tau,
                dim=1
            )

            # Divergence KL entre Teacher et Student.
            loss_dist = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="batchmean"
            )

            # Correction liée à la température.
            loss_dist = loss_dist * (self.tau ** 2)

        # ----------------------------------------------------
        # ÉTAPE 10.4 — PERTE TOTALE
        # ----------------------------------------------------

        loss_total = (
            (1.0 - self.alpha) * loss_cls
            + self.alpha * loss_dist
        )

        return loss_total, loss_cls, loss_dist


# ============================================================
# ÉTAPE 12 — BOUCLE D'APPRENTISSAGE : BATCHES ET EPOCHS
# ============================================================
#
# Cette fonction montre le déroulement de l'apprentissage.
#
# Pour chaque epoch :
#   Pour chaque batch :
#       1. Calculer les prédictions du Student.
#       2. Calculer les prédictions du Teacher.
#       3. Calculer la fonction objective.
#       4. Réinitialiser les gradients.
#       5. Effectuer la rétropropagation.
#       6. Mettre à jour les paramètres du Student.
#
# Nous pouvons afficher les batches sans entraîner
# réellement le modèle, grâce au mode demonstration.
# ============================================================

def boucle_apprentissage(
    student,
    teacher,
    dataloader,
    loss_fn,
    optimizer,
    device,
    epochs=2,
    demonstration=True
):

    student.train()
    teacher.eval()

    for epoch in range(epochs):

        print(f"\n========== EPOCH {epoch + 1}/{epochs} ==========")

        for batch_index, (images, labels) in enumerate(dataloader):

            images = images.to(device)
            labels = labels.to(device)

            print(
                f"Batch {batch_index + 1}/{len(dataloader)} "
                f"| Images : {tuple(images.shape)} "
                f"| Labels : {tuple(labels.shape)}"
            )

            # ------------------------------------------------
            # MODE DÉMONSTRATION
            # ------------------------------------------------
            # Afficher les étapes sans effectuer de calculs
            # coûteux ni modifier les paramètres.
            # ------------------------------------------------

            if demonstration:

                print("  1. Student(images)")
                print("  2. Teacher(images)")
                print("  3. Calculer la loss")
                print("  4. optimizer.zero_grad()")
                print("  5. loss.backward()")
                print("  6. optimizer.step()")

                continue

            # ------------------------------------------------
            # MODE ENTRAÎNEMENT
            # ------------------------------------------------
            # Cette partie constitue une vraie boucle PyTorch.
            # Elle n'est pas exécutée en mode démonstration.
            # ------------------------------------------------

            # 1. Prédictions du Student.
            cls_logits, dist_logits = student(images)

            # 2. Prédictions du Teacher.
            with torch.no_grad():
                teacher_logits = teacher(images)

            # 3. Calcul de la fonction objective.
            loss_total, loss_cls, loss_dist = loss_fn(
                cls_logits,
                dist_logits,
                teacher_logits,
                labels
            )

            # 4. Réinitialisation des gradients.
            optimizer.zero_grad()

            # 5. Rétropropagation.
            loss_total.backward()

            # 6. Mise à jour des paramètres du Student.
            optimizer.step()

            print(f"  Loss totale : {loss_total.item():.4f}")


# ============================================================
# ÉTAPE 11 — TEST DE RÉTROPROPAGATION ET D'OPTIMISATION
# ============================================================
#
# Objectif :
# Vérifier qu'une itération d'entraînement fonctionne.
#
# 1. Initialiser Student et Teacher.
# 2. Créer une image fictive.
# 3. Calculer les prédictions.
# 4. Calculer la perte de distillation.
# 5. Calculer les gradients du Student.
# 6. Mettre à jour les paramètres du Student.
# 7. Vérifier que le Teacher reste inchangé.
#
# Attention :
# Ce test utilise une image et une étiquette fictives.
# Il ne constitue pas encore un entraînement réel.
# ============================================================

if __name__ == "__main__":

    # Pour rendre les résultats reproductibles.
    torch.manual_seed(42)

    # --------------------------------------------------------
    # ÉTAPE 11.1 — CHOISIR L'APPAREIL
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Appareil utilisé :", device)

    # --------------------------------------------------------
    # ÉTAPE 11.2 — INITIALISER LE STUDENT
    # --------------------------------------------------------

    student = DeiTTinyDistilled(
        num_classes=1000,
        drop_path_rate=0.1
    ).to(device)

    # Activer le mode entraînement.
    student.train()

    # --------------------------------------------------------
    # ÉTAPE 11.3 — INITIALISER LE TEACHER
    # --------------------------------------------------------

    teacher = TeacherModel().to(device)

    # Le Teacher reste en mode évaluation.
    teacher.eval()

    # --------------------------------------------------------
    # ÉTAPE 11.4 — INITIALISER LA FONCTION DE PERTE
    # --------------------------------------------------------
    # alpha = 0.5 :
    # - 50 % perte de classification CLS.
    # - 50 % perte de distillation DIST.
    # --------------------------------------------------------

    loss_fn = DistillationLoss(
        distillation_type="hard",
        alpha=0.5
    )

    # --------------------------------------------------------
    # ÉTAPE 11.5 — INITIALISER L'OPTIMISEUR
    # --------------------------------------------------------
    # AdamW modifie les paramètres du Student.
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=0.0001,
        weight_decay=0.05
    )

    # --------------------------------------------------------
    # ÉTAPE 11.6 — CRÉER UN BATCH FICTIF
    # --------------------------------------------------------

    images = torch.randn(
        1, 3, 224, 224,
        device=device
    )

    labels = torch.tensor(
        [5],
        device=device
    )

    print("Images :", images.shape)
    print("Labels :", labels.shape)

    # --------------------------------------------------------
    # ÉTAPE 11.7 — SAUVEGARDER LES POIDS AVANT OPTIMISATION
    # --------------------------------------------------------

    student_weights_before = (
        student.head.weight.detach().clone()
    )

    teacher_weights_before = (
        teacher.model.fc.weight.detach().clone()
    )

    # --------------------------------------------------------
    # ÉTAPE 11.8 — PROPAGATION AVANT DU STUDENT
    # --------------------------------------------------------
    # Ne pas utiliser torch.no_grad() ici.
    # Le Student doit construire un graphe de calcul.
    # --------------------------------------------------------

    cls_logits, dist_logits = student(images)

    print("CLS :", cls_logits.shape)
    print("DIST :", dist_logits.shape)

    # --------------------------------------------------------
    # ÉTAPE 11.9 — PROPAGATION AVANT DU TEACHER
    # --------------------------------------------------------

    with torch.no_grad():
        teacher_logits = teacher(images)

    print("Teacher :", teacher_logits.shape)

    # --------------------------------------------------------
    # ÉTAPE 11.10 — CALCULER LA PERTE
    # --------------------------------------------------------

    loss_total, loss_cls, loss_dist = loss_fn(
        cls_logits,
        dist_logits,
        teacher_logits,
        labels
    )

    print("\n===== HARD DISTILLATION =====")
    print("Loss CLS :", loss_cls.item())
    print("Loss DIST :", loss_dist.item())
    print("Loss totale :", loss_total.item())

    # --------------------------------------------------------
    # ÉTAPE 11.11 — VÉRIFIER LES SORTIES ET LES PERTES
    # --------------------------------------------------------

    assert cls_logits.shape == (1, 1000)
    assert dist_logits.shape == (1, 1000)
    assert teacher_logits.shape == (1, 1000)

    assert torch.isfinite(loss_total)
    assert torch.isfinite(loss_cls)
    assert torch.isfinite(loss_dist)

    assert torch.allclose(
        loss_total,
        0.5 * loss_cls + 0.5 * loss_dist
    )

    # --------------------------------------------------------
    # ÉTAPE 11.12 — RÉINITIALISER LES GRADIENTS
    # --------------------------------------------------------

    optimizer.zero_grad()

    # --------------------------------------------------------
    # ÉTAPE 11.13 — RÉTROPROPAGATION
    # --------------------------------------------------------

    loss_total.backward()

    print("\n===== RÉTROPROPAGATION =====")

    # Vérifier les gradients des deux têtes.
    assert student.head.weight.grad is not None
    assert student.head_dist.weight.grad is not None

    assert torch.isfinite(
        student.head.weight.grad
    ).all()

    assert torch.isfinite(
        student.head_dist.weight.grad
    ).all()

    print("Gradients CLS : calculés")
    print("Gradients DIST : calculés")

    # --------------------------------------------------------
    # ÉTAPE 11.14 — VÉRIFIER LE TEACHER
    # --------------------------------------------------------

    assert all(
        parameter.grad is None
        for parameter in teacher.parameters()
    )

    print("Gradients Teacher : aucun")

    # --------------------------------------------------------
    # ÉTAPE 11.15 — MISE À JOUR DES PARAMÈTRES
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # ÉTAPE 11.16 — COMPARER LES POIDS
    # --------------------------------------------------------

    student_weights_after = (
        student.head.weight.detach()
    )

    teacher_weights_after = (
        teacher.model.fc.weight.detach()
    )

    student_changed = not torch.equal(
        student_weights_before,
        student_weights_after
    )

    teacher_unchanged = torch.equal(
        teacher_weights_before,
        teacher_weights_after
    )

    print("\n===== VÉRIFICATION DES PARAMÈTRES =====")

    print(
        "Poids Student modifiés :",
        student_changed
    )

    print(
        "Poids Teacher inchangés :",
        teacher_unchanged
    )

    assert student_changed
    assert teacher_unchanged

    assert all(
        not parameter.requires_grad
        for parameter in teacher.parameters()
    )

    print("\nTests des étapes 9 à 11 réussis !")

    # ========================================================
    # ÉTAPE 12 — TEST DES BATCHES ET EPOCHS
    # ========================================================

    print("\n===== ÉTAPE 12 : BATCHES ET EPOCHS =====")

    # Créer 8 images fictives.
    images_dataset = torch.randn(
        8, 3, 224, 224
    )

    # Créer 8 étiquettes fictives.
    labels_dataset = torch.randint(
        low=0,
        high=1000,
        size=(8,)
    )

    # Associer chaque image à son étiquette.
    dataset = TensorDataset(
        images_dataset,
        labels_dataset
    )

    # Diviser les 8 images en batches de 4.
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )

    print("Nombre d'images :", len(dataset))
    print("Taille d'un batch :", dataloader.batch_size)
    print("Nombre de batches :", len(dataloader))

    # Vérifications automatiques.
    assert len(dataset) == 8
    assert len(dataloader) == 2

    # Montrer le déroulement de 2 epochs.
    boucle_apprentissage(
        student=student,
        teacher=teacher,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=2,
        demonstration=True
    )

    print("\nTest de l'étape 12 réussi !")
