
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet18,
    regnet_y_16gf,
    RegNet_Y_16GF_Weights
)


# ============================================================
# ÉTAPE 5 — PROFESSEURS (CONVNETS) POUR CIFAR-10
# ============================================================
# Le professeur actuel de train.py (RegNetY ImageNet) sort 1000
# classes : il ne peut pas servir tel quel pour CIFAR-10 (10 classes).
#
# Deux professeurs, comme décidé :
#
# 1. ResNet-18 adapté aux petites images, entraîné sur CIFAR-10
#    (images 32 x 32, comme l'étudiant).
#
# 2. RegNetY-16GF (le professeur du papier), pré-entraîné sur
#    ImageNet puis ajusté (fine-tuning) sur CIFAR-10. Il exige des
#    images 224 x 224 : on agrandit donc les images CIFAR-10.
#
# L'entraînement de ces professeurs se fera à l'étape 6.
# ============================================================


# ------------------------------------------------------------
# PROFESSEUR 1 : RESNET-18 ADAPTÉ À CIFAR-10
# ------------------------------------------------------------
# Le ResNet-18 standard est conçu pour des images de 224 x 224 :
# sa première couche réduit immédiatement la résolution (conv 7x7
# de pas 2, puis max-pooling), ce qui détruirait des images 32 x 32.
# On remplace donc :
# - la première convolution par une 3 x 3 de pas 1,
# - le max-pooling par l'identité (on ne réduit plus).
# ------------------------------------------------------------

def creer_resnet18_cifar(num_classes=10):

    model = resnet18(weights=None, num_classes=num_classes)

    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.maxpool = nn.Identity()

    return model


# ------------------------------------------------------------
# PROFESSEUR 2 : REGNETY-16GF POUR CIFAR-10
# ------------------------------------------------------------
# On garde tout le réseau pré-entraîné sur ImageNet et on remplace
# uniquement la dernière couche (1000 sorties -> num_classes).
# Cette nouvelle couche est aléatoire : il faudra faire un
# fine-tuning avant de s'en servir comme professeur.
# ------------------------------------------------------------

def creer_regnet_cifar(num_classes=10, pretrained=True):

    weights = (
        RegNet_Y_16GF_Weights.IMAGENET1K_V2
        if pretrained else None
    )

    model = regnet_y_16gf(weights=weights)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# ------------------------------------------------------------
# AGRANDISSEMENT DES IMAGES (pour le RegNetY)
# ------------------------------------------------------------
# Le RegNetY attend du 224 x 224, alors que l'étudiant et le
# ResNet travaillent en 32 x 32. Ce module agrandit l'image
# (interpolation bilinéaire) avant de la donner au réseau.
# ------------------------------------------------------------

class AvecAgrandissement(nn.Module):

    def __init__(self, model, taille=224):
        super().__init__()

        self.model = model
        self.taille = taille

    def forward(self, images):

        images = F.interpolate(
            images,
            size=(self.taille, self.taille),
            mode="bilinear",
            align_corners=False
        )

        return self.model(images)


# ------------------------------------------------------------
# PROFESSEUR FIGÉ POUR LA DISTILLATION
# ------------------------------------------------------------
# Une fois le professeur entraîné, on le fige : ses poids ne
# changent plus et il reste en mode évaluation (comme TeacherModel
# dans train.py). Il fournit seulement les logits qui servent
# à calculer le label dur (argmax) pour la tête de distillation.
# ------------------------------------------------------------

class ProfesseurFige(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.model = model

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.eval()

    def train(self, mode=True):

        # Le professeur reste toujours en mode évaluation.
        super().train(False)
        return self

    def forward(self, images):

        with torch.no_grad():
            return self.model(images)


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    images = torch.randn(2, 3, 32, 32)

    # --- ResNet-18 adapté ------------------------------------
    resnet = creer_resnet18_cifar(num_classes=10)
    resnet.eval()

    with torch.no_grad():
        sortie = resnet(images)

    print("ResNet-18 CIFAR :", sortie.shape)

    assert sortie.shape == (2, 10)
    assert isinstance(resnet.maxpool, nn.Identity)
    assert resnet.conv1.kernel_size == (3, 3)

    # --- RegNetY (sans télécharger les poids pour ce test) ---
    regnet = AvecAgrandissement(
        creer_regnet_cifar(num_classes=10, pretrained=False),
        taille=224
    )
    regnet.eval()

    with torch.no_grad():
        sortie = regnet(images)

    print("RegNetY CIFAR   :", sortie.shape)

    assert sortie.shape == (2, 10)

    # --- Professeur figé ------------------------------------
    prof = ProfesseurFige(resnet)

    prof.train()
    assert not prof.model.training
    assert all(not p.requires_grad for p in prof.parameters())

    logits = prof(images)
    assert logits.shape == (2, 10)
    assert logits.argmax(dim=1).shape == (2,)

    print("Tests des professeurs CIFAR-10 réussis !")
