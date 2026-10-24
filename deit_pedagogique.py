"""DeiT pédagogique : architecture et boucle d'apprentissage sans entraînement réel."""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MiniDeiT(nn.Module):
    def __init__(self, num_classes=10, image_size=32, patch_size=8,
                 embed_dim=96, num_heads=3, num_layers=2):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=0.0, activation="gelu", batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.class_head = nn.Linear(embed_dim, num_classes)
        self.dist_head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images):
        x = self.patch_embed(images)              # [B, D, 4, 4]
        x = x.flatten(2).transpose(1, 2)         # [B, 16, D]
        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        dist = self.dist_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, dist, x], dim=1)      # [B, 18, D]
        x = x + self.pos_embed
        x = self.norm(self.transformer(x))
        return self.class_head(x[:, 0]), self.dist_head(x[:, 1])


class SimpleTeacher(nn.Module):
    """Petit CNN illustratif ; NON pré-entraîné."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(16, num_classes)
        )

    def forward(self, images):
        return self.network(images)


def distillation_loss(class_logits, dist_logits, true_labels, teacher_logits):
    """Hard distillation avec deux têtes (DeiT)."""
    teacher_labels = teacher_logits.argmax(dim=1)
    loss_class = nn.functional.cross_entropy(class_logits, true_labels)
    loss_dist = nn.functional.cross_entropy(dist_logits, teacher_labels)
    return 0.5 * loss_class + 0.5 * loss_dist


def demo():
    torch.manual_seed(42)
    student = MiniDeiT()
    teacher = SimpleTeacher()
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    images = torch.randn(20, 3, 32, 32)       # Données fictives
    labels = torch.randint(0, 10, (20,))     # Labels fictifs
    loader = DataLoader(TensorDataset(images, labels), batch_size=5, shuffle=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    student.train()
    for epoch in range(2):
        for batch_images, batch_labels in loader:
            optimizer.zero_grad()
            class_logits, dist_logits = student(batch_images)
            with torch.no_grad():
                teacher_logits = teacher(batch_images)
            loss = distillation_loss(class_logits, dist_logits,
                                     batch_labels, teacher_logits)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/2 | loss du dernier batch = {loss.item():.4f}")

    student.eval()
    with torch.no_grad():
        class_logits, dist_logits = student(images[:2])
        probabilities = (class_logits.softmax(dim=1) +
                         dist_logits.softmax(dim=1)) / 2
    print("Forme des probabilités finales :", tuple(probabilities.shape))
    print("Note : le Teacher n'est pas pré-entraîné ; les résultats n'ont aucune valeur scientifique.")


if __name__ == "__main__":
    demo()
