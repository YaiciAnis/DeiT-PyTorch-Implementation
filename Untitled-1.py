
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# 1. STOCHASTIC DEPTH / DROP PATH

class DropPath(nn.Module):
    """Ignore aleatoirement une branche residuelle, par image, au training."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob doit etre dans [0, 1[.")
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )).floor()
        return x * mask / keep_prob


# 2. PATCH EMBEDDING

class PatchEmbed(nn.Module):
    """Image [B, 3, 224, 224] -> sequence [B, 196, D], si patch_size=16."""

    def __init__(self, image_size=224, patch_size=16,
                 in_channels=3, embed_dim=384):
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size != 0:
            raise ValueError("image_size doit etre positif et divisible par patch_size.")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError("Forme attendue : [batch, canaux, hauteur, largeur].")
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Images attendues : {self.image_size} x {self.image_size}. "
                "Cette version utilise une resolution fixe."
            )
        x = self.proj(images)          # [B, D, 14, 14]
        return x.flatten(2).transpose(1, 2)  # [B, 196, D]


# 3. ATTENTION MULTI-HEAD

class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("dim doit etre divisible par num_heads > 0.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(scores.softmax(dim=-1))
        x = attention @ v
        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(x))


# 4. MLP / FEED-FORWARD NETWORK

class MLP(nn.Module):
    """Applique D -> hidden_dim -> D independamment a chaque token."""

    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


# 5. BLOC TRANSFORMER

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# 6. STUDENT DEIT : CLS + DIST + DEUX CLASSIFIEURS

class DeiT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=384, depth=12, num_heads=6,
                 mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path_rate=0.1):
        super().__init__()
        if depth < 1 or num_classes < 2:
            raise ValueError("Utiliser au moins un bloc et deux classes.")
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_embed = PatchEmbed(
            image_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=rates[i]
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.class_head = nn.Linear(embed_dim, num_classes)
        self.dist_head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

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

    def forward_features(self, images):
        batch_size = images.shape[0]
        patches = self.patch_embed(images)
        cls = self.cls_token.expand(batch_size, -1, -1)
        dist = self.dist_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, dist, patches], dim=1)  # [B, 198, D]
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, images):
        cls_final, dist_final = self.forward_features(images)
        return self.class_head(cls_final), self.dist_head(dist_final)

    @torch.no_grad()
    def predict_proba(self, images):
        """Appeler model.eval() avant cette methode : moyenne des logits puis softmax."""
        class_logits, dist_logits = self(images)
        final_logits = (class_logits + dist_logits) / 2.0
        return final_logits.softmax(dim=1)


# 7. TEACHER ILLUSTRATIF, NON PRE-ENTRAINE

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


# 8. LABEL SMOOTHING

class LabelSmoothingCrossEntropy(nn.Module):
    """Cible lissee : (1 - epsilon) * one_hot(y) + epsilon / K."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing doit etre dans [0, 1[.")
        self.smoothing = smoothing

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        uniform_loss = -log_probs.mean(dim=1)
        return ((1.0 - self.smoothing) * nll
                + self.smoothing * uniform_loss).mean()


# 9. LOSSES : HARD OU SOFT DISTILLATION

def hard_distillation_loss(class_logits, dist_logits, true_labels,
                           teacher_logits, smoothing=0.1):
    """L = 0.5 * CE_lissee(CLS, y) + 0.5 * CE(DIST, argmax(teacher))."""
    loss_cls = LabelSmoothingCrossEntropy(smoothing)(class_logits, true_labels)
    teacher_labels = teacher_logits.detach().argmax(dim=1)
    loss_dist = F.cross_entropy(dist_logits, teacher_labels)
    return 0.5 * loss_cls + 0.5 * loss_dist


def soft_distillation_loss(class_logits, dist_logits, true_labels,
                           teacher_logits, tau=3.0, lam=0.1, smoothing=0.1):
    """L = (1-lam) * CE_lissee(CLS, y) + lam * tau^2 * KL(teacher || DIST)."""
    if tau <= 0.0 or not 0.0 <= lam <= 1.0:
        raise ValueError("Utiliser tau > 0 et lam dans [0, 1].")
    loss_cls = LabelSmoothingCrossEntropy(smoothing)(class_logits, true_labels)
    student_log_probs = F.log_softmax(dist_logits / tau, dim=1)
    teacher_probs = F.softmax(teacher_logits.detach() / tau, dim=1)
    loss_kl = F.kl_div(
        student_log_probs, teacher_probs, reduction="batchmean"
    ) * tau ** 2
    return (1.0 - lam) * loss_cls + lam * loss_kl


# 10. EXPONENTIAL MOVING AVERAGE

class ModelEMA:
    """Copie lissee du student ; aucune backpropagation dans cette copie."""

    def __init__(self, model, decay=0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay doit etre dans [0, 1[.")
        self.ema_model = copy.deepcopy(model).eval()
        self.ema_model.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        student_state = model.state_dict()
        for name, ema_value in self.ema_model.state_dict().items():
            student_value = student_state[name].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(self.decay).add_(student_value, alpha=1.0 - self.decay)
            else:
                # Les compteurs entiers eventuels doivent etre copies directement.
                ema_value.copy_(student_value)


# 11. ENTRAINEMENT D'UNE EPOQUE

def train_one_epoch(student, teacher, dataloader, optimizer, device,
                    use_soft_distillation=False, tau=3.0, lam=0.1,
                    smoothing=0.1, ema=None):
    student.train()
    teacher.eval()
    total_loss = 0.0
    total_images = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        class_logits, dist_logits = student(images)
        with torch.no_grad():
            teacher_logits = teacher(images)

        if use_soft_distillation:
            loss = soft_distillation_loss(
                class_logits, dist_logits, labels, teacher_logits,
                tau=tau, lam=lam, smoothing=smoothing
            )
        else:
            loss = hard_distillation_loss(
                class_logits, dist_logits, labels, teacher_logits,
                smoothing=smoothing
            )

        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(student)

        total_loss += loss.item() * labels.size(0)
        total_images += labels.size(0)

    if total_images == 0:
        raise ValueError("Le dataloader d'entrainement est vide.")
    return total_loss / total_images


# 12. EVALUATION

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model.predict_proba(images).argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
    if total == 0:
        raise ValueError("Le dataloader d'evaluation est vide.")
    return correct / total


# 13. DEMONSTRATION COMPLETE A RESOLUTION FIXE

def demo(num_epochs=10, use_soft_distillation=False):
    """Donnees fictives et teacher aleatoire ; aucune phase de fine-tuning."""
    if num_epochs < 1:
        raise ValueError("Utiliser au moins une epoque.")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    batch_size = 4
    print(f"Appareil : {device}")
    print("Distillation :", "soft" if use_soft_distillation else "hard")

    # Student reduit pour la demonstration : 4 blocs au lieu de 12.
    student = DeiT(
        image_size=224, patch_size=16, num_classes=num_classes,
        embed_dim=192, depth=4, num_heads=3, drop_path_rate=0.1
    ).to(device)

    # Un teacher gele n'est pas necessairement pre-entraine : ici il est aleatoire.
    teacher = SimpleTeacherCNN(num_classes).to(device)
    teacher.eval()
    teacher.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=1e-4, weight_decay=0.05
    )
    ema = ModelEMA(student, decay=0.999)

    images = torch.randn(20, 3, 224, 224)
    labels = torch.randint(0, num_classes, (20,))
    dataset = TensorDataset(images, labels)
    # num_workers=0 facilite l'execution de cette petite demo sur Windows.
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    eval_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    for epoch in range(num_epochs):
        loss = train_one_epoch(
            student, teacher, train_loader, optimizer, device,
            use_soft_distillation=use_soft_distillation,
            tau=3.0, lam=0.1, smoothing=0.1, ema=ema
        )
        print(f"Epoch {epoch + 1}/{num_epochs} | loss = {loss:.4f}")

    # Ici, ces scores sont calcules sur les memes donnees fictives que le training.
    acc = evaluate(student, eval_loader, device)
    acc_ema = evaluate(ema.ema_model, eval_loader, device)
    print(f"Accuracy student (entrainement fictif) = {acc:.4f}")
    print(f"Accuracy EMA (entrainement fictif) = {acc_ema:.4f}")

    student.eval()
    probabilities = student.predict_proba(images[:2].to(device))
    print("Forme des probabilites :", tuple(probabilities.shape))
    print("Classes predites :", probabilities.argmax(dim=1).cpu().tolist())
    print("Note : donnees fictives et teacher non pre-entraine ; scores illustratifs.")
    print("Fin de la demonstration, sans fine-tuning.")


if __name__ == "__main__":
    # Mettre use_soft_distillation=True pour essayer la version soft.
    demo(num_epochs=10, use_soft_distillation=False)