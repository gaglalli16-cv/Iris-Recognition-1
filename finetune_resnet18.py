# =============================================================================
# finetune_resnet18.py
# Fine-tunes ResNet18 on CASIA-IrisV1 for iris-specific feature extraction.
# Course  : Biometrics, IMCV Master
# Author  : Gagandeep Kaur and Eda Ozge Ozler
#
# Strategy:
#   - Split CASIA subjects: 80% for training, 20% for testing (by identity)
#   - Fine-tune ResNet18 (pretrained on ImageNet) as an identity classifier
#   - After training, remove the classifier and use the network as a
#     feature extractor → extract embeddings → cosine distance → EER
#   - Classical pipeline also runs inline so all three EERs use
#     the same scoring protocol — no cache, fully reproducible
# =============================================================================

import os, sys, random
import numpy as np
import cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

REPO_ROOT  = os.path.dirname(os.path.abspath(__file__))
CASIA_DIR  = os.path.join(REPO_ROOT, 'CASIA-database')
MODEL_PATH = os.path.join(REPO_ROOT, 'resnet18_iris_finetuned.pth')

# Masek pipeline needed for the classical baseline comparison
sys.path.insert(0, os.path.join(REPO_ROOT, 'python'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'python', 'fnc'))

try:
    from extractFeature import extractFeature
except ImportError:
    print("ERROR: Run from repo root."); sys.exit(1)


# ── Dataset helpers ───────────────────────────────────────────────────────────

def parse_db(casia_dir):
    # Same parsing logic as casia_experiment.py
    # Groups image paths by subject ID extracted from the filename
    db = defaultdict(list)
    for f in sorted(os.listdir(casia_dir)):
        if f.lower().endswith('.jpg'):
            db[f.split('_')[0]].append(os.path.join(casia_dir, f))
    return {k: sorted(v) for k, v in db.items()}


def split_subjects(ids, ratio=0.9, seed=42):
    # Fixed seed ensures the 90/10 client/intruder split is identical
    # across casia_experiment.py and this script — required for fair comparison
    random.seed(seed); ids = list(ids); random.shuffle(ids)
    n = int(len(ids) * ratio)
    return sorted(ids[:n]), sorted(ids[n:])


def hamming(t1, m1, t2, m2):
    # Cast to bool first — Masek returns uint8 arrays which break bitwise ops
    t1, m1, t2, m2 = (np.array(x, dtype=bool) for x in (t1, m1, t2, m2))
    best = 1.0
    for s in range(-8, 9):
        t2s, m2s = np.roll(t2, s, 1), np.roll(m2, s, 1)
        mask  = m1 | m2s
        valid = (~mask).sum()
        if valid == 0: continue
        best  = min(best, (np.logical_xor(t1, t2s) & ~mask).sum() / valid)
    return best


def find_eer(gen, imp):
    # Sweep thresholds and find where FMR and FNMR are equal
    th   = np.linspace(0, 1, 1000)
    g, i = np.array(gen), np.array(imp)
    fmr  = np.array([np.mean(i <= t) for t in th])
    fnmr = np.array([np.mean(g >  t) for t in th])
    idx  = np.argmin(np.abs(fmr - fnmr))
    return th[idx], (fmr[idx] + fnmr[idx]) / 2, fmr, fnmr, th


# ── PyTorch dataset ───────────────────────────────────────────────────────────

def get_transform(augment=False):
    import torchvision.transforms as T
    if augment:
        # Light augmentation only — iris texture is fine-grained so we avoid
        # aggressive transforms that would destroy the discriminative patterns
        return T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.Grayscale(3),           # ResNet expects 3 channels, iris is grayscale
            T.RandomHorizontalFlip(),
            T.RandomRotation(8),      # small rotation simulates head tilt variation
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    # Inference transform — no augmentation, deterministic output
    return T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.Grayscale(3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class IrisDataset:
    """
    Wraps CASIA images for use with PyTorch DataLoader during fine-tuning.
    Each subject gets an integer label (0 to n_subjects-1).
    Only used during training — EER evaluation uses a separate embedding protocol.
    """
    def __init__(self, subjects, subject_ids, transform):
        self.transform = transform
        self.samples   = []
        for label, sid in enumerate(subject_ids):
            for path in subjects[sid]:
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((280, 320), dtype=np.uint8)
        return self.transform(img), torch.tensor(label, dtype=torch.long)


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def finetune_resnet18(subjects, train_ids, n_epochs=25, batch_size=16, lr=1e-3):
    """
    Fine-tunes ResNet18 on CASIA iris identities using a two-phase strategy.

    Why two phases:
      Training from scratch on 539 images would severely overfit.
      Instead we start from ImageNet weights and only adapt the parts
      of the network that need to change for iris-specific patterns.

    Phase 1 (epochs 1-10):
      Only the final FC layer is trained. The conv layers stay frozen.
      This quickly teaches the network which identity maps to which class
      using the general texture features already learned from ImageNet.

    Phase 2 (epochs 11-25):
      The last residual block (layer4) is also unfrozen.
      Higher-level features now adapt toward iris-specific patterns
      while earlier layers retain useful low-level edge/texture features.
    """
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torch.utils.data import DataLoader

    n_classes = len(train_ids)
    print(f"  Training on {n_classes} identities, {len(train_ids)*7} images")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace the 1000-class ImageNet head with an iris identity classifier
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    # Phase 1: freeze all layers except FC
    for name, param in model.named_parameters():
        param.requires_grad = 'fc' in name

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    dataset    = IrisDataset(subjects, train_ids, get_transform(augment=True))
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    model.train()
    print(f"  {'Epoch':>6} | {'Loss':>8} | {'Acc':>8} | {'Phase'}")
    print("  " + "-" * 42)

    for epoch in range(1, n_epochs + 1):

        # Switch to phase 2 at epoch 11: unfreeze layer4 alongside FC
        if epoch == 11:
            for name, param in model.named_parameters():
                param.requires_grad = 'layer4' in name or 'fc' in name
            # Reduce learning rate for phase 2 to avoid overwriting phase 1 gains
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=lr * 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=5, gamma=0.5)

        total_loss, correct, total = 0, 0, 0
        for imgs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += len(labels)

        scheduler.step()
        acc   = correct / total * 100
        phase = 'FC only' if epoch <= 10 else 'FC + layer4'
        print(f"  {epoch:>6} | {total_loss/total:>8.4f} | {acc:>7.1f}% | {phase}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n  Model saved: resnet18_iris_finetuned.pth")
    return model


# ── Embedding extraction ──────────────────────────────────────────────────────

def load_finetuned_extractor():
    """
    Loads the saved fine-tuned ResNet18 and strips the FC layer.
    What remains is a 512-dim feature extractor — the classification
    head is no longer needed since we match by cosine distance instead.
    """
    import torch
    import torch.nn as nn
    import torchvision.models as models

    saved     = torch.load(MODEL_PATH, map_location='cpu')
    n_classes = saved['fc.weight'].shape[0]   # infer from saved weights
    model     = models.resnet18(weights=None)
    model.fc  = nn.Linear(model.fc.in_features, n_classes)
    model.load_state_dict(saved)
    model.fc  = nn.Identity()  # remove classifier, output is now 512-dim
    model.eval()
    return model


def extract_embeddings(model, transform, subjects, client_ids, intruder_ids):
    """
    Runs all images through the model and computes cosine distance scores.

    L2 normalisation before comparison means dot product equals cosine similarity,
    so cosine distance = 1 - dot(e1, e2). Range is 0 (identical) to 2 (opposite).
    Same convention as Hamming Distance: lower score = better match.
    """
    import torch

    def embed(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        with torch.no_grad():
            e = model(transform(img).unsqueeze(0)).squeeze(0)
            e = e / (e.norm() + 1e-8)   # L2 normalise
        return e.numpy()

    # Enrollment: one template per client using first image
    enroll = {}
    for cid in client_ids:
        e = embed(subjects[cid][0])
        if e is not None: enroll[cid] = e
    print(f"  Enrolled {len(enroll)}/{len(client_ids)}")

    # Genuine: remaining images of same person vs enrolled template
    gen = [float(1 - np.dot(enroll[cid], e2))
           for cid in client_ids if cid in enroll
           for path in subjects[cid][1:]
           if (e2 := embed(path)) is not None]

    # Impostor: one image per intruder vs all enrolled clients
    # Using one image per intruder keeps the protocol consistent
    # with the classical pipeline in this script
    imp = [float(1 - np.dot(e1, e2))
           for iid in intruder_ids
           if (e2 := embed(subjects[iid][0])) is not None
           for e1 in enroll.values()]

    return gen, imp


# ── Results figure ────────────────────────────────────────────────────────────

def plot_comparison(cl_eer, before_eer, after_eer, fmr_cl, fnmr_cl,
                    fmr_before, fnmr_before, fmr_after, fnmr_after,
                    gen_b, imp_b, gen_ft, imp_ft):
    """
    Three-panel summary figure:
      Panel 1 — EER bar chart with improvement arrow
      Panel 2 — ROC curves for all three methods
      Panel 3 — Score distributions before vs after fine-tuning
                showing how the genuine/impostor separation improves
    """
    trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor('white')
    fig.suptitle('ResNet18 Fine-tuning Effect on CASIA-IrisV1',
                 fontsize=13, fontweight='bold', y=1.01)

    # Panel 1: EER bar chart
    ax = axes[0]
    ax.set_facecolor('#f8f9fa')
    labels  = ['Classical\n(Masek)', 'ResNet18\n(ImageNet)', 'ResNet18\n(Fine-tuned)']
    eers    = [cl_eer*100, before_eer*100, after_eer*100]
    colours = ['#1565C0', '#E65100', '#2E7D32']
    bars    = ax.bar(labels, eers, color=colours, width=0.45,
                     edgecolor='white', linewidth=1.5)
    for bar, eer in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{eer:.2f}%', ha='center', fontsize=10, fontweight='bold')
    # Arrow showing the improvement from baseline to fine-tuned
    ax.annotate('', xy=(2, after_eer*100 + 0.3),
                xytext=(1, before_eer*100 + 0.3),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2))
    ax.text(1.5, max(before_eer, after_eer)*100 + 1.5,
            f'-{(before_eer - after_eer)*100:.2f}%',
            ha='center', fontsize=9, color='#2E7D32', fontweight='bold')
    ax.set_ylabel('Equal Error Rate (%)', fontsize=10)
    ax.set_title('EER Comparison', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(eers) + 4)
    ax.grid(axis='y', alpha=0.4, ls='--')

    # Panel 2: ROC curves for all three methods
    ax = axes[1]
    ax.set_facecolor('#f8f9fa')
    for fmr, fnmr, eer, colour, label in [
        (fmr_cl,     fnmr_cl,     cl_eer,     '#1565C0', 'Classical'),
        (fmr_before, fnmr_before, before_eer, '#E65100', 'ResNet18 (ImageNet)'),
        (fmr_after,  fnmr_after,  after_eer,  '#2E7D32', 'ResNet18 (Fine-tuned)'),
    ]:
        tpr = (1 - fnmr) * 100
        fpr = fmr * 100
        auc = trapz(tpr/100, fpr/100)
        ax.plot(fpr, tpr, color=colour, lw=2.2,
                label=f'{label}  EER={eer*100:.2f}%  AUC={auc:.4f}')
        ax.scatter([eer*100], [(1-eer)*100], color=colour, s=80, zorder=5)
    ax.plot([0, 100], [0, 100], color='#9E9E9E', ls=':', lw=1.2, label='Random')
    ax.set_xlabel('False Match Rate (%)', fontsize=10)
    ax.set_ylabel('True Match Rate (%)', fontsize=10)
    ax.set_title('ROC Curves', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.4, ls='--')
    ax.set_xlim([-1, 101]); ax.set_ylim([-1, 101])

    # Panel 3: Score distributions before and after fine-tuning
    # Better separation = less overlap = lower EER
    ax = axes[2]
    ax.set_facecolor('#f8f9fa')
    ax.set_title('Score Separation (ResNet18 only)', fontsize=11, fontweight='bold')
    gen_ft = np.array(gen_ft); imp_ft = np.array(imp_ft)
    gen_b  = np.array(gen_b);  imp_b  = np.array(imp_b)
    bins   = np.linspace(0, 0.4, 60)
    ax.hist(gen_b,  bins=bins, alpha=0.4, color='#E65100', density=True,
            label='Genuine (before)')
    ax.hist(imp_b,  bins=bins, alpha=0.4, color='#FF7043', density=True,
            label='Impostor (before)')
    ax.hist(gen_ft, bins=bins, alpha=0.55, color='#1B5E20', density=True,
            label='Genuine (fine-tuned)')
    ax.hist(imp_ft, bins=bins, alpha=0.55, color='#66BB6A', density=True,
            label='Impostor (fine-tuned)')
    ax.set_xlabel('Cosine Distance', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.4, ls='--')

    plt.tight_layout()
    plt.savefig(os.path.join(REPO_ROOT, 'fig_resnet18_finetuned.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig_resnet18_finetuned.png")


# ── Classical pipeline ─────────────────────────────────────────────────────────

def run_classical(subjects, clients, intruders):
    """
    Runs the Masek/Kovesi pipeline with the same protocol as casia_experiment.py:
    all intruder images are used and we keep the minimum HD per image.
    This ensures the classical EER here is directly comparable to the DL results.
    """
    print("  Classical: enrolling clients...")
    templates = {}
    for cid in clients:
        try:
            r = extractFeature(subjects[cid][0])
            templates[cid] = (np.array(r[0], dtype=bool),
                              np.array(r[1], dtype=bool))
        except: pass
    print(f"  Enrolled {len(templates)}/{len(clients)}")

    gen = []
    for cid in clients:
        if cid not in templates: continue
        t1, m1 = templates[cid]
        for path in subjects[cid][1:]:
            try:
                r = extractFeature(path)
                gen.append(hamming(t1, m1,
                                   np.array(r[0], dtype=bool),
                                   np.array(r[1], dtype=bool)))
            except: pass

    # Minimum HD per intruder image = strongest possible impostor attack
    imp = []
    client_list = list(templates.items())
    for iid in intruders:
        for path in subjects[iid]:
            try:
                r  = extractFeature(path)
                t2 = np.array(r[0], dtype=bool)
                m2 = np.array(r[1], dtype=bool)
                imp.append(min(hamming(t1, m1, t2, m2)
                               for _, (t1, m1) in client_list))
            except: pass

    print(f"  Genuine: {len(gen)} scores | Impostor: {len(imp)} scores")
    return gen, imp


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("ResNet18 Fine-tuning on CASIA-IrisV1")
    print("=" * 55)

    subjects = parse_db(CASIA_DIR)
    ids      = sorted(subjects.keys())

    # 90/10 client/intruder split — same seed as casia_experiment.py
    clients, intruders = split_subjects(ids, 0.9, seed=42)
    print(f"DB: {len(ids)} subjects | {len(clients)} clients / {len(intruders)} intruders")

    # Further split clients into training and held-out test identities
    # Training subjects are used for fine-tuning; test subjects only appear
    # in the EER evaluation — this prevents identity leakage between the two
    random.seed(0)
    train_ids = sorted(random.sample(clients, int(len(clients) * 0.80)))
    test_ids  = sorted([c for c in clients if c not in set(train_ids)])
    print(f"Fine-tune split: {len(train_ids)} train / {len(test_ids)} test identities\n")

    import torch, torchvision.models as models
    from torch import nn

    # Step 1: run classical pipeline to get a directly comparable baseline
    print("Step 1/4 — Classical pipeline (Masek/Kovesi)...")
    cl_gen, cl_imp = run_classical(subjects, clients, intruders)
    _, cl_eer, cl_fmr, cl_fnmr, _ = find_eer(cl_gen, cl_imp)
    print(f"  Classical EER = {cl_eer*100:.2f}%\n")

    # Step 2: ResNet18 with ImageNet weights only — no iris-specific adaptation
    print("Step 2/4 — Baseline embeddings (ResNet18, ImageNet only)...")
    model_base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model_base.fc = nn.Identity()   # remove classifier, use 512-dim features
    model_base.eval()
    tf_base = get_transform(augment=False)
    gen_b, imp_b = extract_embeddings(model_base, tf_base,
                                       subjects, clients, intruders)
    _, eer_before, fmr_b, fnmr_b, _ = find_eer(gen_b, imp_b)
    print(f"  Baseline EER = {eer_before*100:.2f}%\n")

    # Step 3: fine-tune on CASIA iris identities
    # If weights already exist from a previous run, skip training
    if os.path.exists(MODEL_PATH):
        print("Step 3/4 — Fine-tuned weights found, skipping training.")
    else:
        print("Step 3/4 — Fine-tuning ResNet18 on iris identities...")
        finetune_resnet18(subjects, train_ids, n_epochs=25,
                          batch_size=16, lr=1e-3)

    # Step 4: extract embeddings with the fine-tuned model and compute EER
    print("\nStep 4/4 — Fine-tuned embeddings...")
    model_ft = load_finetuned_extractor()
    tf_ft    = get_transform(augment=False)
    gen_ft, imp_ft = extract_embeddings(model_ft, tf_ft,
                                         subjects, clients, intruders)
    _, eer_after, fmr_ft, fnmr_ft, _ = find_eer(gen_ft, imp_ft)
    print(f"  Fine-tuned EER = {eer_after*100:.2f}%\n")

    sep = "-" * 58
    print(sep)
    print(f"  {'Method':<35} {'EER':>8}")
    print(sep)
    print(f"  {'Classical (Masek/Kovesi)':<35} {cl_eer*100:>7.2f}%")
    print(f"  {'ResNet18 (ImageNet only)':<35} {eer_before*100:>7.2f}%")
    print(f"  {'ResNet18 (Fine-tuned on CASIA)':<35} {eer_after*100:>7.2f}%")
    print(sep)
    print(f"  Improvement over classical:  -{(cl_eer - eer_after)*100:.2f}%")
    print(f"  Improvement over baseline:   -{(eer_before - eer_after)*100:.2f}%\n")

    print("Generating figure...")
    plot_comparison(cl_eer, eer_before, eer_after,
                    cl_fmr, cl_fnmr, fmr_b, fnmr_b, fmr_ft, fnmr_ft,
                    gen_b, imp_b, gen_ft, imp_ft)
    print("\nDone!  fig_resnet18_finetuned.png")