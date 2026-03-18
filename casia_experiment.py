# =============================================================================
# CASIA-IrisV1 — Iris Recognition Experiment
# Course  : Biometrics, IMCV Master
# System  : Masek/Kovesi Python port (github.com/yuxiwang66/Iris-Recognition-1)
# Author  : Gagandeep Kaur and Eda Ozge Ozler
#
# Experiment protocol:
#   - 90/10 client/intruder split (seed=42)
#   - Enrollment: first image per client
#   - Genuine scores : images 2-7 vs enrolled template (same person)
#   - Impostor scores: intruder images vs all client templates (min distance)
#   - Performance metric: Equal Error Rate (EER)
# =============================================================================

import os, sys, random
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CASIA_DIR = os.path.join(REPO_ROOT, 'CASIA-database')

# Masek pipeline functions live in the python/ subfolder of the repo
sys.path.insert(0, os.path.join(REPO_ROOT, 'python'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'python', 'fnc'))
from extractFeature import extractFeature


# ── Database ──────────────────────────────────────────────────────────────────

def parse_database(casia_dir):
    # Filename format: 001_1_1.jpg
    # Index 0 after split('_') gives the subject ID: '001'
    # All images for the same subject are grouped into one list
    subjects = defaultdict(list)
    for f in sorted(os.listdir(casia_dir)):
        if f.endswith('.jpg'):
            subjects[f.split('_')[0]].append(os.path.join(casia_dir, f))
    for sid in subjects:
        subjects[sid].sort()
    return dict(subjects)


def split_subjects(ids, ratio=0.9, seed=42):
    # Fixing the seed guarantees the same partition every run
    # so results are reproducible and comparable across experiments
    random.seed(seed)
    ids = list(ids)
    random.shuffle(ids)
    n = int(len(ids) * ratio)
    return sorted(ids[:n]), sorted(ids[n:])


# ── Matching ──────────────────────────────────────────────────────────────────

def hamming_distance(t1, m1, t2, m2):
    """
    Masked Hamming Distance with rotation compensation.

    IrisCodes are binary arrays. HD counts the fraction of bits that differ.
    HD = 0.0 means identical templates; HD ~ 0.5 means unrelated irises
    (converges to random coin-flip probability for different people).

    Two practical adjustments:
    1. Noise mask: bits corrupted by eyelashes or reflections are excluded
       from the count so they don't inflate the distance unfairly.
    2. Rotation: a small head tilt shifts the IrisCode horizontally.
       We try 17 column shifts (-8 to +8) and keep the minimum distance,
       which corresponds to the best rotational alignment.
    """
    t1, m1 = np.array(t1, dtype=bool), np.array(m1, dtype=bool)
    t2, m2 = np.array(t2, dtype=bool), np.array(m2, dtype=bool)
    best = 1.0
    for s in range(-8, 9):
        t2s, m2s = np.roll(t2, s, 1), np.roll(m2, s, 1)
        # Ignore a bit if it is noisy in either of the two templates
        mask  = m1 | m2s
        valid = (~mask).sum()
        if valid == 0: continue
        # XOR gives 1 wherever bits disagree; we only count clean bit positions
        best = min(best, (np.logical_xor(t1, t2s) & ~mask).sum() / valid)
    return best


# ── Enrollment & scoring ──────────────────────────────────────────────────────

def enroll(client_ids, subjects):
    # Registration phase: one template per client using their first image
    templates, failed = {}, 0
    for i, cid in enumerate(client_ids):
        try:
            r = extractFeature(subjects[cid][0])
            templates[cid] = (r[0], r[1])
            if (i + 1) % 10 == 0:
                print(f"  Enrolled {i+1}/{len(client_ids)}")
        except:
            failed += 1
    print(f"  Done: {len(templates)} enrolled, {failed} failed")
    return templates


def genuine_scores(client_ids, subjects, templates):
    # Each client has 7 images total; image 0 was used for enrollment
    # so images 1-6 are used here as probe images for genuine testing
    scores = []
    for cid in client_ids:
        if cid not in templates: continue
        t1, m1 = templates[cid]
        for path in subjects[cid][1:]:
            try:
                r = extractFeature(path)
                scores.append(hamming_distance(t1, m1, r[0], r[1]))
            except: pass
    return scores


def impostor_scores(intruder_ids, subjects, templates):
    # Each intruder image is compared against every enrolled client template
    # We keep the minimum distance — this simulates the strongest possible
    # attack, where the intruder tries to impersonate the most similar client
    scores = []
    client_list = list(templates.items())
    for iid in intruder_ids:
        for path in subjects[iid]:
            try:
                r = extractFeature(path)
                t2, m2 = r[0], r[1]
                scores.append(min(hamming_distance(t1, m1, t2, m2)
                                  for _, (t1, m1) in client_list))
            except: pass
    return scores


# ── EER ───────────────────────────────────────────────────────────────────────

def find_eer(gen, imp):
    """
    Finds the Equal Error Rate by sweeping 1000 decision thresholds.

    At each threshold t:
      FMR  = fraction of impostor scores below t  (wrongly accepted)
      FNMR = fraction of genuine scores above t   (wrongly rejected)

    Lowering t makes the system stricter: fewer impostors pass but more
    genuine users get rejected. Raising t does the opposite.
    EER is the threshold where both error rates are equal — a standard
    single-number summary of biometric system accuracy.
    """
    th   = np.linspace(0, 1, 1000)
    g, i = np.array(gen), np.array(imp)
    fmr  = np.array([np.mean(i <= t) for t in th])
    fnmr = np.array([np.mean(g >  t) for t in th])
    idx  = np.argmin(np.abs(fmr - fnmr))
    return th[idx], (fmr[idx] + fnmr[idx]) / 2, fmr, fnmr, th


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_results(gen, imp, thr, eer, fmr, fnmr, th, tag="seed42"):
    """
    Saves a three-panel figure:
      Panel 1 - Score distributions: genuine and impostor HD histograms.
                The overlap region shows where classification errors occur.
      Panel 2 - FMR and FNMR vs threshold: the two error curves and their
                crossing point at the EER threshold.
      Panel 3 - ROC curve: True Match Rate vs False Match Rate across all
                thresholds. A perfect system hugs the top-left corner.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'CASIA-IrisV1 — Masek/Kovesi System  |  {tag}',
                 fontsize=12, fontweight='bold', y=1.01)

    # Panel 1 — Score distributions
    ax = axes[0]
    ax.set_facecolor('#f8f9fa')
    bins = np.linspace(0, 0.6, 50)
    ax.hist(gen, bins=bins, alpha=0.75, color='#2196F3',
            label=f'Genuine  (n={len(gen)})',  density=True)
    ax.hist(imp, bins=bins, alpha=0.75, color='#F44336',
            label=f'Impostor (n={len(imp)})', density=True)
    ax.axvline(thr, color='#212121', ls='--', lw=2,
               label=f'EER threshold = {thr:.3f}')
    g_hist, edges = np.histogram(gen, bins=bins, density=True)
    i_hist, _     = np.histogram(imp, bins=bins, density=True)
    midpoints     = (edges[:-1] + edges[1:]) / 2
    # Purple shading shows the overlap region where errors are unavoidable
    ax.fill_between(midpoints, np.minimum(g_hist, i_hist),
                    alpha=0.35, color='purple', label='Error region')
    ax.set_xlabel('Hamming Distance', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Score Distributions', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.4, ls='--')

    # Panel 2 — FMR/FNMR vs threshold
    ax = axes[1]
    ax.set_facecolor('#f8f9fa')
    ax.plot(th, fmr  * 100, color='#F44336', lw=2, label='FMR (False Match Rate)')
    ax.plot(th, fnmr * 100, color='#2196F3', lw=2, label='FNMR (False Non-Match Rate)')
    ax.axvline(thr, color='#212121', ls='--', lw=1.8,
               label=f'EER threshold = {thr:.3f}')
    ax.axhline(eer * 100, color='#9C27B0', ls=':', lw=1.5,
               label=f'EER = {eer*100:.2f}%')
    ax.scatter([thr], [eer * 100], color='#9C27B0', s=80, zorder=5)
    ax.set_xlabel('Threshold', fontsize=10)
    ax.set_ylabel('Error Rate (%)', fontsize=10)
    ax.set_title('FMR and FNMR vs Threshold', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='both', alpha=0.4, ls='--')

    # Panel 3 — ROC curve
    ax = axes[2]
    ax.set_facecolor('#f8f9fa')
    tpr = (1 - fnmr) * 100
    ax.fill_between(fmr * 100, tpr, alpha=0.12, color='#2196F3')
    ax.plot(fmr * 100, tpr, color='#2196F3', lw=2.5, label='ROC curve')
    ax.plot([0, 100], [0, 100], color='#9E9E9E', ls=':', lw=1.5, label='Random baseline')
    ax.scatter([eer * 100], [(1 - eer) * 100], color='#F44336',
               s=100, zorder=5, label=f'EER = {eer*100:.2f}%')
    ax.annotate(f'EER = {eer*100:.2f}%',
                xy=(eer*100, (1-eer)*100),
                xytext=(eer*100 + 8, (1-eer)*100 - 10),
                arrowprops=dict(arrowstyle='->', color='#F44336'),
                fontsize=8, color='#F44336')
    ax.set_xlabel('False Match Rate (%)', fontsize=10)
    ax.set_ylabel('True Match Rate (%)', fontsize=10)
    ax.set_title('ROC Curve', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='both', alpha=0.4, ls='--')
    ax.set_xlim([-1, 101]); ax.set_ylim([-1, 101])

    plt.tight_layout()
    fname = os.path.join(REPO_ROOT, f'results_{tag}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: results_{tag}.png")


# ── Main ──────────────────────────────────────────────────────────────────────
# The if __name__ guard is required because Masek's segment.py spawns
# worker processes internally using multiprocessing. Without this guard,
# each worker would re-import this script and trigger an infinite loop.

if __name__ == '__main__':
    print("=" * 55)
    print("CASIA-IrisV1 Iris Recognition Experiment")
    print("=" * 55)

    subjects = parse_database(CASIA_DIR)
    ids      = sorted(subjects.keys())
    print(f"\nDatabase: {len(ids)} subjects, "
          f"{sum(len(v) for v in subjects.values())} images\n")

    clients, intruders = split_subjects(ids, 0.9, seed=42)
    print(f"Split: {len(clients)} clients / {len(intruders)} intruders\n")

    print("Step 1/4 — Enrolling clients...")
    tmpls = enroll(clients, subjects)

    print("\nStep 2/4 — Genuine scores...")
    gen = genuine_scores(clients, subjects, tmpls)
    print(f"  {len(gen)} scores | mean={np.mean(gen):.4f}  std={np.std(gen):.4f}")

    print("\nStep 3/4 — Impostor scores...")
    imp = impostor_scores(intruders, subjects, tmpls)
    print(f"  {len(imp)} scores | mean={np.mean(imp):.4f}  std={np.std(imp):.4f}")

    thr, eer, fmr, fnmr, th = find_eer(gen, imp)
    print(f"\n{'='*40}")
    print(f"EER threshold : {thr:.4f}")
    print(f"Equal Error Rate: {eer*100:.2f}%")
    print(f"{'='*40}")

    print("\nStep 4/4 — Saving plot...")
    plot_results(gen, imp, thr, eer, fmr, fnmr, th, tag="seed42")

    # Repeat the full experiment across 5 different random splits to confirm
    # that the EER is stable and not sensitive to which subjects are clients
    print("\nStability test across 5 seeds...")
    print(f"{'Seed':>6} | {'Threshold':>10} | {'EER':>8}")
    print("-" * 35)
    all_t, all_e = [], []
    for seed in [42, 123, 7, 99, 2026]:
        c, i     = split_subjects(ids, 0.9, seed)
        t_       = enroll(c, subjects)
        g_       = genuine_scores(c, subjects, t_)
        im_      = impostor_scores(i, subjects, t_)
        t, e, *_ = find_eer(g_, im_)
        all_t.append(t); all_e.append(e * 100)
        print(f"{seed:>6} | {t:>10.4f} | {e*100:>7.2f}%")
    print(f"\nThreshold: mean={np.mean(all_t):.4f}  std={np.std(all_t):.4f}")
    print(f"EER:       mean={np.mean(all_e):.2f}%  std={np.std(all_e):.2f}%")
    print("\nDone! Open results_seed42.png to see the plots.")