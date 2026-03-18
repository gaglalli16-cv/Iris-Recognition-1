# generate_pipeline_figure.py
# Generates a 5-column pipeline figure comparing classical and DL iris recognition.
# Each row shows one subject passing through every stage of the classical pipeline,
# with the final column showing a DL matching example using cosine distance.
# Run from repo root with iris_env activated: python generate_pipeline_figure.py

import os, sys
import numpy as np
import cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
from collections import defaultdict

REPO_ROOT  = os.path.dirname(os.path.abspath(__file__))
CASIA_DIR  = os.path.join(REPO_ROOT, 'CASIA-database')
# If the fine-tuned model exists, column 5 uses it — otherwise falls back to MobileNetV2
MODEL_PATH = os.path.join(REPO_ROOT, 'resnet18_iris_finetuned.pth')

sys.path.insert(0, os.path.join(REPO_ROOT, 'python'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'python', 'fnc'))

try:
    from extractFeature import extractFeature
except ImportError:
    print("ERROR: Run from repo root."); sys.exit(1)


# ── Detection ──────────────────────────────────────────────────────────────────

def detect_circles(img):
    """
    Detects pupil and iris boundaries without using HoughCircles.

    HoughCircles requires hand-tuned radius ranges which vary across images.
    Instead we use a two-step approach that adapts to each image:

    Pupil — Otsu threshold finds the naturally dark region (pupil),
    contour fitting gives us the centre and radius without guessing ranges.

    Iris — radial gradient scan anchored to the pupil centre.
    We scan outward in 72 directions and find the radius where brightness
    changes most sharply (the limbus). Using the median across all directions
    makes it robust against eyelids covering the top and bottom.
    """
    h, w   = img.shape
    eq     = cv2.equalizeHist(img)
    blur   = cv2.GaussianBlur(img, (9, 9), 2)

    # Otsu threshold inverted: dark blob = pupil
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # Morphological closing fills small gaps inside the pupil blob
    thr    = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pupil, best_score = None, -1
    cx_img, cy_img = w // 2, h // 2
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 200: continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0: continue
        # Circularity score: 1.0 = perfect circle, lower = more irregular
        circ = 4 * np.pi * area / perim**2
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        cx, cy, r = int(cx), int(cy), int(r)
        # Only accept blobs near the image centre with a plausible pupil radius
        if abs(cx-cx_img) > w*0.30 or abs(cy-cy_img) > h*0.35: continue
        if not (w*0.06 < r < w*0.22): continue
        # Prefer large circular blobs over small irregular ones
        score = circ * area
        if score > best_score:
            best_score = score
            pupil = (cx, cy, r)

    iris = None
    if pupil:
        px, py, pr = pupil
        eq_b   = cv2.GaussianBlur(eq, (5, 5), 1)
        radii  = []
        for angle in np.linspace(0, 2*np.pi, 72, endpoint=False):
            ca, sa = np.cos(angle), np.sin(angle)
            prof, rs = [], []
            for r in range(int(pr*1.6), int(w*0.44), 2):
                xi, yi = int(px + r*ca), int(py + r*sa)
                if 0 <= xi < w and 0 <= yi < h:
                    prof.append(int(eq_b[yi, xi])); rs.append(r)
                else: break
            if len(prof) >= 5:
                # Peak gradient = sharpest brightness transition = limbus boundary
                radii.append(rs[int(np.argmax(np.abs(np.diff(prof))))])
        if radii:
            rm = int(np.median(radii))
            # Iris is always concentric with pupil — same centre, larger radius
            iris = (px, py, max(int(pr*1.8), min(rm, int(w*0.42))))
        else:
            iris = (px, py, int(pr*2.8))   # fallback estimate if scan fails
    return pupil, iris


def score_detection(pupil, iris, img_shape):
    """
    Scores the quality of a circle detection result.
    Used to automatically select the 3 best subjects for the figure.
    A high score means: pupil found, realistic pupil/iris size ratio,
    iris covers a reasonable portion of the image, pupil near centre.
    """
    if pupil is None: return 0
    h, w = img_shape
    px, py, pr = pupil
    score = 4   # base score for finding the pupil at all
    if iris:
        _, _, ir = iris
        ratio = pr / ir
        # Typical pupil/iris radius ratio in NIR images is 0.25-0.42
        score += 5 if 0.25 < ratio < 0.42 else (2 if 0.20 < ratio < 0.50 else -3)
        # Iris should cover 25-42% of image width
        score += 3 if 0.25 < ir/w < 0.42 else (1 if 0.20 < ir/w < 0.46 else -2)
        # Penalise heavily if the pupil centre is far from the image centre
        dist  = np.sqrt((px - w//2)**2 + (py - h//2)**2)
        score += 2 if dist < w*0.10 else (-3 if dist > w*0.25 else 0)
    return score


def rubber_sheet(img, pupil, iris, radial=20, angular=240):
    """
    Manual implementation of Daugman's Rubber Sheet model.

    The annular iris region is remapped from circular to rectangular
    using polar coordinate interpolation. For each point in the output
    strip (row = radial position, column = angle), we linearly interpolate
    between the pupil boundary and the iris boundary at that angle.

    Output: a 20 x 240 strip where each row corresponds to a radial
    distance from the pupil (row 0) to the iris edge (row 19).
    """
    if not pupil or not iris: return None
    px, py, pr = pupil
    ix, iy, ir = iris
    strip = np.zeros((radial, angular), dtype=np.uint8)
    for j, theta in enumerate(np.linspace(0, 2*np.pi, angular, endpoint=False)):
        ca, sa = np.cos(theta), np.sin(theta)
        for i in range(radial):
            r  = i / radial   # normalised radius: 0 at pupil edge, 1 at iris edge
            xi = int((1-r)*(px + pr*ca) + r*(ix + ir*ca))
            yi = int((1-r)*(py + pr*sa) + r*(iy + ir*sa))
            if 0 <= xi < img.shape[1] and 0 <= yi < img.shape[0]:
                strip[i, j] = img[yi, xi]
    return strip


def get_dl_embedding(img):
    """
    Extracts a 512-dim L2-normalised embedding from the fine-tuned ResNet18.
    Falls back to MobileNetV2 (ImageNet) if the fine-tuned weights are not found.
    The embedding is used to compute cosine distance for the matching column.
    """
    import torch, torchvision.models as models, torchvision.transforms as T
    from torch import nn

    tf = T.Compose([T.ToPILImage(), T.Resize((224, 224)),
                    T.Grayscale(3), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if os.path.exists(MODEL_PATH):
        # Load fine-tuned ResNet18 and remove the classification head
        saved    = torch.load(MODEL_PATH, map_location='cpu')
        n_cls    = saved['fc.weight'].shape[0]
        model    = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_cls)
        model.load_state_dict(saved)
        model.fc = nn.Identity()   # 512-dim output
    else:
        # Fallback: MobileNetV2 ImageNet extractor if model not trained yet
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier = nn.Identity()

    model.eval()
    with torch.no_grad():
        if hasattr(model, 'features'):
            e = model.features(tf(img).unsqueeze(0)).mean(dim=[2, 3]).squeeze(0)
        else:
            e = model(tf(img).unsqueeze(0)).squeeze(0)
    e = e / (e.norm() + 1e-8)   # L2 normalise so dot product = cosine similarity
    return e.numpy()


def get_dl_comparison(subject_path, subjects_db):
    """
    Computes a genuine and impostor cosine distance pair for one subject.
    This is shown in column 5 of the figure to demonstrate how DL matching works:
      - Genuine: same person, different image → low distance (green border)
      - Impostor: different person → high distance (red border)
    """
    sid      = os.path.basename(subject_path).split('_')[0]
    enroll   = cv2.imread(subject_path, cv2.IMREAD_GRAYSCALE)
    e_enroll = get_dl_embedding(enroll)

    # Genuine pair: use the second image of the same subject as the probe
    same_imgs    = subjects_db.get(sid, [])
    genuine_path = same_imgs[1] if len(same_imgs) > 1 else same_imgs[0]
    genuine_img  = cv2.imread(genuine_path, cv2.IMREAD_GRAYSCALE)
    e_genuine    = get_dl_embedding(genuine_img)
    genuine_dist = float(1 - np.dot(e_enroll, e_genuine))

    # Impostor pair: pick a subject from the middle of the sorted list
    # to avoid always selecting the same neighbouring subject
    other_sids    = [s for s in subjects_db if s != sid]
    impostor_sid  = other_sids[len(other_sids) // 2]
    impostor_img  = cv2.imread(subjects_db[impostor_sid][0], cv2.IMREAD_GRAYSCALE)
    e_impostor    = get_dl_embedding(impostor_img)
    impostor_dist = float(1 - np.dot(e_enroll, e_impostor))

    return enroll, genuine_img, genuine_dist, impostor_img, impostor_dist


# ── Image selection ────────────────────────────────────────────────────────────

def best_samples(n=3, pool=108):
    """
    Scans all subjects in the database, scores the circle detection quality
    for each one, and returns the n subjects with the highest scores.

    This avoids manually picking images and ensures the figure always shows
    clean, well-detected examples regardless of which images happen to be easy.
    """
    files = sorted(os.listdir(CASIA_DIR))
    seen, scored = set(), []
    for f in files:
        if not f.lower().endswith('.jpg'): continue
        sid = f.split('_')[0]
        if sid in seen: continue   # one image per subject for scoring
        seen.add(sid)
        path = os.path.join(CASIA_DIR, f)
        img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        p, ir = detect_circles(img)
        scored.append((score_detection(p, ir, img.shape), path, p, ir))
        if len(scored) == pool: break

    scored.sort(key=lambda x: x[0], reverse=True)
    # Only use subjects that scored at least 9 — below that the circles are unreliable
    good = [x for x in scored if x[0] >= 9]
    best = (good if len(good) >= n else scored)[:n]
    print(f"  Selected: {[(os.path.basename(p).split('_')[0], s) for s,p,_,_ in best]}")
    return [(p, pu, ir) for _, p, pu, ir in best]


# ── Figure ─────────────────────────────────────────────────────────────────────

def build_figure(samples):
    """
    Builds the 5-column pipeline comparison figure.

    Column layout:
      0 — Original grayscale image from CASIA
      1 — Segmentation: iris (green) and pupil (yellow) boundaries
      2 — Normalized strip: rubber sheet output (20x240 pixels)
      3 — Binary IrisCode: Log-Gabor phase encoding from extractFeature()
      4 — DL matching: enrolled image vs genuine (green) and impostor (red)
    """
    # Build a full subjects lookup dict needed for the impostor selection
    subjects_db = defaultdict(list)
    for f in sorted(os.listdir(CASIA_DIR)):
        if f.lower().endswith('.jpg'):
            subjects_db[f.split('_')[0]].append(os.path.join(CASIA_DIR, f))
    subjects_db = {k: sorted(v) for k, v in subjects_db.items()}

    model_label = 'ResNet18 (Fine-tuned)' if os.path.exists(MODEL_PATH) else 'MobileNetV2 (ImageNet)'
    COL_TITLES  = ['Original\nImage',
                   'Segmentation\n(Iris & Pupil)',
                   'Normalized Strip\n(Rubber Sheet)',
                   'Binary IrisCode\n(Log-Gabor Phase)',
                   f'DL Matching\n({model_label})']

    print("  Computing DL pair comparisons...")
    dl_data = {p: get_dl_comparison(p, subjects_db) for p, _, _ in samples}

    fig = plt.figure(figsize=(18, 3.8 * len(samples)))
    fig.suptitle('Pipeline Stage Comparison — Classical vs Deep Learning',
                 fontsize=13, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(len(samples), 5, figure=fig, hspace=0.4, wspace=0.15)

    for row, (path, pupil, iris) in enumerate(samples):
        sid  = os.path.basename(path).split('_')[0]
        img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Draw segmentation circles on an RGB copy of the image
        seg  = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if iris:  cv2.circle(seg, iris[:2],  iris[2],  (0, 255, 0),   2)  # green = iris
        if pupil: cv2.circle(seg, pupil[:2], pupil[2], (255, 255, 0), 2)  # yellow = pupil

        # Rubber sheet + contrast enhancement for better texture visibility
        strip = rubber_sheet(img, pupil, iris)
        if strip is not None:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 8))
            strip = clahe.apply(strip)

        # IrisCode comes directly from the Masek extractFeature function
        try:    code = extractFeature(path)[0]
        except: code = None

        enroll, genuine_img, g_dist, impostor_img, i_dist = dl_data[path]

        # ── Columns 0-3: pipeline stages ──────────────────────────────────────
        for col, (d, cmap, kwargs) in enumerate([
            (img,  'gray',   {}),
            (seg,  None,     {}),
            (strip, 'gray',  {'aspect': 'auto'}) if strip is not None else (None, None, {}),
            (code.astype(float) if code is not None else None,
             'binary', {'aspect': 'auto', 'interpolation': 'nearest'}),
        ]):
            ax = fig.add_subplot(gs[row, col])
            if row == 0: ax.set_title(COL_TITLES[col], fontsize=9, fontweight='bold')
            if col == 0: ax.set_ylabel(f'Subject {sid}', fontsize=8)

            if d is not None:
                ax.imshow(d, cmap=cmap, **kwargs)
                if col == 2:
                    ax.set_xlabel('Angular (0°→360°)', fontsize=7)
                    ax.set_ylabel('Radial\n(pupil→iris)', fontsize=7)
                    ax.tick_params(labelsize=6)
                    continue
                if col == 3:
                    ax.set_xlabel('Bit index', fontsize=7)
                    ax.set_ylabel('Phase bands', fontsize=7)
                    ax.tick_params(labelsize=6)
                    continue
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, color='gray')
                ax.set_facecolor('#1a1a1a')
            ax.axis('off')

        # ── Column 4: DL matching example ─────────────────────────────────────
        # Shows enrolled image + genuine match (green) + impostor (red)
        # with cosine distance annotated on each pair
        ax = fig.add_subplot(gs[row, 4])
        if row == 0: ax.set_title(COL_TITLES[4], fontsize=9, fontweight='bold')
        ax.axis('off')

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # Top: enrolled template
        ax_e = ax.inset_axes([0.0, 0.62, 1.0, 0.36])
        ax_e.imshow(enroll, cmap='gray')
        ax_e.set_title('Enrolled', fontsize=7, pad=2)
        ax_e.axis('off')

        # Bottom left: genuine match — same person, low distance
        ax_g = ax.inset_axes([0.0, 0.06, 0.47, 0.50])
        ax_g.imshow(genuine_img, cmap='gray')
        ax_g.axis('off')
        for spine in ax_g.spines.values():
            spine.set_edgecolor('#2E7D32'); spine.set_linewidth(3); spine.set_visible(True)
        ax_g.set_title(f'✓ Same person\nd = {g_dist:.3f}',
                       fontsize=7, color='#2E7D32', pad=2, fontweight='bold')

        # Bottom right: impostor — different person, high distance
        ax_i = ax.inset_axes([0.53, 0.06, 0.47, 0.50])
        ax_i.imshow(impostor_img, cmap='gray')
        ax_i.axis('off')
        for spine in ax_i.spines.values():
            spine.set_edgecolor('#B71C1C'); spine.set_linewidth(3); spine.set_visible(True)
        ax_i.set_title(f'✗ Different person\nd = {i_dist:.3f}',
                       fontsize=7, color='#B71C1C', pad=2, fontweight='bold')

        ax.text(0.5, 0.58, '← Cosine Distance →',
                transform=ax.transAxes, ha='center', fontsize=7, color='#555')
        ax.text(0.5, 0.01, 'Low d = match   High d = reject',
                transform=ax.transAxes, ha='center', fontsize=6.5,
                color='gray', style='italic')

    fig.legend(handles=[
        mpatches.Patch(facecolor='none', edgecolor='lime',   lw=2, label='Iris boundary'),
        mpatches.Patch(facecolor='none', edgecolor='yellow', lw=2, label='Pupil boundary'),
    ], loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.35, -0.03))

    out = os.path.join(REPO_ROOT, 'fig1_pipeline_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig1_pipeline_comparison.png")


if __name__ == '__main__':
    print("=" * 50)
    print("Generating Pipeline Stage Figure")
    print("=" * 50)
    print("  Scanning subjects...")
    build_figure(best_samples(n=3, pool=108))
    print("Done!")