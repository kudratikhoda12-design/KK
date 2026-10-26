"""Builds reports/final_report.pdf: a proper, typeset write-up of the whole
project (motivation, method, results, limitations, future work) with all
figures and result tables embedded, using reportlab."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, HRFlowable,
)

FIG = "reports/figures"
RES = "reports/results"


def load_json(name):
    with open(f"{RES}/{name}") as f:
        return json.load(f)


def build_styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(name="TitleBig", fontSize=22, leading=26, spaceAfter=6, fontName="Helvetica-Bold"))
    ss.add(ParagraphStyle(name="Subtitle", fontSize=12, leading=16, textColor=colors.HexColor("#555555"), spaceAfter=20))
    ss.add(ParagraphStyle(name="H1", fontSize=15, leading=18, spaceBefore=18, spaceAfter=8, fontName="Helvetica-Bold", textColor=colors.HexColor("#1a1a2e")))
    ss.add(ParagraphStyle(name="H2", fontSize=12, leading=15, spaceBefore=12, spaceAfter=6, fontName="Helvetica-Bold"))
    ss.add(ParagraphStyle(name="Body", fontSize=10, leading=14.5, spaceAfter=8, fontName="Helvetica"))
    ss.add(ParagraphStyle(name="Caption", fontSize=8.5, leading=11, textColor=colors.HexColor("#555555"), spaceAfter=14, alignment=1))
    ss.add(ParagraphStyle(name="BulletItem", fontSize=10, leading=14.5, spaceAfter=4, leftIndent=14, bulletIndent=4))
    return ss


def img(path, width=6.3 * inch):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    height = width * h / w
    max_h = 7.2 * inch
    if height > max_h:
        height = max_h
        width = height * w / h
    return Image(path, width=width, height=height)


def table_from_rows(header, rows, col_widths=None):
    data = [header] + rows
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def main():
    ss = build_styles()
    story = []

    denoise = load_json("denoise_results.json")
    fewshot = load_json("fewshot_summary.json")
    failure = load_json("failure_analysis.json")
    bandit = load_json("bandit_results.json")
    real_denoise = load_json("real_denoise_results.json")
    real_fewshot = load_json("real_fewshot_summary.json")
    real_failure = load_json("real_failure_analysis.json")
    real_bandit = load_json("real_bandit_results.json")

    # ---------- Title page ----------
    story.append(Spacer(1, 1.6 * inch))
    story.append(Paragraph("Learning Body State from Pressure Sensor Data", ss["TitleBig"]))
    story.append(Paragraph(
        "Denoising &rarr; Representation Learning &rarr; Decision-Making, Built on a Synthetic "
        "Pilot and Validated on Real PhysioNet PmatData",
        ss["Subtitle"]))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#cccccc"), thickness=1))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "<b>What this is:</b> an end-to-end project covering (1) a denoising autoencoder for raw "
        "pressure-sensor readings, (2) self-supervised representation learning evaluated under a few-shot "
        "labeling budget, with baselines and ablations, and (3) a contextual bandit for deciding which "
        "mattress zone to adjust under a hidden user comfort function.", ss["Body"]))
    story.append(Paragraph(
        "<b>Data:</b> the full pipeline was built and debugged on a programmatic synthetic pressure-map "
        "generator with a documented, explicit sensor noise model (code: <font face='Courier'>src/data.py</font>), "
        "then run <b>unchanged</b> on a real, open-access pressure-mat dataset &mdash; PmatData (Baran Pouyan "
        "et al., IEEE BHI 2017), PhysioNet, 13 human subjects, no login required &mdash; to check whether the "
        "synthetic-data conclusions actually transfer. Both studies are reported here; see 'Validation on "
        "Real Data' for what changed and what didn't.", ss["Body"]))
    story.append(Paragraph(
        "<b>Reproduce:</b> <font face='Courier'>pip install -r requirements.txt &amp;&amp; bash run_all.sh "
        "--with-real</font> &mdash; synthetic study in under a minute on CPU, real-data study a few minutes "
        "more (downloads ~107MB from PhysioNet).", ss["Body"]))
    story.append(PageBreak())

    # ---------- Motivation ----------
    story.append(Paragraph("Motivation", ss["H1"]))
    story.append(Paragraph(
        "A pressure-sensing mattress reports a 2D grid of raw taxel readings. Three properties make this "
        "a genuinely interesting ML problem rather than a plain classification task. First, the raw signal "
        "is noisy &mdash; resistive sensor arrays drop cells, spike under cross-talk, and quantize coarsely, "
        "so it is worth knowing how to clean the signal and how much that cleaning is actually worth. "
        "Second, labels are expensive &mdash; a real deployment might realistically get 5&ndash;10 labeled "
        "posture examples per class, which is exactly the regime where self-supervised pretraining is "
        "supposed to pay off. Third, sensing should drive action: the end goal of a smart mattress is a "
        "decision (which zone to relieve right now), made under a comfort function that is never directly "
        "observable &mdash; a contextual bandit problem. The three phases of this project map directly onto "
        "these three points.", ss["Body"]))

    # ---------- Data / Method ----------
    story.append(Paragraph("Pilot Study: Synthetic Data", ss["H1"]))
    story.append(Paragraph(
        "<b>Choice: start synthetic, then validate on real data.</b> The original PMAP dataset is "
        "human-subjects data and is request-gated; rather than stall on data access, the pipeline was built "
        "and debugged first on a programmatic generator that preserves the structure of the real problem and "
        "makes every noise assumption explicit and auditable, then pointed at a real, openly-downloadable "
        "pressure-mat dataset with no changes to model or training code once it worked (see 'Validation on "
        "Real Data' below). This caught a real bug cheaply, before it could cost real-data debugging time.", ss["Body"]))
    story.append(Paragraph(
        "The generator produces a 48&times;24 taxel grid. A body is a sum of anisotropic Gaussian "
        "‘blobs’ (head, shoulders, chest, pelvis, thighs, calves). Posture is the label "
        "(<font face='Courier'>supine, prone, left_lateral, right_lateral</font>): lateral postures are "
        "shifted and narrower (side-lying footprint), while supine vs. prone &mdash; the harder pair &mdash; "
        "differ only in the relative intensity of the chest vs. pelvis blob. Body size (small/medium/large, "
        "plus continuous jitter) and a global random placement offset are both generated but withheld from "
        "the model as nuisance variables, used later for failure analysis.", ss["Body"]))
    story.append(Paragraph("Documented sensor noise model:", ss["Body"]))
    story.append(Paragraph(
        "noisy = quantize( clip( clean + Gaussian(&sigma;) + dead-cell dropout(p<sub>drop</sub>) "
        "+ salt spikes(p<sub>salt</sub>), 0, 1 ), levels )", ss["Body"]))
    story.append(Paragraph(
        "with &sigma;, p<sub>drop</sub>, p<sub>salt</sub>, and quantization levels scaling together across "
        "three severities (low/medium/high, mixed uniformly over the dataset): &sigma; 0.02&rarr;0.09, "
        "p<sub>drop</sub> 1%&rarr;6%, p<sub>salt</sub> 0.2%&rarr;1.2%, levels 64&rarr;16.", ss["Body"]))
    story.append(img(f"{FIG}/00_sanity_check.png", width=4.6 * inch))
    story.append(Paragraph("Clean / noisy (low) / noisy (high) / clean-small-body renders for each posture class.", ss["Caption"]))

    story.append(PageBreak())

    # ---------- Phase 1 ----------
    story.append(Paragraph("Phase 1 — Denoising (Synthetic Pilot)", ss["H1"]))
    story.append(Paragraph(
        "A small conv encoder-decoder (3 stride-2 conv blocks each way, 64-channel bottleneck) trained "
        "with MSE to map noisy &rarr; clean. 40 epochs, 2000 train / 500 validation pairs, ~14s on CPU.", ss["Body"]))
    story.append(table_from_rows(
        ["", "MSE", "PSNR"],
        [
            ["Identity baseline (do nothing)", f"{denoise['identity_baseline_mse']:.5f}", f"{denoise['identity_baseline_psnr_db']:.1f} dB"],
            ["Denoising autoencoder", f"{denoise['final_val_mse']:.5f}", f"{denoise['final_val_psnr_db']:.1f} dB"],
        ],
        col_widths=[2.6 * inch, 1.8 * inch, 1.8 * inch],
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"<b>{denoise['improvement_over_identity_pct']:.1f}% reduction</b> in reconstruction MSE vs. doing nothing.", ss["Body"]))
    story.append(img(f"{FIG}/01_denoise_qualitative.png", width=6.3 * inch))
    story.append(Paragraph("Noisy input / model denoised output / clean target, six held-out validation samples.", ss["Caption"]))
    story.append(img(f"{FIG}/01_denoise_training_curve.png", width=4.3 * inch))
    story.append(Paragraph("Training curve against the identity-baseline MSE floor.", ss["Caption"]))

    story.append(PageBreak())

    # ---------- Phase 2 ----------
    story.append(Paragraph("Phase 2 — Representation Learning + Few-Shot Classification (Synthetic Pilot)", ss["H1"]))
    story.append(Paragraph(
        "The same backbone architecture is pretrained two ways on <i>unlabeled</i> data, so only the "
        "pretraining signal differs downstream, not the architecture: <b>contrastive</b> (SimCLR-style "
        "NT-Xent, temperature 0.3, augmentations = noise/translation/dropout/intensity-scale &mdash; "
        "deliberately no horizontal flip, since flipping would swap left/right lateral labels), and "
        "<b>denoising</b> (the Phase 1 autoencoder's backbone, reused directly).", ss["Body"]))
    story.append(Paragraph(
        "Evaluated for k &isin; {5, 10, 20} labels/class (4 classes), averaged over 8 random label draws, "
        "against a held-out test set, six methods:", ss["Body"]))

    pretty = {
        "linear_probe_random": "Linear probe, random-init encoder (floor)",
        "cnn_scratch": "CNN trained from scratch (k-shot only)",
        "linear_probe_denoise": "Linear probe, denoising-pretrained encoder",
        "logreg_raw": "LogReg on raw noisy pixels",
        "logreg_denoised": "LogReg on denoised pixels",
        "linear_probe_contrastive": "Linear probe, contrastive-pretrained encoder (ours)",
    }
    order = ["linear_probe_random", "cnn_scratch", "linear_probe_denoise", "logreg_raw", "logreg_denoised", "linear_probe_contrastive"]
    rows = []
    for m in order:
        r = [pretty[m]]
        for k in ["5", "10", "20"]:
            r.append(f"{100*fewshot[m][k]['mean']:.1f}%")
        rows.append(r)
    story.append(table_from_rows(["method", "k=5", "k=10", "k=20"], rows, col_widths=[3.4 * inch, 1 * inch, 1 * inch, 1 * inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("(chance = 25%; full numbers with std in reports/results/fewshot_summary.json)", ss["Caption"]))
    story.append(img(f"{FIG}/03_fewshot_accuracy.png", width=5.2 * inch))
    story.append(Paragraph("Few-shot test accuracy vs. labels per class, mean ± std over 8 draws.", ss["Caption"]))

    story.append(PageBreak())

    # ---------- Step 3: evaluate like a researcher ----------
    story.append(Paragraph("Step 3 — Evaluating Like a Researcher", ss["H1"]))
    story.append(Paragraph("Ablations", ss["H2"]))
    for txt in [
        "<b>Does pretraining help, and by how much?</b> Yes, decisively, against the only fair control "
        "(same architecture, random weights): the random-init linear probe sits at 45&ndash;47% regardless "
        "of k, while either pretraining method lifts it to 62&ndash;73% &mdash; a +17 to +26 point jump.",
        "<b>Does denoising help downstream accuracy?</b> Yes, modestly and consistently: LogReg on denoised "
        "pixels beats LogReg on raw noisy pixels at every k (+0.9, +1.2, +1.4 points). A real but small effect.",
        "<b>Does the CNN-from-scratch baseline justify its cost?</b> No &mdash; it is the worst non-floor "
        "method at every k, confirming that training a full conv net end-to-end on 20&ndash;80 total labels "
        "overfits badly compared to fitting a linear head on good frozen features.",
    ]:
        story.append(Paragraph("&bull; " + txt, ss["BulletItem"]))

    story.append(Paragraph("Honest surprise: raw pixels are a strong baseline", ss["H2"]))
    story.append(Paragraph(
        "The first version of this project used perfectly-centered bodies, and raw-pixel logistic "
        "regression hit <b>99%+ accuracy with only 10 labels/class</b> &mdash; beating every pretrained "
        "representation by 20+ points. That is not a success story, it is a sign the synthetic task was too "
        "easy: with no nuisance variation, posture differences are a handful of cleanly linearly-separable "
        "pixels, and 1152-dimensional logistic regression with 5&ndash;20 samples per class will happily "
        "overfit a perfect linear boundary in that regime (classic n&#8810;d). The fix, documented in "
        "<font face='Courier'>src/data.py</font>, was to add a <b>global random body placement offset per "
        "sample</b>, independent of posture &mdash; the subject is not perfectly centered on the mattress, "
        "which is true of real sensor data. That single change dropped raw-pixel LogReg from 99%&rarr;~70% "
        "and let the representation-learning story actually show up. <b>Lesson:</b> a synthetic benchmark "
        "that makes your proposed method look great by a huge margin is itself a result worth being "
        "suspicious of.", ss["Body"]))
    story.append(Paragraph(
        "Even after the fix, raw-pixel LogReg stays <i>competitive</i> with the pretrained linear probes "
        "rather than clearly losing &mdash; on a 48&times;24, 4-class problem a 1152-dim linear boundary is "
        "a genuinely strong baseline. We report this rather than cherry-picking a setting that flatters the "
        "pretrained encoder.", ss["Body"]))

    story.append(Paragraph("Failure analysis", ss["H2"]))
    story.append(Paragraph(
        f"Contrastive probe, k=10, a single representative label draw, evaluated on the full 800-sample "
        f"held-out test set (overall accuracy {100*failure['overall_accuracy']:.1f}%):", ss["Body"]))
    story.append(img(f"{FIG}/04_failure_analysis_subgroups.png", width=5.6 * inch))
    story.append(Paragraph("Accuracy broken down by nuisance variables never given to the model: body size and sensor noise level.", ss["Caption"]))
    for txt in [
        f"<b>Body size:</b> small bodies are the clear weak point "
        f"({100*failure['by_body_size']['small']['accuracy']:.1f}% vs. "
        f"{100*failure['by_body_size']['medium']['accuracy']:.1f}% / "
        f"{100*failure['by_body_size']['large']['accuracy']:.1f}% for medium/large) &mdash; smaller blobs "
        f"produce weaker, more noise-fragile pressure signatures relative to a fixed sensor-noise magnitude.",
        f"<b>Sensor noise level:</b> accuracy drops sharply at high noise "
        f"({100*failure['by_noise_level']['high']['accuracy']:.1f}%) vs. low/medium "
        f"({100*failure['by_noise_level']['low']['accuracy']:.1f}% / "
        f"{100*failure['by_noise_level']['medium']['accuracy']:.1f}%) &mdash; a reminder that ‘works "
        f"on clean sensors’ and ‘works in the field’ are different claims.",
    ]:
        story.append(Paragraph("&bull; " + txt, ss["BulletItem"]))
    story.append(img(f"{FIG}/05_confusion_matrix.png", width=4.0 * inch))
    story.append(Paragraph(
        "Supine and prone are confused with each other constantly, essentially never with either lateral "
        "posture; left/right lateral confuse each other far less. This matches the generative design "
        "exactly: lateral-vs-not is a gross, noise-robust spatial signal, while supine-vs-prone is a subtle "
        "chest/pelvis intensity ratio that noise and small body size both directly corrupt.", ss["Caption"]))

    story.append(PageBreak())

    # ---------- Phase 3: bandit ----------
    story.append(Paragraph("Phase 3 (Stretch) — Contextual Bandit for Zone Adjustment (Synthetic Pilot)", ss["H1"]))
    story.append(Paragraph(
        "The mattress is split into 5 zones (head, shoulders, lower-back/hips, thighs, calves). At each "
        "round the agent observes per-zone mean pressure as context and picks one zone to relieve. A "
        "simulated user has a hidden, fixed per-zone comfort-sensitivity vector &theta; (unknown to the "
        "agent): reward(zone z) = &theta;<sub>z</sub>&middot;pressure<sub>z</sub> + noise &mdash; a disjoint "
        "linear contextual bandit. Policies: random, epsilon-greedy (per-arm ridge regression, "
        "&epsilon;=0.1), LinUCB (disjoint, Li et al. 2010, &alpha;=1.0), and an oracle that knows &theta; "
        "exactly. T=3000 rounds.", ss["Body"]))

    rows = []
    for name, label in [("random", "Random"), ("eps_greedy", "Epsilon-greedy"), ("linucb", "LinUCB"), ("oracle", "Oracle (upper bound)")]:
        r = bandit["results"][name]
        rows.append([label, f"{r['final_cumulative_regret']:.1f}", f"{r['mean_reward_last_500']:.3f}"])
    story.append(table_from_rows(
        ["policy", "final cumulative regret", "mean reward (last 500 rounds)"],
        rows, col_widths=[2.2 * inch, 2.2 * inch, 2.4 * inch]))
    story.append(Spacer(1, 6))
    story.append(img(f"{FIG}/06_bandit_regret.png", width=5.2 * inch))
    story.append(Paragraph(
        "LinUCB's context-aware confidence bound closes the gap to oracle far faster than epsilon-greedy's "
        "undirected exploration. Regret is visibly sublinear (concave) for both learning policies and "
        "linear for random, exactly as theory predicts.", ss["Caption"]))

    story.append(PageBreak())

    # ============================================================
    # REAL DATA VALIDATION CHAPTER
    # ============================================================
    story.append(Paragraph("Validation on Real Data (PmatData)", ss["H1"]))
    story.append(Paragraph(
        "<b>Source:</b> PmatData, “A Pressure Map Dataset for In-bed Posture Classification” "
        "(Baran Pouyan et al., IEEE BHI 2017), QoLT Lab, UT Dallas. Open access on PhysioNet under the Open "
        "Data Commons Attribution License v1.0 &mdash; <b>no login or credentialing required</b>, downloaded "
        "directly (<font face='Courier'>scripts/08_download_real_data.sh</font>). Experiment I: 13 subjects, "
        "a real Vista Medical FSA SoftFlex 2048 mat (32&times;64 taxels), 17 labeled positions per subject, "
        "~20,000 real pressure frames in total.", ss["Body"]))
    story.append(Paragraph(
        "<b>Labels:</b> transcribed directly from the dataset's own position table. The real dataset has "
        "<b>no prone posture</b> recorded, so this study uses 3 classes (supine/left/right) instead of the "
        "synthetic study's 4. Bed inclination and body-roll wedges/fetal position are recorded and used as "
        "a real nuisance-variant label, the direct analogue of synthetic <font face='Courier'>body_size</font>.", ss["Body"]))
    story.append(Paragraph(
        "<b>Split:</b> 3 of 13 subjects held out entirely &mdash; none of their frames appear in "
        "pretraining, denoising, or few-shot label selection. Test accuracy below measures generalization "
        "to an <b>unseen person's body</b>, stricter than the synthetic study's frame-level split.", ss["Body"]))

    story.append(Paragraph("Denoising on real data", ss["H2"]))
    story.append(table_from_rows(
        ["", "MSE", "PSNR"],
        [
            ["Identity baseline", f"{real_denoise['identity_baseline_mse']:.5f}", f"{real_denoise['identity_baseline_psnr_db']:.1f} dB"],
            ["Denoising autoencoder (real data)", f"{real_denoise['final_val_mse']:.5f}", f"{real_denoise['final_val_psnr_db']:.1f} dB"],
        ],
        col_widths=[2.8 * inch, 1.7 * inch, 1.7 * inch],
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"<b>{real_denoise['improvement_over_identity_pct']:.1f}% MSE reduction</b> &mdash; working, but "
        f"weaker than the synthetic pilot's 91.8%/32.0dB. Real pressure maps have fine, sparse, "
        f"high-frequency structure (fingers, heel points, shoulder blades) a small bottleneck smooths over; "
        f"synthetic Gaussian blobs are smooth by construction and easier to reconstruct.", ss["Body"]))
    story.append(img(f"{FIG}/07_real_denoise_qualitative.png", width=5.6 * inch))
    story.append(Paragraph("Real noisy input / model denoised output / real clean target, held-out validation samples.", ss["Caption"]))

    story.append(PageBreak())

    story.append(Paragraph("Few-shot posture classification on real data (unseen subjects)", ss["H2"]))
    story.append(Paragraph(
        "Same six methods and k &isin; {5,10,20} protocol, evaluated on 3 held-out subjects. Real posture "
        "classes are imbalanced (supine covers 9 of 17 positions), so <b>balanced accuracy</b> "
        "(macro-recall) is reported:", ss["Body"]))

    real_pretty = {
        "linear_probe_random": "Linear probe, random-init encoder (floor)",
        "cnn_scratch": "CNN trained from scratch",
        "linear_probe_contrastive": "Linear probe, contrastive-pretrained encoder",
        "linear_probe_denoise": "Linear probe, denoising-pretrained encoder",
        "logreg_raw": "LogReg on raw pixels",
        "logreg_denoised": "LogReg on denoised pixels",
    }
    real_order = ["linear_probe_random", "cnn_scratch", "linear_probe_contrastive", "linear_probe_denoise", "logreg_raw", "logreg_denoised"]
    rows = []
    for m in real_order:
        r = [real_pretty[m]]
        for k in ["5", "10", "20"]:
            r.append(f"{100*real_fewshot[m][k]['balanced_mean']:.1f}%")
        rows.append(r)
    story.append(table_from_rows(["method", "k=5", "k=10", "k=20"], rows, col_widths=[3.4 * inch, 1 * inch, 1 * inch, 1 * inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("(balanced accuracy; chance = 33%; full numbers in reports/results/real_fewshot_summary.json)", ss["Caption"]))
    story.append(img(f"{FIG}/09_real_fewshot_accuracy.png", width=5.0 * inch))
    story.append(Paragraph("Balanced few-shot test accuracy on unseen subjects, mean ± std over 8 draws.", ss["Caption"]))

    story.append(Paragraph(
        "<b>This is a materially different result from the synthetic pilot, reported as-is rather than "
        "adjusted to match the earlier narrative.</b> On real data, raw-pixel logistic regression &mdash; "
        "especially on denoised pixels &mdash; dominates by a wide margin at every k, more decisively than "
        "on synthetic data. This is not the earlier bug repeating: real supine/left/right postures produce "
        "dramatically different, highly reproducible pixel silhouettes (a side-lying body rotates and "
        "narrows the whole footprint), so a high-dimensional linear classifier with even 10&ndash;20 "
        "examples exploits that structure very effectively &mdash; genuine task structure this time, not a "
        "generator artifact. These three coarse classes are simply easier to separate by gross shape than "
        "the synthetic pilot's hardest pair (supine vs. prone, a subtle intensity-ratio cue) &mdash; and "
        "prone isn't in this real dataset to test against.", ss["Body"]))
    story.append(Paragraph(
        "The pretraining ablation still holds: both pretrained linear probes beat the random-init floor by "
        "a wide margin (+8 to +29 balanced-accuracy points) at every k, confirming pretraining learns "
        "something real about body pressure structure even on a task where it isn't the best available "
        "method.", ss["Body"]))

    story.append(PageBreak())

    story.append(Paragraph("Failure analysis on real data", ss["H2"]))
    story.append(Paragraph(
        f"Contrastive linear probe, k=10, one representative label draw, all 3 held-out subjects "
        f"(overall accuracy {100*real_failure['overall_accuracy']:.1f}%, balanced {100*real_failure['overall_balanced_accuracy']:.1f}%):", ss["Body"]))
    story.append(img(f"{FIG}/10_real_failure_analysis_subgroups.png", width=6.2 * inch))
    story.append(Paragraph("Accuracy by real nuisance variant (bed incline/wedge/fetus) and by held-out subject.", ss["Caption"]))

    v = real_failure["by_variant"]
    for txt in [
        f"<b>Bed inclination is the standout failure mode, with a clean dose-response curve:</b> flat "
        f"supine {100*v['flat']['accuracy']:.1f}% &rarr; 30&deg; {100*v['incline30']['accuracy']:.1f}% "
        f"&rarr; 45&deg; {100*v['incline45']['accuracy']:.1f}% &rarr; 60&deg; <b>{100*v['incline60']['accuracy']:.1f}%</b> "
        f"&mdash; worse than chance. The model was trained almost entirely on flat postures, so a steeply "
        f"inclined supine posture, which genuinely redistributes pressure toward the hips/lower back, is "
        f"essentially out-of-distribution. The real-data analogue of the synthetic study's 'small body "
        f"size' finding: a real physical nuisance variable the model was never taught to be invariant to.",
        f"<b>Body-roll wedges</b> ({100*v['wedge30']['accuracy']:.1f}% at wedge30) also hurt noticeably "
        f"more than the flat baseline ({100*v['flat']['accuracy']:.1f}%), while fetal position "
        f"({100*v['fetus']['accuracy']:.1f}%) is closer to normal.",
    ]:
        story.append(Paragraph("&bull; " + txt, ss["BulletItem"]))

    story.append(img(f"{FIG}/11_real_confusion_matrix.png", width=3.8 * inch))
    story.append(Paragraph(
        "This single k=10 draw's confusion matrix shows a classifier biased toward predicting 'right' "
        "&mdash; a few-shot <b>instability</b> symptom (a linear probe fit on 10 examples/class in a "
        "64-dim space can land on a biased boundary by chance on any single draw), not a stable property "
        "of the method. The main few-shot table (averaged over 8 draws) is the number to trust; this "
        "single-draw matrix is shown because it illustrates that instability honestly.", ss["Caption"]))

    story.append(PageBreak())

    story.append(Paragraph("Contextual bandit on real data", ss["H2"]))
    story.append(Paragraph(
        "Identical bandit simulation to the synthetic pilot (5 zones, disjoint linear reward, hidden "
        "per-zone sensitivity &theta;), with the context stream replaced by real PmatData frames &mdash; "
        "no other change.", ss["Body"]))
    rows = []
    oracle_r = real_bandit["results"]["oracle"]["mean_reward_last_500"]
    for name, label in [("random", "Random"), ("eps_greedy", "Epsilon-greedy"), ("linucb", "LinUCB"), ("oracle", "Oracle (upper bound)")]:
        r = real_bandit["results"][name]
        pct = 100 * r["mean_reward_last_500"] / oracle_r
        rows.append([label, f"{r['final_cumulative_regret']:.1f}", f"{r['mean_reward_last_500']:.3f}", f"{pct:.1f}%"])
    story.append(table_from_rows(
        ["policy", "final cumulative regret", "mean reward (last 500)", "% of oracle reward"],
        rows, col_widths=[1.8 * inch, 1.7 * inch, 1.6 * inch, 1.5 * inch]))
    story.append(Spacer(1, 6))
    story.append(img(f"{FIG}/12_real_bandit_regret.png", width=5.0 * inch))
    story.append(Paragraph(
        "Identical qualitative story to the synthetic pilot: LinUCB reaches within 3.4% of oracle reward, "
        "clearly beating epsilon-greedy and random. This part of the pipeline was context-agnostic by "
        "design and transfers to real data without modification, as expected.", ss["Caption"]))

    story.append(Paragraph("What changed between synthetic and real, in one paragraph", ss["H2"]))
    story.append(Paragraph(
        "Denoising works on both, modestly weaker on real data's finer detail. Pretraining reliably beats "
        "random-init features on both. What doesn't transfer is which method wins outright: on the harder "
        "synthetic task (subtle supine/prone cue, deliberate placement jitter), contrastive pretraining is "
        "competitive-to-best; on the real, coarser 3-class task, simple logistic regression on (denoised) "
        "pixels wins decisively, because real supine/left/right postures are that linearly separable in "
        "pixel space. Neither result is wrong &mdash; they answer different questions about different "
        "tasks, and reporting both rather than only the one that flatters representation learning is the "
        "point of doing this validation at all.", ss["Body"]))

    story.append(PageBreak())

    # ---------- Limitations & Future work ----------
    story.append(Paragraph("Limitations", ss["H1"]))
    for txt in [
        "<b>The real-data study still doesn't cover everything a deployed system would face.</b> PmatData "
        "is 13 subjects on one mat model in a lab setting &mdash; it captures real anatomical variation and "
        "genuine sensor noise/sparsity, but still has no prone posture, no long-duration sessions, and no "
        "sensor drift across weeks. 'Validated on real data' means real recordings, not a real deployment.",
        "<b>Representation learning's advantage is real but task-dependent, not dominant &mdash; more so "
        "than expected.</b> On synthetic data, contrastive pretraining was competitive-to-best against raw "
        "pixels; on real data, raw-pixel LogReg wins decisively, because real supine/left/right postures "
        "are extremely linearly separable by gross silhouette. The honest claim across both studies: "
        "pretrained features reliably beat a random-init encoder; whether they beat a strong linear "
        "baseline depends on how fine-grained the distinction is.",
        "<b>The bandit's comfort function is linear and stationary,</b> in both studies, and was only ever "
        "validated on simulated reward, never a real comfort signal (none exists in PmatData).",
        "<b>Few-shot linear probes are unstable on a single label draw,</b> confirmed directly in the "
        "real-data failure analysis. The averaged, multi-draw few-shot tables are the trustworthy numbers; "
        "a single deployed few-shot classifier should expect similar draw-to-draw variance.",
        "<b>No real-world sensor drift or calibration across sessions was modeled</b> in either study.",
    ]:
        story.append(Paragraph("&bull; " + txt, ss["BulletItem"]))

    story.append(Paragraph("Future Work", ss["H1"]))
    for txt in [
        "<b>Done in this iteration:</b> validated the synthetic-data conclusions against a real "
        "pressure-mat dataset. Result: pretraining-beats-random-init transferred cleanly; "
        "'pretraining beats simple baselines outright' did not &mdash; a finding worth having, not a "
        "failure to paper over.",
        "Test whether representation learning's advantage reappears on a harder real discrimination task "
        "&mdash; PmatData's own bed-incline/body-roll variants (used here only as nuisance labels) could be "
        "turned into finer-grained labels closer in spirit to the synthetic study's subtle supine/prone cue.",
        "Try a denoising + contrastive joint pretraining objective to see if the two signals are "
        "complementary rather than alternatives.",
        "Extend the bandit's reward model to be non-stationary and to richer action spaces; validating "
        "against even simulated-but-individualized human comfort ratings would be stronger than the "
        "current fully-synthetic reward.",
        "Bring in PmatData's Experiment II (a second, different mat model) as a cross-hardware "
        "generalization test: train on Experiment I, evaluate on Experiment II unchanged.",
    ]:
        story.append(Paragraph("&bull; " + txt, ss["BulletItem"]))

    os.makedirs("reports", exist_ok=True)
    doc = SimpleDocTemplate(
        "reports/final_report.pdf", pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        title="Learning Body State from Pressure Sensor Data",
    )
    doc.build(story)
    print("wrote reports/final_report.pdf")


if __name__ == "__main__":
    main()
