"""
Bootstrap confidence intervals for Det-LoRA mAP on the test split.
==================================================================

Reports mean-over-seeds mAP@0.5 and mAP@0.5:0.95 with 95 % bootstrap confidence
intervals obtained by resampling the *test images* (n=496), as described in the
evaluation design (Abschnitt "Statistische Auswertung"). The test split is fixed
and seed-independent, so the same resampled image indices are applied across all
three seeds; the per-bootstrap statistic is the seed-mean mAP. An optional paired
mode reports the CI of the difference between two suites under shared resampling.

Efficiency: detection matching is per image, so it is precomputed ONCE per run;
each bootstrap then only re-weights the precomputed (score, is_tp) arrays
(multinomial weights) and recomputes the COCO 101-point AP -- seconds, not hours.
The point estimate (unit weights) equals det_lora.evaluation.metrics.compute_map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.evaluator import (
    apply_shared_quality_calibrator,
    collect_det_lora_joint_predictions,
)
from det_lora.evaluation.metrics import compute_ap, compute_iou_matrix, compute_map
from det_lora.model.det_lora import DetLoRA
from det_lora.model.detector import RFDETRDetector
from det_lora.train import collate_fn

CLASSES = [
    "military_tank",
    "military_truck",
    "military_aircraft",
    "military_helicopter",
    "civilian_car",
    "civilian_aircraft",
]
THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.5 .. 0.95
EPS = 1e-8


# --- prediction loading (cache or collect) ----------------------------------
def _class_ids_from_registry(checkpoint: Path) -> List[int]:
    """Class IDs without instantiating the model (mirrors RFDETRDetector.get_class_id)."""
    with open(checkpoint / "det_lora_registry.json") as f:
        registry = json.load(f)
    added = registry["added_classes"]
    base = registry["base_num_classes"]
    return [base + added.index(c) for c in CLASSES]


def load_test_predictions(
    suite: str, variant: str, seed: int, data_dir: str, cache_dir: Path
) -> Tuple[List[Dict], List[Dict], List[int]]:
    run_key = f"{suite}_{variant}_seed{seed}"
    cache_path = cache_dir / f"{run_key}_test_nall.pt"
    checkpoint = Path(f"experiments/suites/{suite}/model_{variant}/seed_{seed}/det_lora/final")
    if cache_path.exists():
        payload = torch.load(str(cache_path), weights_only=False)
        # baseline caches (baseline_prediction_dump.py) carry their class ids inline
        class_ids = payload.get("class_ids") or _class_ids_from_registry(checkpoint)
        return payload["predictions"], payload["ground_truths"], class_ids

    detector = RFDETRDetector(variant=variant)
    det_lora = DetLoRA(detector=detector)
    det_lora.load_all(str(checkpoint))
    class_ids = [det_lora.get_class_id(c) for c in CLASSES]

    dataset = load_dataset_from_raw(
        raw_dir=data_dir,
        class_filter=CLASSES,
        split="test",
        class_id_offset=detector.base_num_classes,
        img_size=detector.resolution,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)
    preds, gts, _ = collect_det_lora_joint_predictions(det_lora, loader, CLASSES)
    preds = apply_shared_quality_calibrator(preds, det_lora.shared_quality_calibrator)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"predictions": preds, "ground_truths": gts}, str(cache_path))
    return preds, gts, class_ids


# --- per-image matching precompute ------------------------------------------
def _match_image(pred_boxes, pred_scores, gt_boxes, threshold) -> np.ndarray:
    """Greedy per-image matching (mirrors metrics._compute_class_metrics).

    Returns is_tp aligned to pred_scores (already score-sorted by caller)."""
    n_pred = len(pred_scores)
    tp = np.zeros(n_pred)
    if n_pred == 0 or len(gt_boxes) == 0:
        return tp
    iou = compute_iou_matrix(np.asarray(pred_boxes), np.asarray(gt_boxes))  # [P, G]
    matched = np.zeros(len(gt_boxes), dtype=bool)
    for i in range(n_pred):
        best_iou, best_g = 0.0, -1
        for g in range(len(gt_boxes)):
            if matched[g]:
                continue
            if iou[i, g] > best_iou:
                best_iou, best_g = iou[i, g], g
        if best_g >= 0 and best_iou >= threshold:
            tp[i] = 1.0
            matched[best_g] = True
    return tp


def precompute(
    predictions: List[Dict], ground_truths: List[Dict], class_ids: Sequence[int]
) -> Dict:
    """Per (class, threshold): score-sorted (tp, img_id) arrays + per-image GT counts."""
    n_images = len(predictions)
    ngt = {c: np.zeros(n_images) for c in class_ids}
    for i, gt in enumerate(ground_truths):
        labels = np.asarray(gt["labels"]).astype(int)
        for c in class_ids:
            ngt[c][i] = int((labels == c).sum())

    table: Dict[Tuple[int, float], Dict[str, np.ndarray]] = {}
    for c in class_ids:
        # gather per-image class-c detections once (boxes/scores), match per threshold
        per_img = []
        for i, pred in enumerate(predictions):
            labels = np.asarray(pred["labels"]).astype(int)
            mask = labels == c
            scores = np.asarray(pred["scores"])[mask].astype(float)
            boxes = np.asarray(pred["boxes"])[mask]
            order = np.argsort(-scores, kind="stable")
            per_img.append((scores[order], boxes[order]))
        gt_boxes_per_img = [
            np.asarray(g["boxes"])[np.asarray(g["labels"]).astype(int) == c] for g in ground_truths
        ]
        for t in THRESHOLDS:
            scores_all, tp_all, img_all = [], [], []
            for i, (sc, bx) in enumerate(per_img):
                tp = _match_image(bx, sc, gt_boxes_per_img[i], t)
                scores_all.append(sc)
                tp_all.append(tp)
                img_all.append(np.full(len(sc), i, dtype=int))
            scores_cat = np.concatenate(scores_all) if scores_all else np.zeros(0)
            tp_cat = np.concatenate(tp_all) if tp_all else np.zeros(0)
            img_cat = np.concatenate(img_all) if img_all else np.zeros(0, dtype=int)
            order = np.argsort(-scores_cat, kind="stable")
            table[(c, t)] = {"tp": tp_cat[order], "img": img_cat[order]}
    return {"table": table, "ngt": ngt, "n_images": n_images, "class_ids": list(class_ids)}


_RECALL_TH = np.linspace(0, 1, 101)


def _ap_curve(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO 101-point AP, vectorized (equivalent to metrics.compute_ap)."""
    if recall.shape[0] == 0:
        return 0.0
    p_env = np.maximum.accumulate(precision[::-1])[::-1]  # max precision at recall >= r
    idx = np.searchsorted(recall, _RECALL_TH, side="left")
    valid = idx < recall.shape[0]
    out = np.zeros(101)
    out[valid] = p_env[idx[valid]]
    return float(out.mean())


def _ap(precomp: Dict, c: int, t: float, weights: np.ndarray) -> Optional[float]:
    ngt_total = float((weights * precomp["ngt"][c]).sum())
    if ngt_total <= 0:
        return None
    entry = precomp["table"][(c, t)]
    det_w = weights[entry["img"]]
    tp_cum = np.cumsum(entry["tp"] * det_w)
    total_cum = np.cumsum(det_w)  # (tp+fp) weighted cumsum
    recall = tp_cum / ngt_total
    precision = tp_cum / np.maximum(total_cum, EPS)
    return _ap_curve(recall, precision)


def map_from_weights(precomp: Dict, weights: np.ndarray) -> Tuple[float, float]:
    ap50, ap5095 = [], []
    for c in precomp["class_ids"]:
        aps = [_ap(precomp, c, t, weights) for t in THRESHOLDS]
        if aps[0] is None:
            continue
        ap50.append(aps[0])  # THRESHOLDS[0] == 0.5
        ap5095.append(float(np.mean(aps)))
    return (float(np.mean(ap50)) if ap50 else 0.0, float(np.mean(ap5095)) if ap5095 else 0.0)


# --- bootstrap over shared test images, averaged across seeds ----------------
def bootstrap_suite(precomps: List[Dict], n_boot: int, rng: np.random.Generator) -> Dict:
    n = precomps[0]["n_images"]
    unit = np.ones(n)
    point50 = float(np.mean([map_from_weights(p, unit)[0] for p in precomps]))
    point5095 = float(np.mean([map_from_weights(p, unit)[1] for p in precomps]))

    boot50 = np.empty(n_boot)
    boot5095 = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        m50 = [map_from_weights(p, w)[0] for p in precomps]
        m5095 = [map_from_weights(p, w)[1] for p in precomps]
        boot50[b] = np.mean(m50)
        boot5095[b] = np.mean(m5095)

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    return {
        "mAP@0.5": {"point": point50, "ci95": ci(boot50)},
        "mAP@0.5:0.95": {"point": point5095, "ci95": ci(boot5095)},
        "n_images": n,
        "n_seeds": len(precomps),
        "n_boot": n_boot,
    }


def paired_delta(precomps_a, precomps_b, n_boot, rng) -> Dict:
    """CI of (suite_b - suite_a) seed-mean mAP under shared image resampling."""
    n = precomps_a[0]["n_images"]
    d50 = np.empty(n_boot)
    d5095 = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        a50 = np.mean([map_from_weights(p, w)[0] for p in precomps_a])
        b50 = np.mean([map_from_weights(p, w)[0] for p in precomps_b])
        a95 = np.mean([map_from_weights(p, w)[1] for p in precomps_a])
        b95 = np.mean([map_from_weights(p, w)[1] for p in precomps_b])
        d50[b] = b50 - a50
        d5095[b] = b95 - a95

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    return {
        "delta_mAP@0.5": {
            "point": float(np.mean(d50)),
            "ci95": ci(d50),
            "p_gt0": float((d50 > 0).mean()),
        },
        "delta_mAP@0.5:0.95": {
            "point": float(np.mean(d5095)),
            "ci95": ci(d5095),
            "p_gt0": float((d5095 > 0).mean()),
        },
    }


# --- joint bootstrap across variants (aggregate CIs from shared draws) -------
def bootstrap_joint(
    precomps: Dict[str, Dict[str, List[Dict]]],
    suites: Sequence[str],
    variants: Sequence[str],
    n_boot: int,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, np.ndarray]]:
    """One shared image draw per replicate, applied to every (suite, variant, seed).

    Returns stats[suite][variant] of shape (n_boot, 2) with columns
    (mAP@0.5, mAP@0.5:0.95), each entry the seed-mean under that replicate's
    weights. Because every cell of one replicate uses the same weights, aggregate
    statistics (mean over variants) and paired deltas (suite_b - suite_a) can be
    formed INSIDE each replicate before taking percentiles."""
    n = precomps[suites[0]][variants[0]][0]["n_images"]
    stats = {s: {v: np.empty((n_boot, 2)) for v in variants} for s in suites}
    for b in range(n_boot):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        for s in suites:
            for v in variants:
                per_seed = np.array([map_from_weights(p, w) for p in precomps[s][v]])
                stats[s][v][b] = per_seed.mean(axis=0)
        if (b + 1) % 200 == 0:
            print(f"[Bootstrap] joint replicate {b + 1}/{n_boot}")
    return stats


def _ci95(a: np.ndarray) -> List[float]:
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def _metric_block(point: float, boot: np.ndarray) -> Dict:
    return {"point": point, "ci95": _ci95(boot)}


def _delta_block(boot: np.ndarray) -> Dict:
    return {"point": float(np.mean(boot)), "ci95": _ci95(boot), "p_gt0": float((boot > 0).mean())}


def summarize_joint(
    precomps: Dict[str, Dict[str, List[Dict]]],
    stats: Dict[str, Dict[str, np.ndarray]],
    suites: Sequence[str],
    variants: Sequence[str],
    paired: bool,
) -> Dict:
    """Per-variant and aggregate CIs (and paired deltas) from the shared draws."""
    metrics = ["mAP@0.5", "mAP@0.5:0.95"]
    n = precomps[suites[0]][variants[0]][0]["n_images"]
    unit = np.ones(n)
    points = {
        s: {
            v: np.array([map_from_weights(p, unit) for p in precomps[s][v]]).mean(axis=0)
            for v in variants
        }
        for s in suites
    }

    payload: Dict = {"per_variant": {}, "aggregate": {}}
    for v in variants:
        payload["per_variant"][v] = {
            s: {
                m: _metric_block(float(points[s][v][k]), stats[s][v][:, k])
                for k, m in enumerate(metrics)
            }
            for s in suites
        }
    for s in suites:
        agg_boot = np.mean([stats[s][v] for v in variants], axis=0)  # (n_boot, 2)
        agg_point = np.mean([points[s][v] for v in variants], axis=0)
        payload["aggregate"][s] = {
            m: _metric_block(float(agg_point[k]), agg_boot[:, k]) for k, m in enumerate(metrics)
        }

    if paired and len(suites) >= 2:
        a, b = suites[0], suites[1]
        delta = {"baseline": a, "method": b, "per_variant": {}, "aggregate": {}}
        for v in variants:
            d = stats[b][v] - stats[a][v]
            delta["per_variant"][v] = {
                f"delta_{m}": _delta_block(d[:, k]) for k, m in enumerate(metrics)
            }
        d_agg = np.mean([stats[b][v] - stats[a][v] for v in variants], axis=0)
        delta["aggregate"] = {
            f"delta_{m}": _delta_block(d_agg[:, k]) for k, m in enumerate(metrics)
        }
        payload["paired_delta"] = delta
    return payload


def run_joint(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(12345)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    precomps: Dict[str, Dict[str, List[Dict]]] = {}
    did_sanity = False
    for suite in args.suites:
        precomps[suite] = {}
        for variant in args.variants:
            per_seed = []
            for seed in args.seeds:
                print(f"[Bootstrap] {suite} {variant} seed{seed}: loading predictions ...")
                preds, gts, class_ids = load_test_predictions(
                    suite, variant, seed, args.data_dir, cache_dir
                )
                pc = precompute(preds, gts, class_ids)
                if not args.no_sanity and not did_sanity:
                    _sanity_check(pc, preds, gts, class_ids)
                    did_sanity = True
                per_seed.append(pc)
            precomps[suite][variant] = per_seed

    stats = bootstrap_joint(precomps, args.suites, args.variants, args.n_boot, rng)
    payload = {
        "variants": args.variants,
        "seeds": args.seeds,
        "suites": args.suites,
        "n_boot": args.n_boot,
        "n_images": precomps[args.suites[0]][args.variants[0]][0]["n_images"],
        **summarize_joint(precomps, stats, args.suites, args.variants, args.paired),
    }

    for s in args.suites:
        m95 = payload["aggregate"][s]["mAP@0.5:0.95"]
        print(
            f"[Bootstrap] aggregate {s}: mAP@0.5:0.95={m95['point']:.4f} "
            f"[{m95['ci95'][0]:.4f}, {m95['ci95'][1]:.4f}]"
        )
    if "paired_delta" in payload:
        d = payload["paired_delta"]["aggregate"]["delta_mAP@0.5:0.95"]
        print(
            f"[Bootstrap] aggregate delta ({args.suites[1]} - {args.suites[0]}) "
            f"mAP@0.5:0.95={d['point']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] "
            f"P(>0)={d['p_gt0']:.3f}"
        )

    out_path = out_dir / "bootstrap_joint.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Bootstrap] saved {out_path}")


def _sanity_check(precomp: Dict, preds, gts, class_ids) -> None:
    unit = np.ones(precomp["n_images"])
    p50, p5095 = map_from_weights(precomp, unit)
    ref = compute_map(preds, gts, target_class_ids=class_ids)
    assert abs(p50 - ref["mAP@0.5"]) < 1e-3, (p50, ref["mAP@0.5"])
    assert abs(p5095 - ref["mAP@0.5:0.95"]) < 1e-3, (p5095, ref["mAP@0.5:0.95"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suites", nargs="+", default=["thesis_l40_symhn", "thesis_l40_main"])
    p.add_argument(
        "--variants",
        nargs="+",
        default=["nano"],
        help="one variant -> per-variant run (legacy output); several -> "
        "joint run with aggregate CIs formed inside each replicate",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--cache_dir", default="experiments/probes/gate_posthoc_cache")
    p.add_argument("--output_dir", default="experiments/probes/bootstrap_ci")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument(
        "--paired", action="store_true", help="report CI of the delta between the first two suites"
    )
    p.add_argument("--no_sanity", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.variants) > 1:
        run_joint(args)
        return
    run_single(args, args.variants[0])


def run_single(args: argparse.Namespace, variant: str) -> None:
    rng = np.random.default_rng(12345)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_precomps: Dict[str, List[Dict]] = {}
    results: Dict[str, Dict] = {}
    did_sanity = False
    for suite in args.suites:
        precomps = []
        for seed in args.seeds:
            print(f"[Bootstrap] {suite} {variant} seed{seed}: loading predictions ...")
            preds, gts, class_ids = load_test_predictions(
                suite, variant, seed, args.data_dir, cache_dir
            )
            pc = precompute(preds, gts, class_ids)
            if not args.no_sanity and not did_sanity:
                _sanity_check(pc, preds, gts, class_ids)
                did_sanity = True
            precomps.append(pc)
        suite_precomps[suite] = precomps
        res = bootstrap_suite(precomps, args.n_boot, np.random.default_rng(rng.integers(1 << 30)))
        results[suite] = res
        m50, m95 = res["mAP@0.5"], res["mAP@0.5:0.95"]
        print(
            f"[Bootstrap] {suite} {variant}: "
            f"mAP@0.5={m50['point']:.4f} [{m50['ci95'][0]:.4f}, {m50['ci95'][1]:.4f}] | "
            f"mAP@0.5:0.95={m95['point']:.4f} [{m95['ci95'][0]:.4f}, {m95['ci95'][1]:.4f}] "
            f"(n={res['n_images']}, seeds={res['n_seeds']}, B={res['n_boot']})"
        )

    payload = {"variant": variant, "seeds": args.seeds, "suites": results}
    if args.paired and len(args.suites) >= 2:
        a, b = args.suites[0], args.suites[1]
        delta = paired_delta(
            suite_precomps[a],
            suite_precomps[b],
            args.n_boot,
            np.random.default_rng(rng.integers(1 << 30)),
        )
        payload["paired_delta"] = {"baseline": a, "method": b, **delta}
        d = delta["delta_mAP@0.5:0.95"]
        print(
            f"[Bootstrap] delta ({b} - {a}) mAP@0.5:0.95="
            f"{d['point']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] P(>0)={d['p_gt0']:.3f}"
        )

    out_path = out_dir / f"bootstrap_{variant}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Bootstrap] saved {out_path}")


if __name__ == "__main__":
    main()
