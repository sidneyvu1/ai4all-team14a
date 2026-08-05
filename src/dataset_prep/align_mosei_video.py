"""
Aligns CMU-MOSEI FACET vision features to sentiment + 6-emotion labels
directly from raw CSD files. No MMSA package, no CMU-MultimodalSDK install needed.

Requires only: h5py, numpy

Run this once. It saves a single .npz file you'll reuse for all modeling work.
"""

import h5py
import numpy as np

VISION_PATH = "data/CMU-MOSEI/visuals/CMU_MOSEI_VisualOpenFace2.csd"
LABELS_PATH = "data/CMU-MOSEI/labels/CMU_MOSEI_Labels.csd"
OUTPUT_PATH = "data/CMU-MOSEI/mosei_visualopenface_only_processed.npz"

SEQ_LEN = 50       # standard community sequence length
FEATURE_DIM = 713   # FACET feature dimensionality (confirm this matches your data below)


def load_csd(path):
    f = h5py.File(path, "r")
    seq_name = list(f.keys())[0]
    return f[seq_name]["data"], seq_name


def get_vision_slice(vision_intervals, vision_features, start, end):
    """Grab rows of the vision feature stream whose timestamps fall inside [start, end]."""
    mask = (vision_intervals[:, 0] >= start) & (vision_intervals[:, 1] <= end)
    return vision_features[mask]


def pad_or_truncate(seq, target_len=SEQ_LEN, feat_dim=FEATURE_DIM):
    if len(seq) == 0:
        return np.zeros((target_len, feat_dim))
    if len(seq) >= target_len:
        return seq[:target_len]
    pad = np.zeros((target_len - len(seq), feat_dim))
    return np.vstack([seq, pad])


def main():
    print("Loading CSD files...")
    vision_data, vision_seq_name = load_csd(VISION_PATH)
    labels_data, labels_seq_name = load_csd(LABELS_PATH)

    print(f"Vision sequence name: {vision_seq_name}")
    print(f"Labels sequence name: {labels_seq_name}")
    print(f"Number of videos in vision file: {len(vision_data)}")
    print(f"Number of videos in labels file: {len(labels_data)}")

    # Peek at one sample to confirm shapes match expectations
    sample_vid = list(labels_data.keys())[0]
    sample_label_feats = labels_data[sample_vid]["features"][:]
    sample_label_intervals = labels_data[sample_vid]["intervals"][:]
    print(f"\nSample video ID: {sample_vid}")
    print(f"Label features shape (per video): {sample_label_feats.shape}")
    print(f"Label intervals shape (per video): {sample_label_intervals.shape}")
    print(f"First label row (should be ~7 values: sentiment + 6 emotions): {sample_label_feats[0]}")

    if sample_vid in vision_data:
        sample_vision_feats = vision_data[sample_vid]["features"][:]
        print(f"Vision features shape (per video): {sample_vision_feats.shape}")
        actual_feat_dim = sample_vision_feats.shape[1]
        if actual_feat_dim != FEATURE_DIM:
            print(f"\n*** WARNING: actual FACET dim is {actual_feat_dim}, not {FEATURE_DIM}. "
                  f"Update FEATURE_DIM at top of script and rerun. ***\n")
    else:
        print("*** WARNING: sample video not found in vision file, ID formats may differ. ***")

    print("\nStarting full alignment loop...")
    X, y, ids = [], [], []
    skipped_no_vision = 0
    skipped_empty_slice = 0

    for vid in labels_data.keys():
        if vid not in vision_data:
            skipped_no_vision += 1
            continue

        label_feats = labels_data[vid]["features"][:]
        label_intervals = labels_data[vid]["intervals"][:]
        vision_feats = vision_data[vid]["features"][:]
        vision_intervals = vision_data[vid]["intervals"][:]

        for i in range(len(label_feats)):
            start, end = label_intervals[i]
            clip_vision = get_vision_slice(vision_intervals, vision_feats, start, end)
            if len(clip_vision) == 0:
                skipped_empty_slice += 1
                continue
            clip_vision = pad_or_truncate(clip_vision, feat_dim=vision_feats.shape[1])
            X.append(clip_vision)
            y.append(label_feats[i])
            ids.append(f"{vid}_{i}")

    X = np.array(X)
    y = np.array(y)

    print(f"\nDone.")
    print(f"Final X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
    print(f"Videos skipped (no matching vision entry): {skipped_no_vision}")
    print(f"Clips skipped (empty vision slice): {skipped_empty_slice}")

    np.savez(OUTPUT_PATH, X=X, y=y, ids=np.array(ids))
    print(f"\nSaved to {OUTPUT_PATH}")

    print("\nLabel columns, in order: [sentiment, happy, sad, anger, surprise, disgust, fear]")



if __name__ == "__main__":
    main()