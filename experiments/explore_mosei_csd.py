# One-off interactive script for inspecting CMU-MOSEI's raw .csd structure.
# Not part of the shipped pipeline -- run from the repo root if needed.

import h5py

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"[DATASET] {name}  shape={obj.shape}  dtype={obj.dtype}")
    else:
        print(f"[GROUP]   {name}")

print("=== LABELS ===")
f_labels = h5py.File("data/CMU-MOSEI/labels/CMU_MOSEI_Labels.csd", "r")
count = 0
def limited_print(name, obj):
    global count
    if count < 15:
        print_structure(name, obj)
        count += 1
f_labels.visititems(limited_print)

print("\n=== VISION ===")
f_vision = h5py.File("data/CMU-MOSEI/visuals/CMU_MOSEI_VisualFacet42.csd", "r")
count = 0
f_vision.visititems(limited_print)

print("\n=== VISUAL OPEN FACE ===")
f = h5py.File("data/CMU-MOSEI/visuals/CMU_MOSEI_VisualOpenFace2.csd", "r")
seq_name = list(f.keys())[0]
sample_vid = list(f[seq_name]["data"].keys())[0]
print(f[seq_name]["data"][sample_vid]["features"].shape)

intervals = f[seq_name]["data"][sample_vid]["intervals"][:]
duration = intervals[-1][1]  # end time of the last row
print(f"Duration: {duration:.1f}s, Rows: {len(intervals)}, Rate: {len(intervals)/duration:.1f} samples/sec")