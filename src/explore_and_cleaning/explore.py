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