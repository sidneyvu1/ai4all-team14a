import kagglehub

# Download latest version
# This takes a while so let it run while doing something else
path = kagglehub.dataset_download("samarwarsi/cmu-mosei", output_dir="./data")

print("Path to dataset files:", path)