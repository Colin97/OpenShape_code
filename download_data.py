import os
from multiprocessing import Pool
from huggingface_hub import hf_hub_download

def single_uncompress(file, dest="/mnt/data"):
	hf_hub_download(repo_id="OpenShape/openshape-training-data", filename=file, repo_type="dataset", local_dir=dest, local_dir_use_symlinks=False)
	if file.endswith(".tar.gz"):
		os.system("tar -xzf %s/%s -C %s" % (dest, file, dest))
	elif file.endswith(".zip"):
		os.system("unzip %s/%s -d %s" % (dest, file, dest))
	else:
		print("File extension not supported.")
	os.system("rm %s/%s" % (dest, file))


NUM_PROC = 16
pool = Pool(NUM_PROC)

pool.apply_async(single_uncompress, ("meta_data.zip", "./"))

files = ["3D-FUTURE.tar.gz", "ABO.tar.gz", "ShapeNet.tar.gz"]
for file in files:
	pool.apply_async(single_uncompress, (file,))

for i in range(160):
	pool.apply_async(single_uncompress, ("Objaverse/000-%03d.tar.gz" % i,))
	
pool.close()
pool.join()