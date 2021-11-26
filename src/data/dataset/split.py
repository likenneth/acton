import numpy as np
import json
import os
from pathlib import Path
from loader import AISTDataset

genre_list = ["gBR", "gPO", "gLO", "gMH", "gLH", "gHO", "gWA", "gKR", "gJS", "gJB"]
data_dir = Path("../../../../aistplusplus")
assert data_dir.exists()
validation_number = 42

def main():
    seed = 4321
    np.random.seed(seed)
    
    offical_loader = AISTDataset(data_dir / "annotations")
    seq_container = {k: [] for k in genre_list}
    for seq_name in offical_loader.mapping_seq2env.keys():
        genre = seq_name.split("_")[0]
        seq_container[genre].append(seq_name)
    
    val = {k: [] for k in genre_list}
    total = 0
    for genre in genre_list:
        total += len(seq_container[genre])
        lucky = np.random.permutation(len(seq_container[genre]))[:validation_number].tolist()
        val[genre] = [seq_container[genre][_] for _ in lucky]
    print(f"In sum has {total} sequences")

    dump_path = os.path.join("./", "splits", f"split_wseed_{seed}.json")
    with open(dump_path, "w") as f:
        json.dump(val, f)
    print(f"Dumped validation set generated from seed {seed} to {dump_path}")
    

if __name__ == "__main__":
    main()