import os, json
from src.data.dataset.loader import AISTDataset

genre_list = ["gBR", "gPO", "gLO", "gMH", "gLH", "gHO", "gWA", "gKR", "gJS", "gJB"]

# 17 joints of COCO:
# 0 - nose,  1 - left_eye,  2 - right_eye, 3 - left_ear, 4 - right_ear
# 5 - left_shoulder, 6 - right_shoulder, 7 - left_elbow, 8 - right_elbow, 9 - left_wrist, 10 - right_wrist
# 11 - left_hip, 12 - right_hip, 13 - left_knee, 14 - right_knee. 15 - left_ankle, 16 - right_ankle


class SkeletonDataset():
    name = 'stl10'

    def __del__(self, ):
        if hasattr(self, "official_loader"):
            del self.official_loader

    def __init__(self, DATA_DIR, GENRE, SPLIT):
        self.data_dir = DATA_DIR
        self.genre = genre_list[:GENRE]
        self.split = SPLIT
        assert os.path.isdir(self.data_dir)
        self.official_loader = AISTDataset(os.path.join(self.data_dir, "annotations"))
        all_seq = []
        for seq_name_np in self.official_loader.mapping_seq2env.keys():
            seq_name = str(seq_name_np)
            if seq_name.split("_")[0] in self.genre:
                all_seq.append(seq_name)

        with open(os.path.join("src/data/dataset", "splits", f"split_wseed_{self.split}.json"), "r") as f:
            ldd = json.load(f)
            self.validation_split = []
            for genre in self.genre:
                self.validation_split += ldd[genre]

        self.train_split = [_ for _ in all_seq if _ not in self.validation_split]
        # remove files officially deemed as broken
        bad_vids = self.official_loader.filter_file + ["gHO_sFM_cAll_d20_mHO5_ch13", ]
        self.validation_split = [_ for _ in self.validation_split if _ not in bad_vids]
        self.train_split = [_ for _ in self.train_split if _ not in bad_vids]

        print(
            f"{self.genre}-style dances loaded with {len(self.train_split)} training videos and {len(self.validation_split)} validation videos")
