import os
import json
import math
from src.data.dataset.loader import AISTDataset

genre_list = ["gBR", "gPO", "gLO", "gMH", "gLH", "gHO", "gWA", "gKR", "gJS", "gJB"]

class SnippetDataset():
    def __del__(self, ):
        del self.official_loader
        
    def __init__(self, TYPE, DATA_DIR, GENRE, USE_OPTIM, LENGTH, OVERLAP, SPLIT, BS, MIN_LENGTH, MAX_LENGTH):
        self.type = TYPE
        self.data_dir = DATA_DIR
        self.genre = genre_list[:GENRE]
        self.use_optim = USE_OPTIM
        self.length = LENGTH
        self.overlap = OVERLAP
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
        bad_vids = self.official_loader.filter_file #+ ["gHO_sFM_cAll_d20_mHO5_ch13", ]
        self.validation_split = [_ for _ in self.validation_split if _ not in bad_vids]
        self.train_split = [_ for _ in self.train_split if _ not in bad_vids]

        print(f"{self.genre}-style dances loaded with {len(self.train_split)} training videos and {len(self.validation_split)} validation videos")
        # TODO: change to use official splits, 868 train, 70 validation, 470 test
        
    def get_train(self, ):
        for seq_name in self.train_split:
            full_data = self.official_loader.load_keypoint3d(seq_name, use_optim=self.use_optim)
            if full_data.max() > 500:
                print(f"in {seq_name} for train, max number being {full_data.max()}")
            duration = full_data.shape[0]
            hop = 1 - self.overlap
            total_pieces = math.floor((duration / self.length - 1) / hop) # + 1
            for i in range(total_pieces):
                start = int(i * hop * self.length)
                tbr = full_data[start: start + self.length]
                yield tbr

    def get_validation(self, ): 
        for seq_name in self.validation_split:
            per_seq = []
            full_data = self.official_loader.load_keypoint3d(seq_name, use_optim=self.use_optim)
            if full_data.max() > 500:
                print(f"in {seq_name} for val, max number being {full_data.max()}")
            duration = full_data.shape[0]
            for i in range(duration // self.length):
                per_seq.append(full_data[i * self.length: (i + 1) * self.length])
            yield {seq_name: per_seq}
