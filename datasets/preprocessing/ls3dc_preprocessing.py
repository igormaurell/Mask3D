import re
import os
import numpy as np
from fire import Fire
from natsort import natsorted
from loguru import logger
import pandas as pd
import h5py
import pickle

from datasets.preprocessing.base_preprocessing import BasePreprocessing

class LS3DCPreprocessing(BasePreprocessing):
    def __init__(
            self,
            data_dir: str = "../../data/raw/ls3dc",
            save_dir: str = "../../data/processed/ls3dc",
            modes: tuple = ("train", "validation"),
            n_jobs: int = -1,
            min_points: int = 10000
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)

        self.min_points = min_points

        MODE_CSV_MAP = {
            'train': 'train',
            'validation': 'test',
            'test': ''
        }

        #CLASS_LABELS = ['None', 'Plane', 'Cylinder']
        #VALID_CLASS_IDS = np.array([1, 2])

        self.class_map = {
            'none': 0,
            'plane': 1,
            'cylinder': 2
        }

        self.color_map = [
            [255, 255, 255], #None
            [255, 0, 0],   # Plane
            [0, 0, 255]]   # Cylinder
   

        self.create_label_database()

        for mode in self.modes:
            filenames = list(pd.read_csv(self.data_dir / '{}_models.csv'.format(MODE_CSV_MAP[mode])))
            filepaths = [str(self.data_dir / f) for f in filenames]
            self.files[mode] = natsorted(filepaths)

    def create_label_database(self):
        label_database = dict()
        for class_name, class_id in self.class_map.items():
            label_database[class_id] = {
                'color': self.color_map[class_id],
                'name': class_name,
                'validation': True
            }

        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database

    def process_file(self, filepath, mode):
        """process_file.

        Please note, that for obtaining segmentation labels ply files were used.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        filebase = {
            "filepath": filepath,
            "scene": filepath.split("/")[-1],
            "raw_filepath": str(filepath),
            "file_len": -1,
        }

        with h5py.File(filepath, 'r') as h5_file:
            xyz = h5_file['noisy_points'][()] if 'noisy_points' in h5_file.keys() else None
            normals = h5_file['gt_normals'][()] if 'gt_normals' in h5_file.keys() else None
            instance_label = h5_file['gt_labels'][()] if 'gt_labels' in h5_file.keys() else None

            found_soup_ids = []
            soup_id_to_key = {}
            soup_prog = re.compile('(.*)_soup_([0-9]+)$')
            for key in list(h5_file.keys()):
                m = soup_prog.match(key)
                if m is not None:
                    soup_id = int(m.group(2))
                    found_soup_ids.append(soup_id)
                    soup_id_to_key[soup_id] = key

            features_data = []            
            found_soup_ids.sort()
            for i in range(len(found_soup_ids)):
                g = h5_file[soup_id_to_key[i]]
                meta = pickle.loads(g.attrs['meta'])
                features_data.append(meta)

            semantic_label = np.array([ self.class_map['none'] if inst == -1 or features_data[inst]['type'].lower() not in self.class_map.keys() else self.class_map[features_data[inst]['type']] for inst in instance_label])

            instance_label = instance_label[..., None]
            semantic_label = semantic_label[..., None]

        rgb = np.ones(xyz.shape, dtype=xyz.dtype)
        points = np.hstack((xyz, rgb, semantic_label, instance_label))

        filebase["raw_segmentation_filepath"] = ""

        # add segment id as additional feature (DUMMY)
        points = np.hstack((points, normals, np.ones(points.shape[0])[..., None]))  # segment

        points = points[:, [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 6, 7]]  # move segments after RGB

        points[:, :3] = points[:, :3] - points[:, :3].min(0)

        points = points.astype(np.float32)

        if mode == "test":
            points = points[:, :-2]

        file_len = len(points)
        filebase["file_len"] = file_len

        processed_filepath = self.save_dir / mode / f"{filebase['scene'].replace('.txt', '')}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        if mode in ["validation", "test"]:
            blocks = self.splitPointCloud(points)

            filebase["instance_gt_filepath"] = []
            filebase["filepath_crop"] = []
            for block_id, block in enumerate(blocks):
                if len(block) > 10000:
                    if mode == "validation":
                        new_instance_ids = np.unique(block[:, -1], return_inverse=True)[1]

                        assert new_instance_ids.shape[0] == block.shape[0]
                        # == 0 means -1 == no instance
                        # new_instance_ids[new_instance_ids == 0]
                        assert new_instance_ids.max() < 1000, "we cannot encode when there are more than 999 instances in a block"

                        gt_data = (block[:, -2]) * 1000 + new_instance_ids

                        processed_gt_filepath = self.save_dir / "instance_gt" / mode / f"{filebase['scene'].replace('.txt', '')}_{block_id}.txt"
                        if not processed_gt_filepath.parent.exists():
                            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
                        np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
                        filebase["instance_gt_filepath"].append(str(processed_gt_filepath))

                    processed_filepath = self.save_dir / mode / f"{filebase['scene'].replace('.txt', '')}_{block_id}.npy"
                    if not processed_filepath.parent.exists():
                        processed_filepath.parent.mkdir(parents=True, exist_ok=True)
                    np.save(processed_filepath, block.astype(np.float32))
                    filebase["filepath_crop"].append(str(processed_filepath))
                else:
                    print("block was smaller than 1000 points")
                    assert False

        filebase["color_mean"] = [
            float((points[:, 3] / 255).mean()),
            float((points[:, 4] / 255).mean()),
            float((points[:, 5] / 255).mean()),
        ]
        filebase["color_std"] = [
            float(((points[:, 3] / 255) ** 2).mean()),
            float(((points[:, 4] / 255) ** 2).mean()),
            float(((points[:, 5] / 255) ** 2).mean()),
        ]
        return filebase

    def compute_color_mean_std(
            self, train_database_path: str = "./data/processed/ls3dc/train_database.yaml"
    ):
        train_database = self._load_yaml(train_database_path)
        color_mean, color_std = [], []
        for sample in train_database:
            color_std.append(sample["color_std"])
            color_mean.append(sample["color_mean"])

        color_mean = np.array(color_mean).mean(axis=0)
        color_var = np.array(color_std).mean(axis=0) - color_mean ** 2
        color_std = np.sqrt(np.maximum(0, color_var))
        feats_mean_std = {
            "mean": [float(each) for each in color_mean],
            "std": [float(each) for each in color_std],
        }
        self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)

    def splitPointCloud(self, cloud, size=6.0, stride=6):
        limitMax = np.amax(cloud[:, 0:3], axis=0)
        width = int(np.ceil((limitMax[0] - size) / stride)) + 1
        depth = int(np.ceil((limitMax[1] - size) / stride)) + 1
        cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]
        blocks = []
        for (x, y) in cells:
            xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
            ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
            cond = xcond & ycond
            block = cloud[cond, :]
            blocks.append(block)
        return blocks


    @logger.catch
    def fix_bugs_in_labels(self):
        pass

    def _parse_scene_subscene(self, name):
        scene_match = re.match(r"scene(\d{4})_(\d{2})", name)
        return int(scene_match.group(1)), int(scene_match.group(2))


if __name__ == "__main__":
    Fire(LS3DCPreprocessing)
