import unittest
import torch
import os
import sys
import re
from argparse import Namespace

# 添加 simclr 所在路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simclr import SimCLR


class TestInfoNCELoss(unittest.TestCase):
    def setUp(self):
        self.args = Namespace()
        self.args.batch_size = 256
        self.args.n_views = 2
        self.args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.args.temperature = 0.5

        self.simclr = SimCLR(args=self.args, model=None, optimizer=None, scheduler=None)

        self.debug_dir = os.path.dirname(__file__)
        self.debug_files = [
            "../info_nce_debug_9.pt",
            "../info_nce_debug_42.pt",
            "../info_nce_debug_88.pt"
        ]

    def extract_seed(self, filename):
        match = re.search(r"debug_(\d+)\.pt", filename)
        return int(match.group(1)) if match else None

    def test_info_nce_loss_against_debug_files(self):
        for debug_file in self.debug_files:
            with self.subTest(file=debug_file):
                # 提取 seed 并设置
                seed = self.extract_seed(debug_file)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                path = os.path.join(self.debug_dir, debug_file)
                data = torch.load(path, map_location=self.args.device)

                features = data["features"].to(self.args.device)
                expected_logits = data["logits"].to(self.args.device)
                expected_labels = data["labels"].to(self.args.device)

                logits, labels = self.simclr.info_nce_loss(features)

                # 检查 shape 是否一致
                self.assertEqual(logits.shape, expected_logits.shape,
                                 f"{debug_file}: Logits shape mismatch")
                self.assertEqual(labels.shape, expected_labels.shape,
                                 f"{debug_file}: Labels shape mismatch")

                # 检查 logits 的数值是否近似
                self.assertTrue(torch.allclose(logits, expected_logits, atol=1e-4),
                                f"{debug_file}: Logits values differ")

                # 检查 labels 是否完全一致
                self.assertTrue(torch.equal(labels, expected_labels),
                                f"{debug_file}: Labels not equal")


if __name__ == '__main__':
    unittest.main()