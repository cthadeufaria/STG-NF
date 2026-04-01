"""
Evaluate a single STG-NF checkpoint on PoseLift, ShanghaiTech, and UBnormal
and print a combined results table.

Must be run from the stg_nf_official/ directory (scoring_utils uses relative
paths to data/ShanghaiTech/ and data/UBnormal/).

Example:
    python eval_all_datasets.py \
        --checkpoint ../artifacts/stg_nf/multi_runs/Multi/Mar31_2246/Mar31_2248__checkpoint.pth.tar \
        --seg_len 24 --K 8 --L 1 --R 3.0 --device mps
"""

import copy
import os
import numpy as np
import torch

from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from utils.train_utils import init_model_params
from utils.scoring_utils import score_metrics, smooth_scores, get_dataset_scores

PAPER_BASELINES = {
    'PoseLift':    {'auc_roc': 67.46, 'auc_pr': 84.06, 'eer': 39.0},
    'ShanghaiTech': {'auc_roc': 85.90, 'auc_pr': None,  'eer': None},
    'UBnormal':    {'auc_roc': 79.20, 'auc_pr': None,  'eer': None},
}


def evaluate_dataset(trainer, args_for_ds):
    dataset, loader = get_dataset_and_loader(args_for_ds, trans_list=trans_list, only_test=True)
    trainer.test_loader = loader['test']
    normality_scores = trainer.test()

    gt_arr, scores_arr = get_dataset_scores(normality_scores, dataset['test'].metadata, args=args_for_ds)
    scores_arr = smooth_scores(scores_arr)
    gt_np = np.concatenate(gt_arr)
    scores_np = np.concatenate(scores_arr)
    auc_roc, auc_pr, eer = score_metrics(scores_np, gt_np)
    return auc_roc, auc_pr, eer, scores_np.shape[0]


def main():
    parser = init_parser()
    base_args = parser.parse_args()

    if base_args.seed == 999:
        base_args.seed = torch.initial_seed()
        np.random.seed(0)

    # Initialise with PoseLift to build the model (checkpoint determines architecture)
    args0 = copy.deepcopy(base_args)
    args0.dataset = 'PoseLift'
    args0.data_dir = os.path.join(base_args.data_dir, 'PoseLift', 'Pickle_files')
    args0, _ = init_sub_args(args0)

    # Need a dummy dataset to derive model_args (input shape etc.)
    dataset0, loader0 = get_dataset_and_loader(args0, trans_list=trans_list, only_test=True)
    model_args = init_model_params(args0, dataset0)
    model = STG_NF(**model_args)
    trainer = Trainer(
        args0, model, None, loader0['test'],
        optimizer_f=init_optimizer(args0.model_optimizer, lr=args0.model_lr),
        scheduler_f=init_scheduler(args0.model_sched, lr=args0.model_lr, epochs=args0.epochs),
    )
    trainer.load_checkpoint(vars(base_args)['checkpoint'])

    # data_dir mappings per dataset:
    # - PoseLift: init_sub_args uses data_dir/Train and data_dir/Test directly
    # - ShanghaiTech/UBnormal: init_sub_args uses data_dir/<Dataset>/pose/test/
    #   which resolves relative to stg_nf_official/, so use 'data/'
    poselift_data_dir = os.path.join(base_args.data_dir, 'PoseLift', 'Pickle_files')
    dataset_data_dirs = {
        'PoseLift':     poselift_data_dir,
        'ShanghaiTech': 'data/',
        'UBnormal':     'data/',
    }

    results = {}
    for ds in ['PoseLift', 'ShanghaiTech', 'UBnormal']:
        print(f"\n--- Evaluating on {ds} ---")
        args_ds = copy.deepcopy(base_args)
        args_ds.dataset = ds
        args_ds.data_dir = dataset_data_dirs[ds]
        args_ds, _ = init_sub_args(args_ds)
        auc_roc, auc_pr, eer, n = evaluate_dataset(trainer, args_ds)
        results[ds] = (auc_roc, auc_pr, eer, n)

    # Print table
    print("\n\n" + "=" * 75)
    print(f"{'Dataset':<16} {'AUC-ROC':>10} {'AUC-PR':>10} {'EER':>10} {'Samples':>9}  {'Paper AUC-ROC':>14}")
    print("-" * 75)
    for ds, (auc_roc, auc_pr, eer, n) in results.items():
        paper = PAPER_BASELINES[ds]
        paper_str = f"{paper['auc_roc']:.2f}%" if paper['auc_roc'] else "—"
        print(f"{ds:<16} {auc_roc*100:>9.4f}% {auc_pr*100:>9.4f}% {eer*100:>9.4f}% {n:>9}  {paper_str:>14}")
    print("=" * 75)


if __name__ == '__main__':
    main()
