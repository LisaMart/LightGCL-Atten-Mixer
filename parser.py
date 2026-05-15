#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Argument parser for LightGCL-no-GNN variant.
Provides default hyper-parameters for training, evaluation, and SVD configuration.
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="LightGCL-no-GNN – hyper-parameters")

    # -------------------- data --------------------
    parser.add_argument(
        "--data",
        default="diginetica",
        type=str,
        help="dataset folder name (diginetica / yoochoose1_64 / ...)",
    )

    # -------------------- training --------------------
    parser.add_argument("--epoch", default=30, type=int, help="number of training epochs")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate for Adam")
    parser.add_argument("--decay", default=0.99, type=float, help="learning-rate decay (currently unused)")
    parser.add_argument("--batch", default=256, type=int, help="users per batch at test time")
    parser.add_argument("--inter_batch", default=4096, type=int, help="interactions per batch at training time")

    # -------------------- model --------------------
    parser.add_argument("--d", default=64, type=int, help="embedding dimension (|d| in paper)")
    parser.add_argument("--q", default=5, type=int, help="SVD rank for low-rank augmentation")
    parser.add_argument("--gnn_layer", default=2, type=int, help="number of SVD layers (compatibility)")
    parser.add_argument("--dropout", default=0.0, type=float, help="edge dropout rate (unused – no GNN)")
    parser.add_argument("--temp", default=0.2, type=float, help="Info-NCE temperature")
    parser.add_argument("--lambda1", default=0.2, type=float, help="weight of contrastive (SSL) loss")
    parser.add_argument("--lambda2", default=1e-7, type=float, help="L2 regularisation weight")

    # -------------------- hardware --------------------
    parser.add_argument("--cuda", default="0", type=str, help="GPU id to use ('cpu' for CPU)")

    # -------------------- data capping (for quick debug) --------------------
    parser.add_argument("--max_users", type=int, default=50_000, help="maximum number of users")
    parser.add_argument("--max_items", type=int, default=100_000, help="maximum number of items")

    # -------------------- misc --------------------
    parser.add_argument("--note", default=None, type=str, help="free-text note for experiments")

    return parser.parse_args()

# singleton for import in other modules
args = parse_args()
