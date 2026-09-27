import json
from pathlib import Path

import numpy as np

from data import load_unlabeled_data
from predict import predict_features
from train import train


def run_self_training(config, features, labels, base_checkpoint_root=None):
    """Generic self-training loop for single-head classification.

    1) predict unlabeled data with current fold ensemble
    2) keep predictions above confidence_threshold
    3) append pseudo-labeled samples to labeled data
    4) retrain into a new isolated checkpoint folder
    """
    if not bool(config.strategies.self_training.enabled):
        return None

    unlabeled = load_unlabeled_data(config)
    remaining_features = np.asarray(unlabeled)
    remaining_indices = np.arange(len(remaining_features))
    combined_features = np.asarray(features)
    combined_labels = np.asarray(labels)

    current_root = str(base_checkpoint_root or config.paths.path_to_fold_checkpoints)
    root = Path(config.paths.path_to_self_training)
    root.mkdir(parents=True, exist_ok=True)
    rounds_info = []

    for round_number in range(1, int(config.strategies.self_training.max_rounds) + 1):
        if len(remaining_features) == 0:
            break

        probabilities = predict_features(
            config,
            remaining_features,
            checkpoint_root=current_root,
            folds=config.split.folds_to_train,
        )
        if isinstance(probabilities, dict):
            raise ValueError("Base self-training implementation supports single-head classification only")

        confidence = probabilities.max(axis=1)
        pseudo_labels = probabilities.argmax(axis=1)
        mask = confidence >= float(config.strategies.self_training.confidence_threshold)
        selected = int(mask.sum())

        round_dir = root / f"round_{round_number}"
        round_dir.mkdir(parents=True, exist_ok=True)
        if bool(config.strategies.self_training.save_pseudo_labels):
            np.savez(
                round_dir / "pseudo_labels.npz",
                original_unlabeled_indices=remaining_indices[mask],
                labels=pseudo_labels[mask],
                confidence=confidence[mask],
            )

        info = {
            "round": round_number,
            "candidates": int(len(remaining_features)),
            "selected": selected,
            "threshold": float(config.strategies.self_training.confidence_threshold),
        }
        rounds_info.append(info)

        if config.logging.prints:
            print(
                f"Self-training round {round_number}: selected {selected}/{len(remaining_features)} "
                f"pseudo-labels at confidence >= {config.strategies.self_training.confidence_threshold}"
            )

        if selected < int(config.strategies.self_training.min_pseudo_samples):
            break

        combined_features = np.concatenate([combined_features, remaining_features[mask]], axis=0)
        combined_labels = np.concatenate([combined_labels, pseudo_labels[mask]], axis=0)

        checkpoint_root = round_dir / "folds"
        oof_path = round_dir / "oof_predictions.npz"
        train(
            config,
            combined_features,
            combined_labels,
            groups=None,
            fold_ids=None,
            checkpoint_root=checkpoint_root,
            oof_path=oof_path,
            run_tag=f"self_training_round_{round_number}",
            save_global_result=False,
            send_notifications=True,
            save_ensemble_artifacts=False,  # augmented folds cannot be used as honest base OOF
        )
        current_root = str(checkpoint_root)

        keep = ~mask
        remaining_features = remaining_features[keep]
        remaining_indices = remaining_indices[keep]

    manifest = {
        "enabled": True,
        "rounds": rounds_info,
        "final_checkpoint_root": current_root,
        "folds": [int(x) for x in config.split.folds_to_train],
        "final_train_samples": int(len(combined_features)),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
