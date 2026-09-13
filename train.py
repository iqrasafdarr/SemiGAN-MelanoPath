import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm

from data.loaders import (
    BreakHisDataset,
    get_patient_train_val_test_split,
    create_loaders,
)
from eval.evaluator import evaluate_classifier, save_metrics
from models.architectures import Generator, Discriminator
from models.losses import (
    VAT,
    SupervisedCE,
    RotationPredictionLoss,
    AdversarialLoss,
    FeatureMatchingLoss,
    NoiseInjection,
    create_rotations,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemiGANTrainer:
    """SemiGAN-MelanoPath v2 trainer with patient-level evaluation."""

    def __init__(self, config_path, exp_name, label_pct, seed):
        self.seed = seed
        self.label_pct = label_pct
        self.exp_name = exp_name

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info("Device: %s", self.device)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self._set_seed()
        self._setup_paths()
        self._init_models()
        self._init_optimizers()
        self._init_losses()

        self.best_val_f1 = -1.0
        self.best_epoch = None

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _setup_paths(self):
        self.exp_dir = Path(
            f"runs/{self.exp_name}_lbl{self.label_pct}_s{self.seed}"
        )

        self.exp_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.sample_dir = self.exp_dir / "samples"

        self.ckpt_dir.mkdir(exist_ok=True)
        self.sample_dir.mkdir(exist_ok=True)

    def _init_models(self):
        self.G = Generator(
            z_dim=self.config["generator"]["z_dim"],
            base_channels=self.config["generator"]["base_channels"],
            use_spectral_norm=self.config["generator"]["spectral_norm"],
        ).to(self.device)

        self.D = Discriminator(
            base_channels=self.config["discriminator"]["base_channels"],
            num_classes=self.config["data"]["num_classes"],
            num_rotations=self.config["discriminator"]["num_rotations"],
            classifier_dropout=self.config["discriminator"]["classifier_dropout"],
            use_spectral_norm=self.config["discriminator"]["spectral_norm"],
        ).to(self.device)

        self.noise_layer = NoiseInjection(
            self.config["losses"]["noise_sigma"]
        )

    def _init_optimizers(self):
        lr = self.config["optimizer"]["lr"]
        betas = tuple(self.config["optimizer"]["betas"])

        self.opt_G = optim.Adam(
            self.G.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=self.config["optimizer"]["g_weight_decay"],
        )

        self.opt_D = optim.Adam(
            self.D.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=self.config["optimizer"]["d_weight_decay"],
        )

    def _init_losses(self):
        self.supervised_ce = SupervisedCE()
        self.rotation_ce = RotationPredictionLoss()
        self.adversarial_loss = AdversarialLoss()
        self.feature_match = FeatureMatchingLoss()

        self.vat = VAT(
            eps=self.config["losses"]["vat_eps"],
            beta=self.config["losses"]["vat_beta"],
        )

    def train(self, dataset):
        """
        Train using patient-level train/validation/test separation.

        Only the training patients participate in the labeled/unlabeled
        split. Validation and test patients are completely isolated.
        """

        split = get_patient_train_val_test_split(
            dataset,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=self.seed,
        )

        train_indices = split["train_indices"]
        val_indices = split["val_indices"]
        test_indices = split["test_indices"]

        logger.info("=" * 70)
        logger.info("PATIENT-LEVEL DATA SPLIT")
        logger.info("=" * 70)

        logger.info(
            "Train: %d patients | %d images",
            len(split["train_patients"]),
            len(train_indices),
        )

        logger.info(
            "Validation: %d patients | %d images",
            len(split["val_patients"]),
            len(val_indices),
        )

        logger.info(
            "Test: %d patients | %d images",
            len(split["test_patients"]),
            len(test_indices),
        )

        # ------------------------------------------------------------------
        # CRITICAL:
        # The labeled/unlabeled split happens ONLY inside TRAIN.
        # ------------------------------------------------------------------

        train_split_dataset = BreakHisDataset(
            root_dir=dataset.root_dir,
            transform=dataset.transform,
            indices=train_indices,
        )

        split_info = self._get_train_label_split(
            train_split_dataset
        )

        logger.info(
            "Label budget: %s%% | Labeled images: %d | "
            "Unlabeled images: %d",
            self.label_pct,
            split_info["n_labeled"],
            split_info["n_unlabeled"],
        )

        labeled_loader, unlabeled_loader = create_loaders(
            train_split_dataset,
            split_info["labeled_indices"],
            split_info["unlabeled_indices"],
            batch_size=self.config["training"]["batch_size"],
        )

        num_epochs = self.config["training"]["num_epochs"]
        n_critic = self.config["training"]["n_critic"]

        results = {
            "epoch": [],
            "train_loss_D": [],
            "train_loss_G": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_roc_auc": [],
        }

        # ------------------------------------------------------------------
        # TRAIN
        # ------------------------------------------------------------------

        for epoch in range(num_epochs):
            epoch_loss_D = 0.0
            epoch_loss_G = 0.0
            n_batches = 0

            self.D.train()
            self.G.train()

            labeled_iter = iter(labeled_loader)
            unlabeled_iter = iter(unlabeled_loader)

            max_batches = max(
                len(labeled_loader),
                len(unlabeled_loader),
            )

            pbar = tqdm(
                range(max_batches),
                desc=f"Epoch {epoch + 1}/{num_epochs}",
            )

            for _ in pbar:

                try:
                    labeled_batch = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    labeled_batch = next(labeled_iter)

                try:
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled_batch = next(unlabeled_iter)

                for _ in range(n_critic):
                    loss_D = self._train_discriminator(
                        labeled_batch,
                        unlabeled_batch,
                    )
                    epoch_loss_D += loss_D.item()

                loss_G = self._train_generator()
                epoch_loss_G += loss_G.item()

                n_batches += 1

                pbar.set_postfix(
                    D=f"{loss_D.item():.4f}",
                    G=f"{loss_G.item():.4f}",
                )

            epoch_loss_D /= max(
                1,
                n_batches * n_critic
            )

            epoch_loss_G /= max(
                1,
                n_batches
            )

            # ------------------------------------------------------------------
            # VALIDATION
            # ------------------------------------------------------------------

            val_metrics = evaluate_classifier(
                model=self.D,
                dataset=dataset,
                indices=val_indices,
                device=self.device,
                batch_size=self.config["training"]["batch_size"],
                num_workers=0,
            )

            results["epoch"].append(epoch + 1)
            results["train_loss_D"].append(epoch_loss_D)
            results["train_loss_G"].append(epoch_loss_G)

            results["val_accuracy"].append(
                val_metrics["accuracy"]
            )
            results["val_precision"].append(
                val_metrics["precision"]
            )
            results["val_recall"].append(
                val_metrics["recall"]
            )
            results["val_f1"].append(
                val_metrics["f1"]
            )
            results["val_roc_auc"].append(
                val_metrics["roc_auc"]
            )

            logger.info(
                "Epoch %d | D=%.4f | G=%.4f | "
                "Val Acc=%.4f | Val F1=%.4f | Val AUC=%s",
                epoch + 1,
                epoch_loss_D,
                epoch_loss_G,
                val_metrics["accuracy"],
                val_metrics["f1"],
                (
                    f"{val_metrics['roc_auc']:.4f}"
                    if val_metrics["roc_auc"] is not None
                    else "N/A"
                ),
            )

            # ------------------------------------------------------------------
            # BEST CHECKPOINT
            # ------------------------------------------------------------------

            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.best_epoch = epoch + 1

                self._save_best_checkpoint(
                    epoch,
                    val_metrics,
                )

                logger.info(
                    "New best validation F1: %.4f",
                    self.best_val_f1,
                )

            if (
                epoch + 1
            ) % self.config["logging"]["sample_interval"] == 0:
                self._save_samples(epoch)

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)

        # ------------------------------------------------------------------
        # FINAL TEST
        # ------------------------------------------------------------------

        logger.info("=" * 70)
        logger.info("LOADING BEST CHECKPOINT")
        logger.info("=" * 70)

        best_path = self.ckpt_dir / "best.pt"

        if best_path.exists():
            checkpoint = torch.load(
                best_path,
                map_location=self.device,
            )

            self.G.load_state_dict(
                checkpoint["G"]
            )

            self.D.load_state_dict(
                checkpoint["D"]
            )

        logger.info(
            "Best validation epoch: %s",
            self.best_epoch,
        )

        test_metrics = evaluate_classifier(
            model=self.D,
            dataset=dataset,
            indices=test_indices,
            device=self.device,
            batch_size=self.config["training"]["batch_size"],
            num_workers=0,
        )

        test_path = self.exp_dir / "test_metrics.json"

        save_metrics(
            test_metrics,
            test_path,
        )

        logger.info("=" * 70)
        logger.info("FINAL TEST RESULTS")
        logger.info("=" * 70)

        logger.info(
            "Accuracy:  %.4f",
            test_metrics["accuracy"],
        )
        logger.info(
            "Precision: %.4f",
            test_metrics["precision"],
        )
        logger.info(
            "Recall:    %.4f",
            test_metrics["recall"],
        )
        logger.info(
            "F1:        %.4f",
            test_metrics["f1"],
        )
        logger.info(
            "ROC-AUC:   %s",
            (
                f"{test_metrics['roc_auc']:.4f}"
                if test_metrics["roc_auc"] is not None
                else "N/A"
            ),
        )

        # Save complete experiment results.
        complete_results = {
            "label_pct": self.label_pct,
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "best_val_f1": self.best_val_f1,
            "train_patients": len(split["train_patients"]),
            "val_patients": len(split["val_patients"]),
            "test_patients": len(split["test_patients"]),
            "train_images": len(train_indices),
            "val_images": len(val_indices),
            "test_images": len(test_indices),
            "training_history": results,
            "test_metrics": test_metrics,
        }

        with open(
            self.exp_dir / "results.json",
            "w",
        ) as f:
            json.dump(
                complete_results,
                f,
                indent=2,
            )

        return complete_results

    def _get_train_label_split(self, dataset):
        """
        Create a patient-level labeled/unlabeled split inside the
        training set only.

        Returns integer SAMPLE INDICES, not patient IDs.
        """

        rng = np.random.default_rng(self.seed)

        patient_to_indices = {}

        for idx, sample in enumerate(dataset.samples):
            patient_id = sample["patient_id"]

            if patient_id not in patient_to_indices:
                patient_to_indices[patient_id] = []

            patient_to_indices[patient_id].append(idx)

        patients = list(patient_to_indices.keys())

        if not patients:
            raise RuntimeError(
                "No patients were found in the training dataset."
            )

        patient_labels = {}

        for patient_id, indices in patient_to_indices.items():
            labels = [
                dataset.samples[i]["label"]
                for i in indices
            ]

            unique_labels = set(labels)

            if len(unique_labels) != 1:
                raise RuntimeError(
                    f"Patient {patient_id} contains multiple labels: "
                    f"{unique_labels}"
                )

            patient_labels[patient_id] = labels[0]

        if self.label_pct >= 100:
            labeled_patients = set(patients)

        else:
            requested_patients = max(
                1,
                round(
                    len(patients)
                    * self.label_pct
                    / 100.0
                ),
            )

            classes = sorted(
                set(patient_labels.values())
            )

            requested_patients = max(
                requested_patients,
                len(classes),
            )

            labeled_patients = set()

            for class_label in classes:
                class_patients = [
                    p
                    for p in patients
                    if patient_labels[p] == class_label
                ]

                selected = rng.choice(
                    class_patients,
                    size=1,
                    replace=False,
                )

                labeled_patients.update(
                    selected.tolist()
                )

            remaining = [
                p
                for p in patients
                if p not in labeled_patients
            ]

            additional_needed = (
                requested_patients
                - len(labeled_patients)
            )

            if additional_needed > 0 and remaining:
                additional_needed = min(
                    additional_needed,
                    len(remaining),
                )

                selected = rng.choice(
                    remaining,
                    size=additional_needed,
                    replace=False,
                )

                labeled_patients.update(
                    selected.tolist()
                )

        labeled_indices = []

        for patient_id in patients:
            if patient_id in labeled_patients:
                labeled_indices.extend(
                    patient_to_indices[patient_id]
                )

        labeled_indices = sorted(labeled_indices)

        labeled_index_set = set(labeled_indices)

        unlabeled_indices = [
            idx
            for idx in range(len(dataset.samples))
            if idx not in labeled_index_set
        ]

        n_labeled = len(labeled_indices)
        n_unlabeled = len(unlabeled_indices)

        if n_labeled == 0:
            raise RuntimeError(
                "Labeled split is empty."
            )

        if n_unlabeled == 0 and self.label_pct < 100:
            raise RuntimeError(
                "Unlabeled split is empty."
            )

        labeled_patient_count = len(labeled_patients)

        actual_patient_pct = (
            labeled_patient_count
            / len(patients)
            * 100.0
        )

        actual_image_pct = (
            n_labeled
            / len(dataset.samples)
            * 100.0
        )

        logger.info(
            "Label budget requested: %.2f%%",
            self.label_pct,
        )

        logger.info(
            "Labeled patients: %d / %d (%.2f%%)",
            labeled_patient_count,
            len(patients),
            actual_patient_pct,
        )

        logger.info(
            "Labeled images: %d / %d (%.2f%%)",
            n_labeled,
            len(dataset.samples),
            actual_image_pct,
        )

        logger.info(
            "Unlabeled images: %d / %d",
            n_unlabeled,
            len(dataset.samples),
        )

        return (
            labeled_indices,
            unlabeled_indices,
            n_labeled,
            n_unlabeled,
        )

    def _train_discriminator(
        self,
        labeled_batch,
        unlabeled_batch,
    ):
        self.D.train()
        self.opt_D.zero_grad()

        x_labeled = labeled_batch["image"].to(
            self.device
        )

        y_labeled = labeled_batch["label"].to(
            self.device
        )

        x_labeled = self.noise_layer(
            x_labeled
        )

        self._real_batch_for_g = (
            x_labeled.detach()
        )

        class_logits, fake_logits = self.D(
            x_labeled
        )

        loss_supervised = self.supervised_ce(
            class_logits,
            y_labeled,
        )

        loss_D = (
            self.config["losses"]["supervised_weight"]
            * loss_supervised
        )

        x_unlabeled = unlabeled_batch[
            "image"
        ].to(self.device)

        x_unlabeled = self.noise_layer(
            x_unlabeled
        )

        x_rot, y_rot = create_rotations(
            x_unlabeled
        )

        x_rot = x_rot.to(self.device)
        y_rot = y_rot.to(self.device)

        (
            _,
            rotation_logits,
            _,
            _,
        ) = self.D(
            x_rot,
            return_rotation=True,
        )

        loss_rotation = self.rotation_ce(
            rotation_logits,
            y_rot,
        )

        loss_D += (
            self.config["losses"]["rotation_weight"]
            * loss_rotation
        )

        # GAN adversarial loss.
        z = torch.randn(
            x_labeled.size(0),
            self.config["generator"]["z_dim"],
            device=self.device,
        )

        fake_images = self.G(z).detach()

        _, fake_logits_fake = self.D(
            fake_images
        )

        fake_targets = torch.zeros(
            x_labeled.size(0),
            device=self.device,
        )

        real_targets = torch.ones(
            x_labeled.size(0),
            device=self.device,
        )

        loss_fake = self.adversarial_loss(
            fake_logits_fake,
            fake_targets,
        )

        _, fake_logits_real = self.D(
            x_labeled
        )

        loss_real = self.adversarial_loss(
            fake_logits_real,
            real_targets,
        )

        loss_adversarial = (
            loss_fake + loss_real
        ) / 2.0

        loss_D += (
            self.config["losses"]["adversarial_weight"]
            * loss_adversarial
        )

        # VAT.
        def logit_fn(x):
            class_logits, _ = self.D(x)
            return class_logits

        loss_vat = self.vat(
            self.D,
            x_unlabeled,
            logit_fn,
        )

        loss_D += (
            self.config["losses"]["vat_weight"]
            * loss_vat
        )

        loss_D.backward()

        self.opt_D.step()

        return loss_D

    def _train_generator(self):
        self.G.train()
        self.opt_G.zero_grad()

        batch_size = (
            self.config["training"]["batch_size"]
        )

        z = torch.randn(
            batch_size,
            self.config["generator"]["z_dim"],
            device=self.device,
        )

        fake_images = self.G(z)

        with torch.no_grad():
            real_feat, _, _ = self.D(
                self._real_batch_for_g,
                return_features=True,
            )

        fake_feat, _, fake_logits_fake = self.D(
            fake_images,
            return_features=True,
        )

        loss_feature_match = self.feature_match(
            fake_feat,
            real_feat,
        )

        real_targets = torch.ones(
            z.size(0),
            device=self.device,
        )

        loss_adversarial = self.adversarial_loss(
            fake_logits_fake,
            real_targets,
        )

        loss_G = (
            self.config["losses"]["feature_match_weight"]
            * loss_feature_match
            +
            self.config["losses"]["adversarial_weight"]
            * loss_adversarial
        )

        loss_G.backward()

        self.opt_G.step()

        return loss_G

    def _save_best_checkpoint(
        self,
        epoch,
        val_metrics,
    ):
        checkpoint = {
            "epoch": epoch,
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
            "val_metrics": val_metrics,
            "label_pct": self.label_pct,
            "seed": self.seed,
        }

        torch.save(
            checkpoint,
            self.ckpt_dir / "best.pt",
        )

    def _save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
        }

        torch.save(
            checkpoint,
            self.ckpt_dir
            / f"ckpt_epoch_{epoch + 1}.pt",
        )

    def _save_samples(self, epoch):
        self.G.eval()

        with torch.no_grad():
            z = torch.randn(
                16,
                self.config["generator"]["z_dim"],
                device=self.device,
            )

            fake_images = self.G(z)

        torch.save(
            fake_images,
            self.sample_dir
            / f"samples_epoch_{epoch + 1}.pt",
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
    )

    parser.add_argument(
        "--exp",
        default="semigan_v2",
    )

    parser.add_argument(
        "--label_pct",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    trainer = SemiGANTrainer(
        args.config,
        args.exp,
        args.label_pct,
        args.seed,
    )

    dataset = BreakHisDataset(
        root_dir=trainer.config["data"]["breakhis_root"],
        transform=None,
    )

    results = trainer.train(dataset)

    with open(
        trainer.exp_dir / "result.txt",
        "w",
    ) as f:
        for i, epoch in enumerate(
            results["training_history"]["epoch"]
        ):
            loss_d = results[
                "training_history"
            ]["train_loss_D"][i]

            loss_g = results[
                "training_history"
            ]["train_loss_G"][i]

            val_f1 = results[
                "training_history"
            ]["val_f1"][i]

            f.write(
                f"{epoch}\t"
                f"{loss_d:.6f}\t"
                f"{loss_g:.6f}\t"
                f"{val_f1:.6f}\n"
            )


if __name__ == "__main__":
    main()
