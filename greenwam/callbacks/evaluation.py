# SPDX-License-Identifier: OpenMDW-1.1
"""Deterministic GreenWAM denoising evaluation, separate from generated-rollout QA."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def _single(value):
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        return _single(value[0]) if value else None
    return value


def metric_family(data_batch, output_batch):
    """Keep joint labels stable and never aggregate EEF errors into their units.

    GreenWAM enforces one sample per packed microbatch. Contract IDs survive collation
    and are required for model routing; optional representation strings are not
    needed to distinguish the new EEF bank after packing.
    """
    from greenwam.data.state_eef import STATE_EEF_BANK_OFFSET

    contract = _single(data_batch.get("action_contract_id"))
    family = _single(output_batch.get("_action_family"))
    if family is None:
        family = contract
    if contract is not None and int(contract) >= STATE_EEF_BANK_OFFSET:
        # Keep each EEF head separate even when an admitted dataset spans families.
        return f"{family}/eef_contract_{int(contract)}"
    native_eef = _single(data_batch.get("greenwam_action_representation")) == "native_eef"
    has_action = _single(data_batch.get("action")) is not None
    if contract is None and _single(data_batch.get("domain_id")) == 8 and (has_action or native_eef):
        # Domain 8 without a GreenWAM contract is the separately qualified DROID replay.
        return f"{family if family is not None else 'domain_8'}/native_eef"
    return "visual" if family is None else str(family)


SUPPORTED_TOPOLOGY = (
    "GreenWAM production CP=TP=CFGP=1 and WORLD_SIZE=dp_shard*dp_replicate; "
    "RankPartitionedDataLoader shards the mixture by world rank"
)


def metric_dataset(data_batch):
    """Explicit dataset label of the admitted atom when the batch carries ``greenwam_dataset``.

    Family labels stay ``mode/family``. Do not guess from layout_id or atom_id.
    Visual and action readers copy this field from the admitted atom.
    """
    value = _single(data_batch.get("greenwam_dataset"))
    if value is not None and str(value) != "":
        return str(value)
    return None


def _parallel_int(dims, name):
    if not hasattr(dims, name):
        raise ValueError(name)
    value = getattr(dims, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(name)
    return value


def _tensor_parallel_degree(model):
    parallelism = getattr(getattr(model, "config", None), "parallelism", None)
    if parallelism is None:
        return None
    if isinstance(parallelism, dict):
        if "tensor_parallel_degree" not in parallelism:
            return None
        value = parallelism["tensor_parallel_degree"]
    elif hasattr(parallelism, "tensor_parallel_degree"):
        value = parallelism.tensor_parallel_degree
    else:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("tensor_parallel_degree")
    return value


def validation_gather_plan(model):
    """Accept only the production GreenWAM data-parallel topology, or a single process.

    Distributed global metrics require ``model.parallel_dims`` with resolved
    ``cp=cfgp=1`` and ``world_size == dp_shard * dp_replicate``. A present
    tensor-parallel degree must be 1. Unknown or disagreeing topology is
    refused; ranks are never treated as independent by default.
    """
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    world = torch.distributed.get_world_size() if distributed else 1
    rank = torch.distributed.get_rank() if distributed else 0
    topology = dict(world_size=world, rank=rank, supported=SUPPORTED_TOPOLOGY)
    if not distributed:
        topology["mode"] = "single_process"
        return dict(ok=True, reason=None, topology=topology)
    dims = getattr(model, "parallel_dims", None)
    if dims is None:
        topology["mode"] = "distributed"
        return dict(ok=False, reason="missing model.parallel_dims", topology=topology)
    try:
        cp, cfgp = _parallel_int(dims, "cp"), _parallel_int(dims, "cfgp")
        dp_shard, dp_replicate = _parallel_int(dims, "dp_shard"), _parallel_int(dims, "dp_replicate")
        tp = _tensor_parallel_degree(model)
    except (TypeError, ValueError):
        topology["mode"] = "distributed"
        return dict(ok=False, reason="model.parallel_dims is missing resolved CP/CFG/DP integers", topology=topology)
    topology.update(mode="distributed", cp=cp, cfgp=cfgp, dp_shard=dp_shard, dp_replicate=dp_replicate, tp=tp)
    if cp != 1 or cfgp != 1 or tp not in (None, 1):
        return dict(ok=False, reason="GreenWAM global metrics require CP=TP=CFGP=1", topology=topology)
    if dp_shard < 1 or dp_replicate < 1 or world != dp_shard * dp_replicate:
        return dict(ok=False, reason="WORLD_SIZE must equal dp_shard*dp_replicate", topology=topology)
    return dict(ok=True, reason=None, topology=topology)


def _fold_additive(dest, rows):
    for key, row in (rows or {}).items():
        bucket = dest[key]
        for field, value in row.items():
            bucket[field] += float(value)


def merge_metric_payloads(payloads):
    """Sum additive counts/sums across ranks. Means are computed later from totals."""
    metric = MetricAccumulator()
    contributing = []
    unattributed = 0
    for payload in payloads or ():
        if not payload or not payload.get("contribute"):
            continue
        contributing.append(payload.get("rank"))
        unattributed += int(payload.get("unattributed_samples") or 0)
        _fold_additive(metric.rows, payload.get("rows"))
        _fold_additive(metric.dataset_rows, payload.get("dataset_rows"))
    metric.unattributed_samples = unattributed
    return metric, contributing


def dataset_attribution(metric):
    attributed = sum(int(row.get("samples", 0)) for row in metric.dataset_rows.values())
    unattributed = int(metric.unattributed_samples)
    return dict(
        accepted_batch_field="greenwam_dataset",
        attributed_samples=attributed,
        unattributed_samples=unattributed,
        complete=unattributed == 0 and attributed > 0,
        note="Dataset attribution uses the explicit dataset label of each admitted atom; this report alone does not qualify the complete evaluation cohort",
    )


def action_error_sums(predictions, targets, masks, conditions, *, sigmas=None):
    result = dict(action_velocity_abs_sum=0.0, action_velocity_sq_sum=0.0, action_velocity_count=0)
    if sigmas is not None:
        result.update(action_denoised_abs_sum=0.0, action_denoised_sq_sum=0.0, action_denoised_count=0)
    for index, (pred, target, valid, condition) in enumerate(zip(predictions, targets, masks, conditions, strict=True)):
        mask = valid.to(device=pred.device, dtype=torch.bool) & ~condition.to(device=pred.device, dtype=torch.bool)
        error = pred.float() - target.float()
        difference = error[mask.expand_as(pred)]
        if not torch.isfinite(difference).all():
            raise ValueError("Evaluation requires finite active action predictions")
        result["action_velocity_abs_sum"] += float(difference.abs().sum())
        result["action_velocity_sq_sum"] += float(difference.square().sum())
        result["action_velocity_count"] += difference.numel()
        if sigmas is not None:
            # x_t = sigma * epsilon + (1-sigma) * q. For velocity epsilon-q,
            # q_hat=x_t-sigma*v_hat, so q_hat-q = -sigma*(v_hat-v).
            difference = (error * sigmas[index].to(pred.device))[mask.expand_as(pred)]
            if not torch.isfinite(difference).all():
                raise ValueError("Expected finite denoised state error")
            result["action_denoised_abs_sum"] += float(difference.abs().sum())
            result["action_denoised_sq_sum"] += float(difference.square().sum())
            result["action_denoised_count"] += difference.numel()
    return result


class ValidationMixin:
    @torch.no_grad()
    def validation_step(self, data_batch, iteration=0):
        identity = [str(_single(data_batch.get(k))) for k in ("greenwam_mode", "atom_id", "mixture_next_global_step")]
        seed = int(hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:8], 16)
        py_state, numpy_state, training = random.getstate(), np.random.get_state(), self.training
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        try:
            self.eval()
            self._validation_active = True
            with torch.random.fork_rng(devices=devices):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                # Iteration also seeds the framework's explicit noise generator.
                return self.training_step(data_batch, seed)
        finally:
            self._validation_active = False
            random.setstate(py_state)
            np.random.set_state(numpy_state)
            self.train(training)

    def _compute_losses(self, *args, **kwargs):
        total, values = super()._compute_losses(*args, **kwargs)
        if getattr(self, "_validation_active", False):
            action = kwargs["data_batch_packed"].action
            if action is not None:
                values.update(
                    action_error_sums(
                        kwargs["out_net"]["preds_action"],
                        kwargs["gen_data_noised"].vt_target_action,
                        action.action_valid_mask,
                        action.condition_mask,
                        sigmas=kwargs["gen_data_noised"].sigmas_action,
                    )
                )
        return total, values


def build_validation_model(config, **kwargs):
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    class ValidationModel(ValidationMixin, OmniMoTModel):
        pass

    return ValidationModel(config, **kwargs)


def reset_validation_loader(loader):
    """Packing loaders otherwise retain their child iterators between passes."""
    for inner in loader.dataloader_list:
        child = getattr(inner, "dataloader", inner)
        if child.num_workers != 0:
            raise ValueError("Fixed GreenWAM validation currently requires workers=0")
        child.dataset.set_start_local_draw(0)
        child.generator = torch.Generator().manual_seed(4242)
    loader.global_id = 0
    loader._child_iterators_initialized = False
    loader._initialize_child_iterators_once()


class MetricAccumulator:
    def __init__(self):
        self.rows = defaultdict(lambda: defaultdict(float))
        self.dataset_rows = defaultdict(lambda: defaultdict(float))
        self.unattributed_samples = 0

    def add(self, mode, family, values, dataset=None):
        # Validate the complete sample before mutating sums/counts. Missing
        # measurements must not appear as either a zero loss or a counted row.
        measurements = {}
        for modality, active in (("vision", mode != "inverse"), ("action", mode in {"inverse", "wam", "policy"})):
            if active:
                key = "flow_matching_loss_" + modality
                if key not in values:
                    raise ValueError(f"Missing required validation loss: {key}")
                value = float(values[key])
                if not math.isfinite(value):
                    raise ValueError("Expected finite validation loss")
                measurements[modality + "_flow_sum"] = value
        for key in (
            "action_velocity_abs_sum",
            "action_velocity_sq_sum",
            "action_velocity_count",
            "action_denoised_abs_sum",
            "action_denoised_sq_sum",
            "action_denoised_count",
        ):
            if key in values:
                value = float(values[key])
                if not math.isfinite(value):
                    raise ValueError("Expected finite validation action errors")
                measurements[key] = value
        self._accumulate(self.rows[f"{mode}/{family}"], measurements)
        label = None if dataset is None else str(dataset)
        if label:
            self._accumulate(self.dataset_rows[f"{mode}/{family}/{label}"], measurements)
        else:
            self.unattributed_samples += 1

    @staticmethod
    def _accumulate(row, measurements):
        row["samples"] += 1
        for key, value in measurements.items():
            row[key] += value

    def additive_payload(self, *, rank, contribute, reason, topology):
        return dict(
            rank=rank,
            contribute=bool(contribute),
            reason=reason,
            topology=topology,
            unattributed_samples=int(self.unattributed_samples),
            rows={key: dict(row) for key, row in self.rows.items()},
            dataset_rows={key: dict(row) for key, row in self.dataset_rows.items()},
        )

    def report(self):
        return self._report(self.rows)

    def report_by_dataset(self):
        return self._report(self.dataset_rows)

    @staticmethod
    def _report(rows):
        result = {}
        for key, row in sorted(rows.items()):
            out = {"samples": int(row["samples"])}
            for modality in ("vision", "action"):
                if modality + "_flow_sum" in row:
                    out[modality + "_flow_loss"] = row[modality + "_flow_sum"] / row["samples"]
            count = row.get("action_velocity_count", 0)
            if count:
                out.update(
                    action_velocity_mae=row["action_velocity_abs_sum"] / count,
                    action_velocity_rmse=math.sqrt(row["action_velocity_sq_sum"] / count),
                    valid_action_values=int(count),
                    action_error_space="model_space",
                )
            count = row.get("action_denoised_count", 0)
            if count:
                out.update(
                    action_denoised_mae=row["action_denoised_abs_sum"] / count,
                    action_denoised_rmse=math.sqrt(row["action_denoised_sq_sum"] / count),
                )
            result[key] = out
        return result


def build_evaluation_callback(output_dir):
    from cosmos_framework.utils.callback import Callback

    class GreenWAMEvaluation(Callback):
        def on_validation_start(self, model, dataloader_val, iteration=0):
            self.metric, self.started = MetricAccumulator(), time.monotonic()
            self.random_state = (
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state(),
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            )
            reset_validation_loader(dataloader_val)

        def on_validation_step_end(self, model, data_batch, output_batch, loss, iteration=0):
            mode = str(_single(data_batch["greenwam_mode"]))
            self.metric.add(
                mode, metric_family(data_batch, output_batch), output_batch, dataset=metric_dataset(data_batch)
            )

        def on_validation_end(self, model, iteration=0):
            try:
                distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
                rank = torch.distributed.get_rank() if distributed else 0
                world = torch.distributed.get_world_size() if distributed else 1
                plan = validation_gather_plan(model)
                payload = self.metric.additive_payload(
                    rank=rank, contribute=plan["ok"], reason=plan["reason"], topology=plan["topology"]
                )
                if not plan["ok"]:
                    payload["rows"], payload["dataset_rows"] = {}, {}
                    payload["unattributed_samples"] = 0
                gathered = [payload]
                if distributed:
                    gathered = [None] * world
                    torch.distributed.all_gather_object(gathered, payload)
                baseline = "stage initialization checkpoint" if iteration == 0 else "trained checkpoint"
                scope = (
                    "fixed-noise flow and one-step denoised action errors on valid channels; "
                    "not generated rollout/task accuracy"
                )
                local = dict(
                    schema_version=1,
                    iteration=iteration,
                    rank=rank,
                    world_size=world,
                    aggregation="rank_local",
                    baseline=baseline,
                    scope=scope,
                    elapsed_s=time.monotonic() - self.started,
                    metrics=self.metric.report(),
                    metrics_by_dataset=self.metric.report_by_dataset(),
                    dataset_attribution=dataset_attribution(self.metric),
                )
                output = Path(output_dir)
                output.mkdir(parents=True, exist_ok=True)
                with (output / f"step_{iteration:08d}.rank_{rank}.jsonl").open("a") as stream:
                    stream.write(json.dumps(local, sort_keys=True) + "\n")
                if rank == 0:
                    if plan["ok"] and all(item and item.get("contribute") for item in gathered):
                        merged, contributing = merge_metric_payloads(gathered)
                        global_report = dict(
                            schema_version=2,
                            iteration=iteration,
                            rank=0,
                            world_size=world,
                            aggregation="global_count_weighted",
                            contributing_ranks=contributing,
                            topology=plan["topology"],
                            baseline=baseline,
                            scope=scope,
                            elapsed_s=time.monotonic() - self.started,
                            metrics=merged.report(),
                            metrics_by_dataset=merged.report_by_dataset(),
                            dataset_attribution=dataset_attribution(merged),
                            eval08_complete=False,
                        )
                    else:
                        reason = plan["reason"] or "unsupported or unknown GreenWAM topology"
                        global_report = dict(
                            schema_version=2,
                            iteration=iteration,
                            rank=0,
                            world_size=world,
                            aggregation="global_refused",
                            aggregation_error=reason,
                            contributing_ranks=[],
                            topology=plan["topology"],
                            baseline=baseline,
                            scope=scope,
                            elapsed_s=time.monotonic() - self.started,
                            metrics={},
                            metrics_by_dataset={},
                            dataset_attribution=dataset_attribution(MetricAccumulator()),
                            eval08_complete=False,
                        )
                    with (output / f"step_{iteration:08d}.global.jsonl").open("a") as stream:
                        stream.write(json.dumps(global_report, sort_keys=True) + "\n")
            finally:
                random.setstate(self.random_state[0])
                np.random.set_state(self.random_state[1])
                torch.set_rng_state(self.random_state[2])
                if self.random_state[3] is not None:
                    torch.cuda.set_rng_state(self.random_state[3])

    return GreenWAMEvaluation()
