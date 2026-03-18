import argparse, logging
logging.basicConfig(level="INFO",
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")

log = logging.getLogger("main")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("CTR-GCN AVA Pipeline")

    # ── Execution mode ────────────────────────────────────────────────────
    p.add_argument(
        "--mode", required=True,
        choices=["download", "annotate", "train", "evaluate",
                 "full_pipeline", "test_run", "status", "export",
                 "infer", "audit", "registry"],
        help=(
            "Pipeline stage to run.\n"
            "  download      : download videos for selected classes only\n"
            "  annotate      : extract skeletons (skips existing .npz files)\n"
            "  train         : fine-tune model (phases 1, 2, or both)\n"
            "  evaluate      : evaluate best checkpoint on val set\n"
            "  full_pipeline : download -> annotate -> train -> evaluate\n"
            "  test_run      : smoke test with synthetic data\n"
            "  status        : print per-class download/annotation counts\n"
            "  export        : export best checkpoint to ONNX + TorchScript\n"
            "  infer         : run model on a new video\n"
            "  audit         : audit dataset before training\n"
            "  registry      : view experiment history\n"
        )
    )

    # ── Class selection (the core new feature) ────────────────────────────
    class_group = p.add_mutually_exclusive_group()
    class_group.add_argument(
        "--classes", nargs="+", metavar="NAME",
        help=(
            "One or more class names (partial match OK).\n"
            "Examples:\n"
            "  --classes eat drink walk\n"
            "  --classes 'punch/slap' 'hug (a person)'\n"
            "  --classes run   # matches 'run/jog'"
        )
    )
    class_group.add_argument(
        "--class_ids", nargs="+", type=int, metavar="ID",
        help=(
            "One or more AVA class IDs (integers).\n"
            "Examples:\n"
            "  --class_ids 17 20 14\n"
            "  --class_ids 49 50 52 63 74"
        )
    )
    class_group.add_argument(
        "--class_category", metavar="CATEGORY",
        choices=["movement", "object", "interaction", "pose"],
        help="Select all classes in a category (e.g. --class_category interaction)"
    )
    class_group.add_argument(
        "--all_classes", action="store_true",
        help="Use all 15 target classes defined in class_config.yaml"
    )

    # ── Stage skip flags ──────────────────────────────────────────────────
    p.add_argument(
        "--start_from",
        choices=["download", "annotate", "train", "evaluate"],
        default=None,
        help=(
            "Skip all stages before this one.\n"
            "  --start_from annotate  -> skip download, run annotate+train+evaluate\n"
            "  --start_from train     -> skip download+annotate, run train+evaluate\n"
            "  --start_from evaluate  -> only run evaluation on best checkpoint\n"
            "Requires data from skipped stages to already exist on disk."
        )
    )
    p.add_argument(
        "--skip_download",   action="store_true",
        help="Skip download stage even if running full_pipeline"
    )
    p.add_argument(
        "--skip_annotate",   action="store_true",
        help="Skip annotation stage even if running full_pipeline"
    )
    p.add_argument(
        "--skip_train",      action="store_true",
        help="Skip training stage (useful to re-run evaluation only)"
    )

    # ── Training phase control ────────────────────────────────────────────
    p.add_argument(
        "--phase", choices=["1", "2", "both"], default="both",
        help="Which training phase to run (default: both)"
    )

    # ── Resume / checkpoint ───────────────────────────────────────────────
    p.add_argument(
        "--resume_run_id", default=None, metavar="RUN_ID",
        help=(
            "Resume a specific training run.\n"
            "Format: {timestamp}_{phase}_{experiment_name}\n"
            "Example: --resume_run_id 20240315_143200_phase1_ctrgcn_ava"
        )
    )

    # ── Inference arguments ────────────────────────────────────────────────
    p.add_argument("--input_video", default=None, metavar="PATH",
                   help="Path to video file for inference (--mode infer)")
    p.add_argument("--checkpoint",  default=None, metavar="PATH",
                   help="Path to checkpoint for inference/export")
    p.add_argument("--output_json", default=None, metavar="PATH",
                   help="Output JSON path for inference results")
    p.add_argument("--use_tta",     action="store_true",
                   help="Use Test-Time Augmentation for inference/evaluation")
    p.add_argument("--threshold",   type=float, default=0.3,
                   help="Action prediction threshold (default: 0.3)")

    # ── System ────────────────────────────────────────────────────────────
    p.add_argument("--config",     default="configs/base_config.yaml")
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--test_mode",  action="store_true")
    p.add_argument("--dry_run",    action="store_true",
                   help="Print what would be done without doing it")
    p.add_argument("--workers",    type=int, default=None,
                   help="Override num_workers in config")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── 1. Resolve which classes to use ──────────────────────────────────
    from utils.class_registry import ClassRegistry
    from utils.class_resolver import resolve_classes
    registry = ClassRegistry()                    # loads class_config.yaml
    selected = resolve_classes(args, registry)    # returns ClassRegistry subset
    log.info(f"Selected {selected.num_classes} classes: "
             f"{selected.class_names}")

    # ── 2. Build config with selected classes ─────────────────────────────
    from utils.config_loader import load_config, apply_class_selection
    config = load_config(args.config)
    config = apply_class_selection(config, selected)
    if args.device:       config.device       = args.device
    if args.workers:      config.num_workers   = args.workers
    if args.test_mode:    config.test_mode     = True
    if args.resume_run_id: config.resume_run_id = args.resume_run_id

    # ── 3. Resolve which stages to run ───────────────────────────────────
    from utils.stage_resolver import resolve_stages
    stages = resolve_stages(args)
    log.info(f"Stages to run: {stages}")

    if args.dry_run:
        _print_dry_run_plan(stages, selected, config)
        return

    # ── 4. Execute stages in order ───────────────────────────────────────
    run_stages(stages, config, selected, args)


def _print_dry_run_plan(stages, selected, config):
    """Print what would be done without doing it."""
    print("\n" + "=" * 60)
    print("DRY RUN — no actions will be performed")
    print("=" * 60)
    print(f"  Classes  : {selected.class_names}")
    print(f"  Class IDs: {selected.class_ids}")
    print(f"  Stages   : {stages}")
    print(f"  Dataset  : {config.dataset}")
    print(f"  Device   : {config.device}")
    print(f"  Test mode: {config.test_mode}")
    print("=" * 60 + "\n")


def run_stages(stages, config, selected, args):
    """
    Execute resolved stages in order.
    Each stage receives the config (with class selection applied)
    and the selected ClassRegistry.
    """
    from pathlib import Path
    from utils.stage_resolver import validate_stage_prerequisites
    from utils.class_resolver import save_class_selection

    # Validate prerequisites for skipped stages
    warnings = validate_stage_prerequisites(stages, config, selected)
    for w in warnings:
        log.warning(w)

    # Save class selection for reproducibility
    run_dir = Path(config.checkpoint_dir) / "runs" / config.experiment_name
    save_class_selection(selected, str(run_dir))

    for stage in stages:
        log.info(f"\n{'='*60}\nRunning stage: {stage.upper()}\n{'='*60}")

        if stage == "download":
            _run_download(config, selected)

        elif stage == "annotate":
            _run_annotate(config, selected)

        elif stage == "train":
            _run_train(config, selected, args)

        elif stage == "evaluate":
            _run_evaluate(config, selected, args)

        elif stage == "status":
            from utils.pipeline_status import print_pipeline_status
            print_pipeline_status(config, selected)

        elif stage == "test_run":
            from scripts.test_run import run_smoke_test
            run_smoke_test(args)

        elif stage == "export":
            from scripts.export_onnx import export
            export(config, checkpoint_path=getattr(args, 'checkpoint', None))

        elif stage == "infer":
            _run_infer(config, selected, args)

        elif stage == "audit":
            _run_audit(config, selected)

        elif stage == "registry":
            _run_registry()

        log.info(f"Stage '{stage}' complete.\n")


def _run_download(config, selected):
    from scripts.download_ava_kinetics import download_ava_kinetics
    from scripts.download_ava import download_ava
    if config.dataset in ("ava_kinetics", "both"):
        download_ava_kinetics(config, selected)
    if config.dataset in ("ava", "both"):
        download_ava(config, selected)


def _run_annotate(config, selected):
    from annotation.batch_annotate import BatchAnnotator
    BatchAnnotator(config, selected).run()


def _run_train(config, selected, args):
    from training.pipeline import run_phase, run_full_pipeline
    phase = getattr(args, "phase", "both")
    if phase == "both":
        run_full_pipeline(class_registry=selected)
    elif phase == "1":
        run_phase("configs/phase1_ava_kinetics.yaml", "phase1",
                  class_registry=selected)
    else:
        run_phase("configs/phase2_ava.yaml", "phase2",
                  class_registry=selected)


def _run_evaluate(config, selected, args=None):
    from evaluation.ava_evaluator import AVAEvaluator
    use_tta = getattr(args, 'use_tta', False) if args else False
    AVAEvaluator(config, selected).evaluate(use_tta=use_tta)


def _run_infer(config, selected, args):
    """Run inference on a video."""
    from inference.video_inference import VideoInference
    if not args.input_video:
        log.error("--input_video is required for --mode infer")
        return
    vi = VideoInference(
        checkpoint_path=args.checkpoint or _find_best_checkpoint(config),
        class_registry=selected,
        config=config,
        use_tta=args.use_tta,
        threshold=args.threshold,
    )
    output = args.output_json or "outputs/inference_results.json"
    vi.run(args.input_video, output_path=output)


def _run_audit(config, selected):
    """Run dataset audit."""
    from utils.dataset_auditor import DatasetAuditor, AuditError
    from pathlib import Path
    auditor = DatasetAuditor()
    annotation_dir = str(Path(config.data_dir) / "annotations")
    try:
        auditor.audit(annotation_dir, selected, min_quality=0.30)
    except AuditError as e:
        log.error(f"Audit FAILED:\n{e}")


def _run_registry():
    """Print experiment registry summary."""
    from utils.experiment_registry import ExperimentRegistry
    registry = ExperimentRegistry()
    registry.print_summary(last_n=20)


def _find_best_checkpoint(config):
    """Auto-find best.pth in checkpoint dir."""
    from pathlib import Path
    runs_dir = Path(config.checkpoint_dir) / "runs"
    best_paths = list(runs_dir.rglob("best.pth"))
    if best_paths:
        return str(best_paths[-1])
    log.error("No best.pth found. Use --checkpoint to specify.")
    return None


if __name__ == "__main__":
    main()
