import argparse, logging
logging.basicConfig(level="INFO",
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")

def main():
    p = argparse.ArgumentParser("CTR-GCN AVA Pipeline")
    p.add_argument("--mode", required=True,
        choices=["setup","download","annotate","train","test_run",
                 "evaluate","full_pipeline","export"])
    p.add_argument("--config",      default="configs/base_config.yaml")
    p.add_argument("--phase",       choices=["1","2","both"], default="both")
    p.add_argument("--resume_run_id", default=None)
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--test_mode",   action="store_true")
    p.add_argument("--classes",     nargs="+", default=None)
    args = p.parse_args()

    from utils.config_loader import load_config
    overrides = {"device": args.device}
    if args.test_mode:   overrides["test_mode"] = True
    if args.classes:     overrides["target_classes"] = args.classes
    if args.resume_run_id: overrides["resume_run_id"] = args.resume_run_id

    if args.mode == "setup":
        import subprocess
        subprocess.run(["bash", "scripts/setup_env.sh"], check=True)

    elif args.mode == "download":
        cfg = load_config(args.config, overrides)
        from scripts.download_ava_kinetics import download_ava_kinetics
        from scripts.download_ava import download_ava
        if cfg.dataset in ("ava_kinetics", "both"):
            download_ava_kinetics(cfg)
        if cfg.dataset in ("ava", "both"):
            download_ava(cfg)

    elif args.mode == "annotate":
        cfg = load_config(args.config, overrides)
        from annotation.batch_annotate import BatchAnnotator
        BatchAnnotator(cfg).run()

    elif args.mode == "train":
        from training.pipeline import run_phase, run_full_pipeline
        if args.phase == "both":
            run_full_pipeline()
        elif args.phase == "1":
            run_phase("configs/phase1_ava_kinetics.yaml", "phase1")
        else:
            run_phase("configs/phase2_ava.yaml", "phase2")

    elif args.mode == "test_run":
        from scripts.test_run import run_smoke_test
        run_smoke_test(args)

    elif args.mode == "evaluate":
        cfg = load_config(args.config, overrides)
        from evaluation.ava_evaluator import AVAEvaluator
        AVAEvaluator(cfg).evaluate()

    elif args.mode == "export":
        from scripts.export_onnx import export
        cfg = load_config(args.config, overrides)
        export(cfg)

    elif args.mode == "full_pipeline":
        from training.pipeline import run_full_pipeline
        run_full_pipeline()

if __name__ == "__main__":
    main()
