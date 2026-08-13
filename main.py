#!/usr/bin/env python3
"""Unified CLI for densNet dental tissue classification.

All features are exposed here; logic lives in the ``densnet`` package.

Examples:
  python main.py --help
  python main.py --train
  python main.py --predict all
  python main.py --predict dentin
  python main.py --evaluate
  python main.py --detect --image path/to/xray.png
"""

from __future__ import annotations

import argparse

from densnet.constants import CLASS_NAMES


def _cmd_train() -> None:
    from densnet.train_runner import run_training

    run_training()


def _cmd_predict(target: str) -> None:
    from densnet.predict_batch import run_folder_predictions

    if target == "all":
        for name in CLASS_NAMES:
            run_folder_predictions(name)
        return
    run_folder_predictions(target)


def _cmd_evaluate() -> None:
    from densnet.metrics import run_evaluation

    run_evaluation()


def _cmd_detect(image: str | None) -> None:
    from densnet.detect import run_detect_cli

    run_detect_cli(image)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "densNet — DenseNet121 dental tissue classifier "
            f"({', '.join(CLASS_NAMES)})."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py --help
  python main.py --train
  python main.py --predict all
  python main.py --predict enamel
  python main.py --evaluate
  python main.py --detect
  python main.py --detect --image radiograph.png
""",
    )

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--train",
        action="store_true",
        help="Train DenseNet121 on segmented_dental_adiography/",
    )
    action.add_argument(
        "--predict",
        metavar="TARGET",
        choices=[*CLASS_NAMES, "all"],
        help="Batch-predict image-testing/{TARGET}_test (or all classes)",
    )
    action.add_argument(
        "--evaluate",
        action="store_true",
        help="Compute Precision/Recall/Accuracy/F1 from prediction CSVs",
    )
    action.add_argument(
        "--detect",
        action="store_true",
        help="Sliding-window detection on a full radiograph",
    )
    parser.add_argument(
        "--image",
        metavar="PATH",
        help="Image path for --detect (prompts if omitted)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI flags and run the selected feature."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.image and not args.detect:
        parser.error("--image can only be used with --detect")

    if args.train:
        _cmd_train()
    elif args.predict is not None:
        _cmd_predict(args.predict)
    elif args.evaluate:
        _cmd_evaluate()
    elif args.detect:
        _cmd_detect(args.image)


if __name__ == "__main__":
    main()
