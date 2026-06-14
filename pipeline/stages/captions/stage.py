from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.llm_defaults import DEFAULT_CAPTIONS_MODEL, DEFAULT_LLM_BASE_URL
from pipeline.parallel_defaults import DEFAULT_STAGE_WORKERS
from pipeline.manifest import DEFAULT_MANIFEST_NAME, manifest_path
from pipeline.stages.base import PipelineStage


class CaptionsStage(PipelineStage):
    name = "captions"
    description = "VLM captions + robot labels (then prune + video2smpl in same run)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("captions stage")
        group.add_argument(
            "--manifest",
            type=str,
            default=None,
            help=f"Manifest path (default: <root_dir>/{DEFAULT_MANIFEST_NAME}, updated in place).",
        )
        group.add_argument(
            "--output-manifest",
            type=str,
            default=None,
            help="Optional separate output; default is same file as --manifest.",
        )
        group.add_argument("--model", type=str, default=DEFAULT_CAPTIONS_MODEL)
        group.add_argument(
            "--vision-detail",
            type=str,
            default="high",
            choices=("low", "high", "auto", "original"),
        )
        group.add_argument("--num-frames", type=int, default=16)
        group.add_argument("--max-side", type=int, default=768)
        group.add_argument(
            "--caption-lang",
            choices=("en", "zh", "bilingual"),
            default="en",
        )
        group.add_argument("--sleep", type=float, default=0.5)
        group.add_argument("--workers", type=int, default=DEFAULT_STAGE_WORKERS)
        group.add_argument("--force-recaption", action="store_true")
        group.add_argument("--dry-run", action="store_true")
        group.add_argument("--timeout", type=float, default=600.0)
        group.add_argument("--max-retries", type=int, default=2)
        group.add_argument(
            "--caption-parse-retries",
            type=int,
            default=2,
            help="Re-call VLM when JSON validation fails (total tries = 1 + value).",
        )
        group.add_argument(
            "--caption-temperature",
            type=float,
            default=0.0,
            help="Caption API temperature (0 = deterministic).",
        )
        group.add_argument(
            "--json-mode",
            action="store_true",
            help="Enable response_format=json_object (off by default for vision VLMs).",
        )
        group.add_argument(
            "--no-json-mode",
            action="store_true",
            help="Force-disable JSON mode.",
        )
        group.add_argument("--base-url", type=str, default=DEFAULT_LLM_BASE_URL)
        group.add_argument("--http-referer", type=str, default="")
        group.add_argument("--x-title", type=str, default="video2smpl-manifest-captions")
        group.add_argument("--heartbeat-sec", type=float, default=15.0)
        group.add_argument(
            "--no-drop-invalid-skill-category",
            action="store_true",
            help="Keep samples when skill_category validation fails (default: drop row + sample dir).",
        )

    def validate_args(self, args: argparse.Namespace) -> None:
        from pipeline.manifest import load_manifest_list, manifest_path, resolve_video_rel

        root = Path(args.root_dir).resolve()
        mpath = manifest_path(root, getattr(args, "manifest_name", None))
        if not mpath.exists():
            raise ValueError(
                f"Manifest not found: {mpath}. Run the select stage first (--from-stage select)."
            )
        rows = load_manifest_list(mpath)
        if not rows:
            raise ValueError(f"Manifest is empty: {mpath}")
        missing = [str(r.get("sample_id", "?")) for r in rows if not resolve_video_rel(r)]
        if missing:
            raise ValueError(
                f"{len(missing)} sample(s) lack video_path. Run the select stage before captions."
            )

    def run(self, args: argparse.Namespace) -> None:
        root = Path(args.root_dir).resolve()
        manifest_name = getattr(args, "manifest_name", DEFAULT_MANIFEST_NAME)

        if args.manifest:
            manifest_p = Path(args.manifest).expanduser()
            if not manifest_p.is_absolute():
                manifest_p = (root / manifest_p).resolve()
        else:
            manifest_p = manifest_path(root, manifest_name)

        out_manifest = args.output_manifest
        if out_manifest:
            out_p = Path(out_manifest).expanduser()
            if not out_p.is_absolute():
                out_p = (root / out_p).resolve()
        else:
            out_p = manifest_p

        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from generate_sequence_captions import main as captions_main

        argv = [
            "generate_sequence_captions",
            "--manifest",
            str(manifest_p),
            "--pipeline-root",
            str(root),
            "--output-manifest",
            str(out_p),
            "--model",
            args.model,
            "--vision-detail",
            args.vision_detail,
            "--num-frames",
            str(args.num_frames),
            "--max-side",
            str(args.max_side),
            "--caption-lang",
            args.caption_lang,
            "--sleep",
            str(args.sleep),
            "--workers",
            str(args.workers),
            "--timeout",
            str(args.timeout),
            "--max-retries",
            str(args.max_retries),
            "--caption-parse-retries",
            str(args.caption_parse_retries),
            "--caption-temperature",
            str(args.caption_temperature),
            "--base-url",
            args.base_url,
            "--http-referer",
            args.http_referer,
            "--x-title",
            args.x_title,
            "--heartbeat-sec",
            str(args.heartbeat_sec),
        ]
        if args.dry_run:
            argv.append("--dry-run")
        if args.force_recaption:
            argv.append("--force-recaption")
        if getattr(args, "json_mode", False):
            argv.append("--json-mode")
        if getattr(args, "no_json_mode", False):
            argv.append("--no-json-mode")
        if getattr(args, "no_drop_invalid_skill_category", False):
            argv.append("--no-drop-invalid-skill-category")

        old_argv = sys.argv
        try:
            sys.argv = argv
            exit_code = captions_main()
        finally:
            sys.argv = old_argv

        if exit_code != 0:
            raise SystemExit(exit_code)
