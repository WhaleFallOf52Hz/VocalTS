import argparse
import contextlib
import io
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import sys
import traceback
import warnings
from multiprocessing import get_context
from time import time

import librosa
from tqdm import tqdm

from inference.msst_infer import MSSeparator
from utils.logger import get_logger
from webui.utils import load_configs


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}
BASE_PIPELINE_STEPS = [
	{
		"name": "step_1",
		"model_name": "big_beta5e.ckpt",
		"input_stem": None,
		"output_stems": {"vocals": "vocals", "other": "other"},
		"next_input_stem": "vocals",
	},
	{
		"name": "step_2",
		"model_name": "mel_band_roformer_karaoke_becruily.ckpt",
		"input_stem": "vocals",
		"output_stems": {"Vocals": "Vocals", "Instrumental": "Instrumental"},
		"next_input_stem": "Vocals",
	},
]
STEP_3_VARIANTS = {
	"mono": {
		"name": "step_3",
		"model_name": "dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt",
		"input_stem": "Vocals",
		"output_stems": {"noreverb": "noreverb", "reverb": "reverb"},
		"next_input_stem": None,
	},
	"stereo": {
		"name": "step_3",
		"model_name": "dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt",
		"input_stem": "Vocals",
		"output_stems": {"dry": "noreverb", "other": "reverb"},
		"next_input_stem": None,
	},
}
PROGRESS_TOKENS = ("Processing audio chunks", "Total progress")


def build_parser():
	parser = argparse.ArgumentParser(
		description="Three-step MSST batch audio processing pipeline",
		formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60),
	)
	input_group = parser.add_mutually_exclusive_group(required=True)
	input_group.add_argument("--input_file", type=str, help="Path to a single audio file.")
	input_group.add_argument("--input_dir", type=str, help="Path to a folder of audio files.")
	parser.add_argument("--output_dir", type=str, required=True, help="Directory to store pipeline results.")
	parser.add_argument("--output_format", choices=["wav", "flac", "mp3"], default="flac", help="Output format (default: %(default)s).")
	parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cuda", help="Device to use for inference (default: %(default)s).")
	parser.add_argument("--device_ids", nargs="+", type=int, default=[0], help="List of GPU ids, only used when device is cuda (default: %(default)s).")
	parser.add_argument(
		"--dereverb_mode",
		choices=["auto", "mono", "stereo"],
		default="auto",
		help="Dereverb model selection for step_3 (default: %(default)s).",
	)
	parser.add_argument("--use_tta", action="store_true", help="Enable test time augmentation.")
	parser.add_argument("-j", "--jobs", type=int, default=4, help="Number of worker processes for folder mode (default: %(default)s).")
	parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
	return parser


def is_audio_file(path):
	return os.path.isfile(path) and os.path.splitext(path)[1].lower() in AUDIO_EXTENSIONS


def discover_audio_files(input_dir):
	files = []
	for name in sorted(os.listdir(input_dir)):
		path = os.path.join(input_dir, name)
		if is_audio_file(path):
			files.append(path)
	return files


def ensure_valid_inputs(args, logger):
	if args.jobs < 1:
		raise ValueError("--jobs must be greater than or equal to 1.")

	if args.input_file:
		if not os.path.isfile(args.input_file):
			raise FileNotFoundError(f"Input file does not exist: {args.input_file}")
		if not is_audio_file(args.input_file):
			raise ValueError(f"Unsupported audio file extension: {args.input_file}")
	else:
		if not os.path.isdir(args.input_dir):
			raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
		audio_files = discover_audio_files(args.input_dir)
		if not audio_files:
			raise ValueError(f"No supported audio files found in: {args.input_dir}")
		logger.info(f"Discovered {len(audio_files)} audio files in {args.input_dir}")

	if args.input_file and args.jobs != 4:
		logger.warning("--jobs is only used in folder mode; the provided value will be ignored.")


def load_models_info(project_root):
	models_info_path = os.path.join(project_root, "data", "models_info.json")
	return load_configs(models_info_path)


def build_step_config(step, models_info, project_root):
	model_name = step["model_name"]
	if model_name not in models_info:
		raise KeyError(f"Model not found in models_info.json: {model_name}")

	model_info = models_info[model_name]
	model_path = model_info["target_position"]
	config_path = model_path.replace("pretrain", "configs") + ".yaml"
	model_type = model_info["model_type"]

	if not os.path.isabs(model_path):
		model_path = os.path.join(project_root, model_path.lstrip("./"))
	if not os.path.isabs(config_path):
		config_path = os.path.join(project_root, config_path.lstrip("./"))

	if not os.path.isfile(model_path):
		raise FileNotFoundError(f"Model file does not exist: {model_path}")
	if not os.path.isfile(config_path):
		raise FileNotFoundError(f"Config file does not exist: {config_path}")

	step_config = dict(step)
	step_config.update(
		{
			"model_type": model_type,
			"model_path": model_path,
			"config_path": config_path,
		}
	)
	return step_config


def load_step_configs(project_root):
	models_info = load_models_info(project_root)
	return {
		"base": [build_step_config(step, models_info, project_root) for step in BASE_PIPELINE_STEPS],
		"dereverb": {
			mode: build_step_config(step, models_info, project_root)
			for mode, step in STEP_3_VARIANTS.items()
		},
	}


def create_separator(step_config, args, logger):
	return MSSeparator(
		model_type=step_config["model_type"],
		config_path=step_config["config_path"],
		model_path=step_config["model_path"],
		device=args["device"],
		device_ids=args["device_ids"],
		output_format=args["output_format"],
		use_tta=args["use_tta"],
		store_dirs={},
		logger=logger,
		debug=args["debug"],
	)


def create_output_dirs(output_dir):
	for step in BASE_PIPELINE_STEPS:
		for stem in step["output_stems"].values():
			os.makedirs(os.path.join(output_dir, step["name"], stem), exist_ok=True)
	for stem in ("noreverb", "reverb"):
		os.makedirs(os.path.join(output_dir, "step_3", stem), exist_ok=True)


def get_channel_count(audio):
	if getattr(audio, "ndim", 1) == 1:
		return 1
	return int(audio.shape[0])


def resolve_dereverb_mode(requested_mode, channel_count):
	if requested_mode != "auto":
		return requested_mode
	return "mono" if channel_count <= 1 else "stereo"


def get_separator_for_step(step_config, separator_cache, args, logger):
	model_name = step_config["model_name"]
	if model_name not in separator_cache:
		with suppress_internal_progress():
			separator_cache[model_name] = create_separator(step_config, args, logger)
	return separator_cache[model_name]


class FilterProgressStream(io.TextIOBase):
	def __init__(self, target_stream):
		self.target_stream = target_stream

	def writable(self):
		return True

	def write(self, text):
		if not text:
			return 0
		if self._is_progress_text(text):
			return len(text)
		self.target_stream.write(text)
		return len(text)

	def flush(self):
		self.target_stream.flush()

	@staticmethod
	def _is_progress_text(text):
		cleaned = text.replace("\r", "").strip()
		if not cleaned:
			return True
		return any(token in cleaned for token in PROGRESS_TOKENS)


@contextlib.contextmanager
def suppress_internal_progress():
	stdout_filter = FilterProgressStream(sys.stdout)
	stderr_filter = FilterProgressStream(sys.stderr)
	with contextlib.redirect_stdout(stdout_filter), contextlib.redirect_stderr(stderr_filter):
		yield


@contextlib.contextmanager
def temporary_console_level(logger, level):
	console_handler = getattr(logger, "console_handler", None)
	if console_handler is None:
		yield
		return
	original_level = console_handler.level
	console_handler.setLevel(level)
	try:
		yield
	finally:
		console_handler.setLevel(original_level)


def run_pipeline_for_file(audio_path, output_dir, step_configs, separator_cache, args, logger):
	file_name = os.path.splitext(os.path.basename(audio_path))[0]
	current_audio = None
	current_sr = None
	active_steps = list(step_configs["base"])

	for index, step in enumerate(active_steps):
		separator = get_separator_for_step(step, separator_cache, args, logger)
		if index == 0:
			sample_rate = getattr(separator.config.audio, "sample_rate", 44100)
			current_audio, current_sr = librosa.load(audio_path, sr=sample_rate, mono=False)
			logger.debug(f"Loaded {audio_path} with sample rate {current_sr}")
			dereverb_mode = resolve_dereverb_mode(args["dereverb_mode"], get_channel_count(current_audio))
			active_steps.append(step_configs["dereverb"][dereverb_mode])
			logger.info(f"{file_name}: step_3 dereverb mode resolved to {dereverb_mode}")

		logger.info(f"{file_name}: running {step['name']} with {step['model_name']}")
		with suppress_internal_progress():
			results = separator.separate(current_audio)

		for result_stem, output_stem in step["output_stems"].items():
			if result_stem not in results:
				raise KeyError(f"Missing expected stem '{result_stem}' from {step['model_name']}")
			stem_dir = os.path.join(output_dir, step["name"], output_stem)
			separator.save_audio(results[result_stem], current_sr, f"{file_name}_{output_stem}", stem_dir)

		next_stem = step["next_input_stem"]
		if next_stem:
			current_audio = results[next_stem].T
		else:
			current_audio = None


def chunk_list(items, chunks):
	if chunks <= 1 or len(items) <= 1:
		return [items]
	chunks = min(chunks, len(items))
	result = [[] for _ in range(chunks)]
	for index, item in enumerate(items):
		result[index % chunks].append(item)
	return [chunk for chunk in result if chunk]


def worker_process(worker_id, audio_files, output_dir, step_configs, args, progress_queue=None):
	logger = get_logger(console_level=logging.DEBUG if args["debug"] else logging.INFO)
	if not args["debug"]:
		warnings.filterwarnings("ignore", category=UserWarning)

	separator_cache = {}
	success_files = []
	failed_files = []

	try:
		with temporary_console_level(logger, logging.DEBUG if args["debug"] else logging.ERROR):
			for audio_path in audio_files:
				try:
					run_pipeline_for_file(audio_path, output_dir, step_configs, separator_cache, args, logger)
					success_files.append(audio_path)
					if progress_queue is not None:
						progress_queue.put({"type": "file_done", "success": True, "file": audio_path, "worker_id": worker_id})
				except Exception as exc:
					error_message = str(exc)
					failed_files.append({"file": audio_path, "error": error_message})
					if progress_queue is not None:
						progress_queue.put(
							{
								"type": "file_done",
								"success": False,
								"file": audio_path,
								"error": error_message,
								"traceback": traceback.format_exc() if args["debug"] else "",
								"worker_id": worker_id,
							}
						)
	finally:
		for separator in separator_cache.values():
			try:
				separator.del_cache()
			except Exception:
				logger.warning("Failed to clean separator cache", exc_info=args["debug"])
		if progress_queue is not None:
			progress_queue.put({"type": "worker_done", "worker_id": worker_id})

	return {"worker_id": worker_id, "success_files": success_files, "failed_files": failed_files}


def process_single_file(args, project_root, logger):
	step_configs = load_step_configs(project_root)
	create_output_dirs(args.output_dir)
	runtime_args = {
		"device": args.device,
		"device_ids": args.device_ids,
		"output_format": args.output_format,
		"dereverb_mode": args.dereverb_mode,
		"use_tta": args.use_tta,
		"debug": args.debug,
	}
	separator_cache = {}
	try:
		with temporary_console_level(logger, logging.DEBUG if args.debug else logging.ERROR):
			with tqdm(total=1, desc="Pipeline progress", dynamic_ncols=True) as progress_bar:
				run_pipeline_for_file(args.input_file, args.output_dir, step_configs, separator_cache, runtime_args, logger)
				progress_bar.update(1)
	finally:
		for separator in separator_cache.values():
			separator.del_cache()
	return [args.input_file], []


def process_folder(args, project_root, logger):
	audio_files = discover_audio_files(args.input_dir)
	step_configs = load_step_configs(project_root)
	create_output_dirs(args.output_dir)

	if args.device == "cuda" and args.jobs > 1:
		logger.warning("Running multiple CUDA worker processes may exhaust GPU memory because each worker loads at least three models, and auto dereverb mode may load a fourth.")

	if args.jobs == 1:
		runtime_args = {
			"device": args.device,
			"device_ids": args.device_ids,
			"output_format": args.output_format,
			"dereverb_mode": args.dereverb_mode,
			"use_tta": args.use_tta,
			"debug": args.debug,
		}
		with tqdm(total=len(audio_files), desc="Pipeline progress", dynamic_ncols=True) as progress_bar:
			result = worker_process(0, audio_files, args.output_dir, step_configs, runtime_args)
			progress_bar.update(len(result["success_files"]) + len(result["failed_files"]))
		return result["success_files"], result["failed_files"]

	chunks = chunk_list(audio_files, args.jobs)
	runtime_args = {
		"device": args.device,
		"device_ids": args.device_ids,
		"output_format": args.output_format,
		"dereverb_mode": args.dereverb_mode,
		"use_tta": args.use_tta,
		"debug": args.debug,
	}
	ctx = get_context("spawn")
	success_files = []
	failed_files = []
	progress_queue = ctx.Queue()
	processes = []
	for index, chunk in enumerate(chunks):
		process = ctx.Process(
			target=worker_process,
			args=(index, chunk, args.output_dir, step_configs, runtime_args, progress_queue),
			name=f"msst_pipeline_worker_{index}",
		)
		process.start()
		processes.append(process)

	completed_files = 0
	completed_workers = 0
	with tqdm(total=len(audio_files), desc="Pipeline progress", dynamic_ncols=True) as progress_bar:
		while completed_workers < len(processes):
			event = progress_queue.get()
			if event["type"] == "worker_done":
				completed_workers += 1
				continue
			if event["type"] == "file_done":
				completed_files += 1
				progress_bar.update(1)
				if event["success"]:
					success_files.append(event["file"])
				else:
					failed_files.append({"file": event["file"], "error": event["error"]})
					logger.error(f"Failed to process {event['file']}: {event['error']}")
					if args.debug and event.get("traceback"):
						logger.error(event["traceback"])

	for process in processes:
		process.join()
		if process.exitcode not in (0, None):
			logger.error(f"Worker process exited abnormally: {process.name} (exit code {process.exitcode})")

	if completed_files != len(audio_files):
		logger.warning(f"Expected {len(audio_files)} completion events, but received {completed_files}.")
	return success_files, failed_files


def main():
	project_root = os.path.dirname(os.path.abspath(__file__))
	sys.path.append(project_root)

	parser = build_parser()
	args = parser.parse_args()

	logger = get_logger(console_level=logging.DEBUG if args.debug else logging.INFO)
	if not args.debug:
		warnings.filterwarnings("ignore", category=UserWarning)

	start_time = time()

	try:
		ensure_valid_inputs(args, logger)
		os.makedirs(args.output_dir, exist_ok=True)

		if args.input_file:
			success_files, failed_files = process_single_file(args, project_root, logger)
		else:
			success_files, failed_files = process_folder(args, project_root, logger)

		elapsed = time() - start_time
		logger.info(f"Pipeline completed in {elapsed:.2f}s. Success: {len(success_files)}, Failed: {len(failed_files)}")
		if success_files:
			logger.info("Successful files:")
			for path in success_files:
				logger.info(f"  - {path}")
		if failed_files:
			logger.warning("Failed files:")
			for item in failed_files:
				logger.warning(f"  - {item['file']}: {item['error']}")

		if failed_files and not success_files:
			return 1
		return 0
	except Exception as exc:
		logger.error(f"Pipeline failed: {exc}\n{traceback.format_exc()}")
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
