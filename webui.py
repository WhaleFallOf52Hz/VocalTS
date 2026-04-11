#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import gradio as gr


ROOT = Path(__file__).resolve().parent
LINKED_DATA = ROOT / "linked_data"
AVATARS_ROOT = LINKED_DATA / "avatars"
INFERENCE_ROOT = LINKED_DATA / "inference_data"

DDSP_DIR = ROOT / "third_party" / "DDSP-SVC"
SOVITS_DIR = ROOT / "third_party" / "SO-VITS-SVC"
MSST_DIR = ROOT / "third_party" / "MSST-WebUI"

NCM_EXT = ".ncm"
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}

SESSION_CACHE_LOCK = threading.Lock()
SESSION_PREPROCESS_CACHE: Dict[str, Optional[str]] = {
    "signature": None,
    "run_dir": None,
}


def run_cmd(
    cmd: List[str],
    cwd: Path,
    logs: List[str],
    cuda_visible_devices: str = "",
) -> None:
    env = os.environ.copy()
    cuda_visible_devices = (cuda_visible_devices or "").strip()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)

    logs.append(f"\n$ (cwd={cwd}) {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout:
        logs.append(proc.stdout.rstrip())
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def run_cmd_stream(
    cmd: List[str],
    cwd: Path,
    logs: List[str],
    cuda_visible_devices: str = "",
) -> Iterator[None]:
    env = os.environ.copy()
    cuda_visible_devices = (cuda_visible_devices or "").strip()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)

    logs.append(f"\n$ (cwd={cwd}) {' '.join(cmd)}")
    yield

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        if line:
            logs.append(line)
            yield

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(cmd)}")


def _extract_avatar_from_target(target: Path) -> Optional[str]:
    parts = target.resolve().parts
    for idx, part in enumerate(parts):
        if part == "avatars" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def detect_avatar_candidates() -> Dict[str, Optional[str]]:
    candidates: Dict[str, Optional[str]] = {
        "ddsp_data": None,
        "ddsp_exp": None,
        "sovits_dataset": None,
        "sovits_logs": None,
    }

    links = {
        "ddsp_data": DDSP_DIR / "data",
        "ddsp_exp": DDSP_DIR / "exp",
        "sovits_dataset": SOVITS_DIR / "dataset" / "44k",
        "sovits_logs": SOVITS_DIR / "logs" / "44k",
    }

    for key, link_path in links.items():
        if link_path.is_symlink():
            try:
                target = link_path.resolve()
                candidates[key] = _extract_avatar_from_target(target)
            except Exception:
                candidates[key] = None

    return candidates


def _detect_model_default_avatar(model_keys: List[str], model_name: str) -> Tuple[Optional[str], str]:
    candidates = detect_avatar_candidates()
    values = [candidates.get(k) for k in model_keys if candidates.get(k)]
    if not values:
        return None, f"{model_name}: 未检测到软链接 avatar，请手动选择。"

    unique = sorted(set(values))
    if len(unique) == 1:
        return unique[0], f"{model_name}: 自动识别 avatar = {unique[0]}"

    detail = {k: candidates.get(k) for k in model_keys}
    return unique[0], f"{model_name}: 软链接结果不一致，默认使用 {unique[0]}，请手动确认。详情: {detail}"


def detect_ddsp_default_avatar() -> Tuple[Optional[str], str]:
    return _detect_model_default_avatar(["ddsp_data", "ddsp_exp"], "DDSP-SVC")


def detect_sovits_default_avatar() -> Tuple[Optional[str], str]:
    return _detect_model_default_avatar(["sovits_dataset", "sovits_logs"], "SO-VITS-SVC")


def build_detect_message() -> str:
    ddsp_msg = detect_ddsp_default_avatar()[1]
    sovits_msg = detect_sovits_default_avatar()[1]
    return ddsp_msg + "\n\n" + sovits_msg


def list_avatars() -> List[str]:
    if not AVATARS_ROOT.exists():
        return []
    return sorted([p.name for p in AVATARS_ROOT.iterdir() if p.is_dir()])


def list_ddsp_ckpts() -> List[str]:
    exp_dir = DDSP_DIR / "exp"
    if not exp_dir.exists():
        return []
    paths = sorted(exp_dir.rglob("*.pt"))
    return [str(p.resolve()) for p in paths]


def list_sovits_models() -> List[str]:
    logs_dir = SOVITS_DIR / "logs" / "44k"
    if not logs_dir.exists():
        return []
    paths = sorted(
        p for p in logs_dir.rglob("*.pth")
        if p.is_file() and p.name.startswith("G")
    )
    return [str(p.resolve()) for p in paths]


def list_sovits_configs() -> List[str]:
    cands: List[Path] = []
    p1 = SOVITS_DIR / "configs" / "config.json"
    p2 = SOVITS_DIR / "logs" / "44k" / "config.json"
    if p1.exists():
        cands.append(p1)
    if p2.exists():
        cands.append(p2)
    return [str(p.resolve()) for p in cands]


def create_next_inference_dir() -> Path:
    INFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
    existing_numbers: List[int] = []
    for p in INFERENCE_ROOT.iterdir():
        if p.is_dir() and re.fullmatch(r"\d+", p.name):
            existing_numbers.append(int(p.name))

    next_idx = (max(existing_numbers) + 1) if existing_numbers else 1
    for idx in range(next_idx, 100000):
        candidate = INFERENCE_ROOT / f"{idx:04d}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("无法分配新的推理目录编号。")


def copy_uploaded_file(uploaded_path: str, run_dir: Path) -> Path:
    if not uploaded_path:
        raise ValueError("未上传文件。")
    src = Path(uploaded_path)
    if not src.exists():
        raise FileNotFoundError(f"上传文件不存在: {src}")
    dst = run_dir / src.name
    shutil.copy2(src, dst)
    return dst


def compute_upload_signature(uploaded_path: str) -> str:
    src = Path(uploaded_path)
    if not src.exists():
        raise FileNotFoundError(f"上传文件不存在: {src}")
    stat = src.stat()
    return f"{src.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"


def format_logs(logs: List[str]) -> str:
    return "\n".join(logs)


def find_first_audio_file(folder: Path) -> Path:
    candidates = [
        p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]
    if not candidates:
        raise RuntimeError(f"未在目录中找到音频文件: {folder}")
    return candidates[0]


def preprocess_to_cleaned(
    uploaded_path: str,
    cuda_devices: str,
    logs: List[str],
) -> Tuple[Path, str]:
    logs.append("[Step 1/3] 歌曲预处理：开始")
    run_dir = create_next_inference_dir()
    logs.append(f"本次推理目录: {run_dir}")

    copied = copy_uploaded_file(uploaded_path, run_dir)
    ext = copied.suffix.lower()

    if ext == NCM_EXT:
        logs.append("[Step 1/3] 检测到 ncm，执行 ncmdump 转换。")
        run_cmd(["ncmdump", str(copied), "-o", str(run_dir)], cwd=ROOT, logs=logs)
        audio_input = find_first_audio_file(run_dir)
    elif ext in AUDIO_EXTS:
        logs.append("[Step 1/3] 检测到音频文件，跳过 ncmdump。")
        audio_input = copied
    else:
        raise ValueError(f"不支持的输入格式: {copied.suffix}")

    cleaned_dir = run_dir / "cleaned"
    cmd = [
        sys.executable,
        "msst_pipeline.py",
        "--input_file",
        str(audio_input),
        "--output_dir",
        str(cleaned_dir),
        "--output_format",
        "flac",
        "--dereverb_mode",
        "auto",
    ]
    run_cmd(cmd, cwd=MSST_DIR, logs=logs, cuda_visible_devices=cuda_devices)
    logs.append("[Step 1/3] 歌曲预处理：完成")

    base_name = audio_input.stem
    return run_dir, base_name


def preprocess_to_cleaned_stream(
    uploaded_path: str,
    cuda_devices: str,
    logs: List[str],
    state: Dict[str, str],
) -> Iterator[None]:
    logs.append("[Step 1/3] 歌曲预处理：开始")
    run_dir = create_next_inference_dir()
    logs.append(f"本次推理目录: {run_dir}")
    yield

    copied = copy_uploaded_file(uploaded_path, run_dir)
    ext = copied.suffix.lower()

    if ext == NCM_EXT:
        logs.append("[Step 1/3] 检测到 ncm，执行 ncmdump 转换。")
        yield
        for _ in run_cmd_stream(["ncmdump", str(copied), "-o", str(run_dir)], cwd=ROOT, logs=logs):
            yield
        audio_input = find_first_audio_file(run_dir)
    elif ext in AUDIO_EXTS:
        logs.append("[Step 1/3] 检测到音频文件，跳过 ncmdump。")
        audio_input = copied
        yield
    else:
        raise ValueError(f"不支持的输入格式: {copied.suffix}")

    cleaned_dir = run_dir / "cleaned"
    cmd = [
        sys.executable,
        "msst_pipeline.py",
        "--input_file",
        str(audio_input),
        "--output_dir",
        str(cleaned_dir),
        "--output_format",
        "flac",
        "--dereverb_mode",
        "auto",
    ]
    for _ in run_cmd_stream(cmd, cwd=MSST_DIR, logs=logs, cuda_visible_devices=cuda_devices):
        yield

    logs.append("[Step 1/3] 歌曲预处理：完成")
    state["run_dir"] = str(run_dir)
    state["base_name"] = audio_input.stem
    yield


def get_or_create_preprocessed(
    uploaded_path: str,
    cuda_devices: str,
    logs: List[str],
) -> Tuple[Path, str]:
    signature = compute_upload_signature(uploaded_path)

    with SESSION_CACHE_LOCK:
        cached_signature = SESSION_PREPROCESS_CACHE.get("signature")
        cached_run_dir = SESSION_PREPROCESS_CACHE.get("run_dir")

    if cached_signature == signature and cached_run_dir:
        cached_dir = Path(cached_run_dir)
        dry_dir = cached_dir / "cleaned" / "step_3" / "noreverb"
        if cached_dir.exists() and dry_dir.exists():
            logs.append("[Step 1/3] 复用已预处理结果：未检测到曲目变化，跳过预处理。")
            first_dry, _ = get_processed_inputs(cached_dir)
            return cached_dir, first_dry.stem

    run_dir, base_name = preprocess_to_cleaned(uploaded_path, cuda_devices, logs)
    with SESSION_CACHE_LOCK:
        SESSION_PREPROCESS_CACHE["signature"] = signature
        SESSION_PREPROCESS_CACHE["run_dir"] = str(run_dir)
    return run_dir, base_name


def get_or_create_preprocessed_stream(
    uploaded_path: str,
    cuda_devices: str,
    logs: List[str],
    state: Dict[str, str],
) -> Iterator[None]:
    signature = compute_upload_signature(uploaded_path)

    with SESSION_CACHE_LOCK:
        cached_signature = SESSION_PREPROCESS_CACHE.get("signature")
        cached_run_dir = SESSION_PREPROCESS_CACHE.get("run_dir")

    if cached_signature == signature and cached_run_dir:
        cached_dir = Path(cached_run_dir)
        dry_dir = cached_dir / "cleaned" / "step_3" / "noreverb"
        if cached_dir.exists() and dry_dir.exists():
            logs.append("[Step 1/3] 复用已预处理结果：未检测到曲目变化，跳过预处理。")
            first_dry, _ = get_processed_inputs(cached_dir)
            state["run_dir"] = str(cached_dir)
            state["base_name"] = first_dry.stem
            yield
            return

    for _ in preprocess_to_cleaned_stream(uploaded_path, cuda_devices, logs, state):
        yield
    with SESSION_CACHE_LOCK:
        SESSION_PREPROCESS_CACHE["signature"] = signature
        SESSION_PREPROCESS_CACHE["run_dir"] = state["run_dir"]


def get_processed_inputs(run_dir: Path) -> Tuple[Path, Optional[Path]]:
    dry_dir = run_dir / "cleaned" / "step_3" / "noreverb"
    har_dir = run_dir / "cleaned" / "step_2" / "Instrumental"

    if not dry_dir.exists():
        raise RuntimeError(f"未找到 dry 输出目录: {dry_dir}")

    dry_files = [p for p in sorted(dry_dir.iterdir()) if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    if not dry_files:
        raise RuntimeError("未找到干声音频输出(step_3/noreverb)。")

    har_file = None
    if har_dir.exists():
        har_files = [p for p in sorted(har_dir.iterdir()) if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
        if har_files:
            har_file = har_files[0]

    return dry_files[0], har_file


def ensure_avatar_selected(avatar: str) -> str:
    avatar = (avatar or "").strip()
    if not avatar:
        raise ValueError("avatar 不能为空。")
    avatar_dir = AVATARS_ROOT / avatar
    if not avatar_dir.exists():
        raise ValueError(f"avatar 不存在: {avatar}")
    return avatar


def switch_avatar(avatar: str, model_name: str, logs: List[str]) -> None:
    run_cmd(
        [sys.executable, "func_scripts/1_switch_avatar.py", avatar, model_name],
        cwd=ROOT,
        logs=logs,
    )


def on_ddsp_avatar_change(avatar: str) -> Tuple[dict, str]:
    logs: List[str] = []
    try:
        avatar = ensure_avatar_selected(avatar)
        switch_avatar(avatar, "DDSP-SVC", logs)
        ddsp_ckpts = list_ddsp_ckpts()
        message = build_detect_message() + "\n\n[DDSP] avatar 切换完成，已刷新权重列表。"
        return (
            gr.update(choices=ddsp_ckpts, value=ddsp_ckpts[0] if ddsp_ckpts else None),
            message,
        )
    except Exception as exc:
        message = build_detect_message() + f"\n\n[DDSP] avatar 切换失败: {exc}"
        return gr.update(), message


def on_sovits_avatar_change(avatar: str) -> Tuple[dict, dict, str]:
    logs: List[str] = []
    try:
        avatar = ensure_avatar_selected(avatar)
        switch_avatar(avatar, "SO-VITS-SVC", logs)
        sovits_models = list_sovits_models()
        sovits_configs = list_sovits_configs()
        message = build_detect_message() + "\n\n[SO-VITS] avatar 切换完成，已刷新权重列表。"
        return (
            gr.update(choices=sovits_models, value=sovits_models[0] if sovits_models else None),
            gr.update(choices=sovits_configs, value=sovits_configs[0] if sovits_configs else None),
            message,
        )
    except Exception as exc:
        message = build_detect_message() + f"\n\n[SO-VITS] avatar 切换失败: {exc}"
        return gr.update(), gr.update(), message


def run_ddsp_infer(
    uploaded_file: str,
    avatar: str,
    ddsp_ckpt: str,
    key_shift: int,
    spk_id: int,
    pitch_extractor: str,
    process_harmonic: bool,
    cuda_devices: str,
) -> Iterator[Tuple[str, str]]:
    logs: List[str] = []
    try:
        status = "准备中..."
        yield status, format_logs(logs)
        key_shift = int(key_shift)
        spk_id = int(spk_id)
        avatar = ensure_avatar_selected(avatar)
        if not ddsp_ckpt:
            raise ValueError("请先选择 DDSP 模型权重(.pt)。")

        progress(0.05, desc="Step 1/3 歌曲预处理")
        state: Dict[str, str] = {}
        for _ in get_or_create_preprocessed_stream(uploaded_file, cuda_devices, logs, state):
            yield "[Step 1/3] 歌曲预处理中...", format_logs(logs)
        run_dir = Path(state["run_dir"])

        dry_file, har_file = get_processed_inputs(run_dir)

        progress(0.20, desc="切换 DDSP avatar")
        for _ in run_cmd_stream(
            [sys.executable, "func_scripts/1_switch_avatar.py", avatar, "DDSP-SVC"],
            cwd=ROOT,
            logs=logs,
        ):
            yield "切换 DDSP avatar...", format_logs(logs)

        out_root = run_dir / "transfered" / "DDSP-SVC" / avatar
        out_root.mkdir(parents=True, exist_ok=True)

        dry_out = out_root / f"{dry_file.stem}_DryVocal.flac"
        cmd_dry = [
            sys.executable,
            "main_reflow.py",
            "-i",
            str(dry_file),
            "-m",
            ddsp_ckpt,
            "-o",
            str(dry_out),
            "-k",
            str(key_shift),
            "-id",
            str(spk_id),
            "-method",
            "auto",
            "-ts",
            "0.0",
            "-pe",
            pitch_extractor,
        ]
        logs.append("[Step 2/3] 干声转换：DDSP-SVC")
        progress(0.45, desc="Step 2/3 干声转换")
        for _ in run_cmd_stream(cmd_dry, cwd=DDSP_DIR, logs=logs, cuda_visible_devices=cuda_devices):
            yield "[Step 2/3] 干声转换中...", format_logs(logs)

        if process_harmonic and har_file is not None:
            har_out = out_root / f"{har_file.stem}_Harmonic.flac"
            cmd_har = [
                sys.executable,
                "main_reflow.py",
                "-i",
                str(har_file),
                "-m",
                ddsp_ckpt,
                "-o",
                str(har_out),
                "-k",
                str(key_shift),
                "-id",
                str(spk_id),
                "-method",
                "auto",
                "-ts",
                "0.0",
                "-pe",
                pitch_extractor,
            ]
            logs.append("[Step 3/3] 和声转换：DDSP-SVC")
            progress(0.75, desc="Step 3/3 和声转换")
            for _ in run_cmd_stream(cmd_har, cwd=DDSP_DIR, logs=logs, cuda_visible_devices=cuda_devices):
                yield "[Step 3/3] 和声转换中...", format_logs(logs)
        elif process_harmonic:
            logs.append("未找到和声音频(step_2/Instrumental)，已跳过和声推理。")
        else:
            logs.append("[Step 3/3] 和声转换：未启用，已跳过。")

        logs.append(f"\nDDSP 推理完成，输出目录: {out_root}")
        progress(1.0, desc="完成")
        yield "✅ DDSP 推理完成", format_logs(logs)
        return
    except Exception as exc:
        logs.append(f"\n[ERROR] {exc}")
        logs.append(traceback.format_exc())
        yield "❌ DDSP 推理失败", format_logs(logs)
        return


def run_sovits_infer(
    uploaded_file: str,
    avatar: str,
    model_path: str,
    config_path: str,
    key_shift: int,
    speaker_name: str,
    enhance: bool,
    process_harmonic: bool,
    cuda_devices: str,
) -> Iterator[Tuple[str, str]]:
    logs: List[str] = []
    try:
        status = "准备中..."
        yield status, format_logs(logs)
        key_shift = int(key_shift)
        avatar = ensure_avatar_selected(avatar)
        if not model_path:
            raise ValueError("请先选择 SO-VITS 模型权重(.pth)。")
        if not config_path:
            raise ValueError("请先选择 SO-VITS 配置文件(config.json)。")

        progress(0.05, desc="Step 1/3 歌曲预处理")
        state: Dict[str, str] = {}
        for _ in get_or_create_preprocessed_stream(uploaded_file, cuda_devices, logs, state):
            yield "[Step 1/3] 歌曲预处理中...", format_logs(logs)
        run_dir = Path(state["run_dir"])

        dry_file, har_file = get_processed_inputs(run_dir)

        progress(0.20, desc="切换 SO-VITS avatar")
        for _ in run_cmd_stream(
            [sys.executable, "func_scripts/1_switch_avatar.py", avatar, "SO-VITS-SVC"],
            cwd=ROOT,
            logs=logs,
        ):
            yield "切换 SO-VITS avatar...", format_logs(logs)

        out_root = run_dir / "transfered" / "SO-VITS-SVC" / avatar
        out_root.mkdir(parents=True, exist_ok=True)

        dry_out = out_root / f"{dry_file.stem}_DryVocal.flac"
        cmd_dry = [
            sys.executable,
            "inference_main.py",
            "-m",
            model_path,
            "-c",
            config_path,
            "-n",
            str(dry_file),
            "-t",
            str(key_shift),
            "-o",
            str(dry_out),
            "-f0p",
            "rmvpe",
            "-wf",
            "flac",
        ]
        speaker_name = (speaker_name or "").strip()
        if speaker_name:
            cmd_dry.extend(["-s", speaker_name])
        if enhance:
            cmd_dry.append("--enhance")
        logs.append("[Step 2/3] 干声转换：SO-VITS-SVC")
        progress(0.45, desc="Step 2/3 干声转换")
        for _ in run_cmd_stream(cmd_dry, cwd=SOVITS_DIR, logs=logs, cuda_visible_devices=cuda_devices):
            yield "[Step 2/3] 干声转换中...", format_logs(logs)

        if process_harmonic and har_file is not None:
            har_out = out_root / f"{har_file.stem}_Harmonic.flac"
            cmd_har = [
                sys.executable,
                "inference_main.py",
                "-m",
                model_path,
                "-c",
                config_path,
                "-n",
                str(har_file),
                "-t",
                str(key_shift),
                "-o",
                str(har_out),
                "-f0p",
                "rmvpe",
                "-wf",
                "flac",
            ]
            if speaker_name:
                cmd_har.extend(["-s", speaker_name])
            if enhance:
                cmd_har.append("--enhance")
            logs.append("[Step 3/3] 和声转换：SO-VITS-SVC")
            progress(0.75, desc="Step 3/3 和声转换")
            for _ in run_cmd_stream(cmd_har, cwd=SOVITS_DIR, logs=logs, cuda_visible_devices=cuda_devices):
                yield "[Step 3/3] 和声转换中...", format_logs(logs)
        elif process_harmonic:
            logs.append("未找到和声音频(step_2/Instrumental)，已跳过和声推理。")
        else:
            logs.append("[Step 3/3] 和声转换：未启用，已跳过。")

        logs.append(f"\nSO-VITS 推理完成，输出目录: {out_root}")
        progress(1.0, desc="完成")
        yield "✅ SO-VITS 推理完成", format_logs(logs)
        return
    except Exception as exc:
        logs.append(f"\n[ERROR] {exc}")
        logs.append(traceback.format_exc())
        yield "❌ SO-VITS 推理失败", format_logs(logs)
        return


def refresh_choices() -> Tuple[dict, dict, dict, dict, dict, str]:
    avatars = list_avatars()
    ddsp_detected, ddsp_msg = detect_ddsp_default_avatar()
    sovits_detected, sovits_msg = detect_sovits_default_avatar()
    ddsp_default_avatar = ddsp_detected if ddsp_detected in avatars else (avatars[0] if avatars else None)
    sovits_default_avatar = sovits_detected if sovits_detected in avatars else (avatars[0] if avatars else None)

    ddsp_ckpts = list_ddsp_ckpts()
    sovits_models = list_sovits_models()
    sovits_configs = list_sovits_configs()
    merged_msg = build_detect_message()

    return (
        gr.update(choices=avatars, value=ddsp_default_avatar),
        gr.update(choices=avatars, value=sovits_default_avatar),
        gr.update(choices=ddsp_ckpts, value=ddsp_ckpts[0] if ddsp_ckpts else None),
        gr.update(choices=sovits_models, value=sovits_models[0] if sovits_models else None),
        gr.update(choices=sovits_configs, value=sovits_configs[0] if sovits_configs else None),
        merged_msg,
    )


def build_ui() -> gr.Blocks:
    avatars = list_avatars()
    ddsp_detected, ddsp_msg = detect_ddsp_default_avatar()
    sovits_detected, sovits_msg = detect_sovits_default_avatar()
    ddsp_default_avatar = ddsp_detected if ddsp_detected in avatars else (avatars[0] if avatars else None)
    sovits_default_avatar = sovits_detected if sovits_detected in avatars else (avatars[0] if avatars else None)
    detect_msg = build_detect_message()

    ddsp_ckpts = list_ddsp_ckpts()
    sovits_models = list_sovits_models()
    sovits_configs = list_sovits_configs()

    with gr.Blocks(title="VocalTS Inference WebUI") as demo:
        gr.Markdown("# VocalTS 推理 WebUI")
        gr.Markdown(f"项目根目录: {ROOT}")
        detect_info = gr.Markdown(detect_msg)

        refresh_btn = gr.Button("刷新 avatar/模型列表")

        with gr.Tabs():
            with gr.Tab("DDSP-SVC"):
                ddsp_file = gr.File(label="上传 ncm/mp3/flac/wav", type="filepath")
                ddsp_avatar = gr.Dropdown(label="avatar", choices=avatars, value=ddsp_default_avatar, allow_custom_value=True)
                ddsp_ckpt = gr.Dropdown(label="DDSP 权重(.pt)", choices=ddsp_ckpts, value=ddsp_ckpts[0] if ddsp_ckpts else None, allow_custom_value=True)
                ddsp_status = gr.Markdown("状态：就绪")
                with gr.Row():
                    ddsp_cuda = gr.Textbox(label="CUDA_VISIBLE_DEVICES", value="0", placeholder="例如: 0 或 0,1；留空则不设置")
                    ddsp_key = gr.Number(label="变调 key", value=0, precision=0)
                    ddsp_spk_id = gr.Number(label="spk_id", value=1, precision=0)
                ddsp_pitch = gr.Dropdown(label="pitch extractor", choices=["rmvpe", "harvest", "dio", "parselmouth", "crepe", "fcpe"], value="rmvpe")
                ddsp_harmonic = gr.Checkbox(label="同时处理和声(step_2/Instrumental)", value=True)
                ddsp_run = gr.Button("开始 DDSP 推理")
                ddsp_logs = gr.Textbox(label="日志", lines=26)

                ddsp_run.click(
                    fn=run_ddsp_infer,
                    inputs=[ddsp_file, ddsp_avatar, ddsp_ckpt, ddsp_key, ddsp_spk_id, ddsp_pitch, ddsp_harmonic, ddsp_cuda],
                    outputs=[ddsp_status, ddsp_logs],
                )

                ddsp_avatar.change(
                    fn=on_ddsp_avatar_change,
                    inputs=[ddsp_avatar],
                    outputs=[ddsp_ckpt, detect_info],
                )

            with gr.Tab("SO-VITS-SVC"):
                sovits_file = gr.File(label="上传 ncm/mp3/flac/wav", type="filepath")
                sovits_avatar = gr.Dropdown(label="avatar", choices=avatars, value=sovits_default_avatar, allow_custom_value=True)
                sovits_model = gr.Dropdown(label="SO-VITS 权重(.pth)", choices=sovits_models, value=sovits_models[0] if sovits_models else None, allow_custom_value=True)
                sovits_config = gr.Dropdown(label="config.json", choices=sovits_configs, value=sovits_configs[0] if sovits_configs else None, allow_custom_value=True)
                sovits_status = gr.Markdown("状态：就绪")
                with gr.Row():
                    sovits_cuda = gr.Textbox(label="CUDA_VISIBLE_DEVICES", value="0", placeholder="例如: 0 或 0,1；留空则不设置")
                    sovits_key = gr.Number(label="变调 key", value=0, precision=0)
                sovits_speaker = gr.Textbox(label="speaker 名称(可选)", value="", placeholder="为空则自动使用模型默认说话人")
                with gr.Row():
                    sovits_enhance = gr.Checkbox(label="启用 enhance", value=True)
                    sovits_harmonic = gr.Checkbox(label="同时处理和声(step_2/Instrumental)", value=True)
                sovits_run = gr.Button("开始 SO-VITS 推理")
                sovits_logs = gr.Textbox(label="日志", lines=26)

                sovits_run.click(
                    fn=run_sovits_infer,
                    inputs=[sovits_file, sovits_avatar, sovits_model, sovits_config, sovits_key, sovits_speaker, sovits_enhance, sovits_harmonic, sovits_cuda],
                    outputs=[sovits_status, sovits_logs],
                )

                sovits_avatar.change(
                    fn=on_sovits_avatar_change,
                    inputs=[sovits_avatar],
                    outputs=[sovits_model, sovits_config, detect_info],
                )

        refresh_btn.click(
            fn=refresh_choices,
            inputs=[],
            outputs=[ddsp_avatar, sovits_avatar, ddsp_ckpt, sovits_model, sovits_config, detect_info],
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue().launch(server_name="0.0.0.0", server_port=7861)


if __name__ == "__main__":
    main()
