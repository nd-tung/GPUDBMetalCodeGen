#!/usr/bin/env python3
"""Local Python web UI for the GPUDB Metal TPC-H runner."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
SQL_DIR = ROOT / "sql"
DATA_DIR = ROOT / "data"
BIN_PATH = ROOT / "build" / "bin" / "GPUDBCodegen"
RUNS_DIR = ROOT / "tmp" / "web_runs"
PREPARE_DATA_SCRIPT = ROOT / "scripts" / "run_all_queries.sh"

QUERY_IDS = {f"q{i}" for i in range(1, 23)}
SCALE_FACTORS = ("sf1", "sf10", "sf20", "sf50", "sf100")
TABLES = ("region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem")
BUILD_LOCK = threading.Lock()
DATA_LOCK = threading.Lock()
SUITE_LOCK = threading.Lock()
SYSTEM_INFO_CACHE: dict[str, Any] | None = None

TIMING_FIELDS = [
    "scale_factor",
    "query",
    "route",
    "analyze_ms",
    "plan_ms",
    "codegen_ms",
    "metal_compile_ms",
    "pso_ms",
    "compile_overhead_ms",
    "load_source",
    "load_bytes",
    "load_mibps",
    "ingest_ms",
    "data_load_ms",
    "io_ms",
    "preprocess_ms",
    "buffer_setup_ms",
    "gpu_compute_ms",
    "cpu_compute_ms",
    "query_compute_ms",
    "query_execution_ms",
    "end_to_end_ms",
    "execute_wall_ms",
    "execute_residual_ms",
    "hook_cpu_ms",
    "hook_gpu_ms",
    "result_collect_ms",
    "host_post_ms",
    "validation_ms",
    "gpu_trials_n",
    "gpu_p10_ms",
    "gpu_p50_ms",
    "gpu_p90_ms",
    "gpu_mad_ms",
    "hot_execution_ms",
]


class AppConfig:
    host = "127.0.0.1"
    port = 8000
    auto_build = True
    default_timeout = 300


CONFIG = AppConfig()


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.details = details or {}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def query_title(query_id: str, sql: str) -> str:
    first_line = sql.splitlines()[0].strip() if sql.splitlines() else ""
    match = re.match(r"--\s*TPC-H\s+Query\s+\d+:\s*(.+)", first_line, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return f"TPC-H {query_id.upper()}"


def load_queries() -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for n in range(1, 23):
        query_id = f"q{n}"
        path = SQL_DIR / f"{query_id}.sql"
        sql = read_text(path) if path.exists() else f"-- Missing {path.name}"
        title = query_title(query_id, sql)
        queries.append(
            {
                "id": query_id,
                "number": n,
                "label": f"Q{n} - {title}",
                "title": title,
                "sql": sql.strip() + "\n",
            }
        )
    return queries


def scale_dir(scale: str) -> Path:
    return DATA_DIR / f"SF-{scale[2:]}"


def validate_scale(scale: Any) -> str:
    normalized = str(scale or "sf1").lower()
    if normalized not in SCALE_FACTORS:
        raise ApiError("Invalid scale factor.")
    return normalized


def data_status(scale: str) -> dict[str, Any]:
    scale = validate_scale(scale)
    path = scale_dir(scale)
    has_data = lambda table, ext: (path / f"{table}.{ext}").is_file() and (path / f"{table}.{ext}").stat().st_size > 0
    colbin_present = [table for table in TABLES if has_data(table, "colbin")]
    tbl_present = [table for table in TABLES if has_data(table, "tbl")]
    missing_colbin = [table for table in TABLES if table not in colbin_present]
    missing_tbl = [table for table in TABLES if table not in tbl_present]
    ready = len(colbin_present) == len(TABLES)
    tbl_complete = len(tbl_present) == len(TABLES)
    return {
        "id": scale,
        "label": scale.upper(),
        "path": str(path.relative_to(ROOT)),
        "directory_exists": path.exists(),
        "expected_tables": list(TABLES),
        "colbin_tables": colbin_present,
        "tbl_tables": tbl_present,
        "missing_colbin": missing_colbin,
        "missing_tbl": missing_tbl,
        "colbin_count": len(colbin_present),
        "tbl_count": len(tbl_present),
        "expected_count": len(TABLES),
        "ready": ready,
        "tbl_complete": tbl_complete,
        "can_generate": True,
        "message": (
            f"{scale.upper()} ready: {len(TABLES)}/{len(TABLES)} .colbin tables"
            if ready
            else f"{scale.upper()} missing data: {len(colbin_present)}/{len(TABLES)} .colbin tables"
        ),
    }


def load_scales() -> list[dict[str, Any]]:
    return [data_status(scale) for scale in SCALE_FACTORS]


def command_text(args: list[str], timeout: int = 3) -> str:
    try:
        proc = subprocess.run(
            args,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def sysctl_value(name: str) -> str:
    return command_text(["sysctl", "-n", name])


def format_ram(bytes_text: str) -> tuple[int, str]:
    try:
        value = int(bytes_text)
    except (TypeError, ValueError):
        return 0, "unknown RAM"
    gib = value / (1024**3)
    if abs(gib - round(gib)) < 0.05:
        return value, f"{round(gib):.0f} GB"
    return value, f"{gib:.1f} GB"


def find_gpu_name(value: Any) -> str:
    preferred = ("sppci_model", "spdisplays_chipset-model", "chipset_model", "_name")
    if isinstance(value, dict):
        for key in preferred:
            found = value.get(key)
            if isinstance(found, str) and found.strip():
                return found.strip()
        for child in value.values():
            found = find_gpu_name(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_gpu_name(child)
            if found:
                return found
    return ""


def load_system_info() -> dict[str, Any]:
    global SYSTEM_INFO_CACHE
    if SYSTEM_INFO_CACHE is not None:
        return SYSTEM_INFO_CACHE

    chip = sysctl_value("machdep.cpu.brand_string") or sysctl_value("hw.model") or "unknown chip"
    physical_cores = sysctl_value("hw.physicalcpu") or sysctl_value("hw.ncpu")
    logical_cores = sysctl_value("hw.logicalcpu")
    ram_bytes, ram_label = format_ram(sysctl_value("hw.memsize"))

    gpu = ""
    profiler = command_text(["system_profiler", "SPDisplaysDataType", "-json"], timeout=8)
    if profiler:
        try:
            gpu = find_gpu_name(json.loads(profiler))
        except json.JSONDecodeError:
            gpu = ""
    if not gpu:
        gpu = chip

    cpu_label = f"{physical_cores} CPU cores" if physical_cores else "unknown CPU cores"
    if logical_cores and logical_cores != physical_cores:
        cpu_label += f" / {logical_cores} threads"

    SYSTEM_INFO_CACHE = {
        "chip": chip,
        "cpu_cores": int(physical_cores) if physical_cores.isdigit() else 0,
        "cpu_threads": int(logical_cores) if logical_cores.isdigit() else 0,
        "cpu_label": cpu_label,
        "gpu": gpu,
        "ram_bytes": ram_bytes,
        "ram": ram_label,
        "summary": f"{chip} | {cpu_label} | GPU {gpu} | RAM {ram_label}",
    }
    return SYSTEM_INFO_CACHE


def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def tail(text: str, limit: int = 60000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def ensure_binary() -> dict[str, Any]:
    if BIN_PATH.exists():
        return {"built": False, "log": ""}
    if not CONFIG.auto_build:
        raise ApiError(
            "Backend binary is missing.",
            details={"binary": str(BIN_PATH), "hint": "Run make from the repo root."},
        )

    with BUILD_LOCK:
        if BIN_PATH.exists():
            return {"built": False, "log": ""}
        try:
            proc = subprocess.run(
                ["make"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired as exc:
            raise ApiError(
                "Timed out while building GPUDBCodegen.",
                status=500,
                details={
                    "stdout": tail(exc.stdout or ""),
                    "stderr": tail(exc.stderr or ""),
                },
            ) from exc

        build_log = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
        if proc.returncode != 0 or not BIN_PATH.exists():
            raise ApiError(
                "Failed to build GPUDBCodegen.",
                status=500,
                details={"returncode": proc.returncode, "build_log": tail(build_log)},
            )
        return {"built": True, "log": tail(build_log, 20000)}


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    mode = str(payload.get("mode") or "tpch").lower()
    if mode not in {"tpch", "custom"}:
        raise ApiError("Invalid run mode.")

    query_id = str(payload.get("query_id") or "q1").lower()
    if query_id not in QUERY_IDS:
        raise ApiError("Invalid TPC-H query.")

    scale = validate_scale(payload.get("scale"))

    custom_sql = str(payload.get("custom_sql") or "").strip()
    if mode == "custom" and not custom_sql:
        raise ApiError("Custom SQL is empty.")

    return {
        "mode": mode,
        "query_id": query_id,
        "scale": scale,
        "custom_sql": custom_sql,
        "warmup": clamp_int(payload.get("warmup"), 0, 0, 25),
        "repeat": clamp_int(payload.get("repeat"), 1, 1, 50),
        "timeout": clamp_int(payload.get("timeout"), CONFIG.default_timeout, 10, 1200),
    }


def validate_suite_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "scale": validate_scale(payload.get("scale")),
        "warmup": clamp_int(payload.get("warmup"), 0, 0, 25),
        "repeat": clamp_int(payload.get("repeat"), 1, 1, 50),
        "timeout": clamp_int(payload.get("timeout"), CONFIG.default_timeout, 10, 1200),
    }


def ensure_data_ready(scale: str) -> dict[str, Any]:
    status = data_status(scale)
    if status["ready"]:
        return status
    raise ApiError(
        "TPC-H data is missing for the selected scale factor.",
        status=409,
        details={
            "data_status": status,
            "hint": "Use Generate Data before running this query.",
        },
    )


def parse_timing_csv(stdout: str) -> dict[str, Any]:
    timing_rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.startswith("TIMING_CSV,"):
            continue
        row = next(csv.reader([line]))
        values = row[1:]
        parsed: dict[str, Any] = {}
        for key, value in zip(TIMING_FIELDS, values):
            if key in {"scale_factor", "query", "route", "load_source"}:
                parsed[key] = value
            elif key in {"load_bytes", "gpu_trials_n"}:
                try:
                    parsed[key] = int(float(value))
                except ValueError:
                    parsed[key] = value
            else:
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
        timing_rows.append(parsed)
    return timing_rows[-1] if timing_rows else {}


def parse_timing_block(stdout: str) -> str:
    match = re.search(r"\n\s*Timing Breakdown.*?(?=\nTIMING_CSV,|\Z)", stdout, re.DOTALL)
    return match.group(0).strip() if match else ""


def parse_result_block(stdout: str, query_name: str) -> str:
    pattern = rf"\n{re.escape(query_name)} Results:\s*\n(.*?)(?=\n\s*Timing Breakdown|\nHOST_POST_CSV,|\nTIMING_CSV,|\Z)"
    match = re.search(pattern, stdout, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def make_run_dir() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(tempfile.mkdtemp(prefix=f"{timestamp}_", dir=RUNS_DIR))


def run_codegen(payload: dict[str, Any]) -> dict[str, Any]:
    spec = validate_payload(payload)
    current_data_status = ensure_data_ready(spec["scale"])
    build = ensure_binary()

    run_dir = make_run_dir()
    query_name = "SQL" if spec["mode"] == "custom" else spec["query_id"].upper()
    cmd = [
        str(BIN_PATH),
        spec["scale"],
        "--warmup",
        str(spec["warmup"]),
        "--repeat",
        str(spec["repeat"]),
        "--full-result",
        "--dump-msl",
        str(run_dir),
    ]

    if spec["mode"] == "custom":
        sql_file = run_dir / "custom.sql"
        sql_file.write_text(spec["custom_sql"] + "\n", encoding="utf-8")
        cmd.extend(["--sql-file", str(sql_file)])
    else:
        cmd.append(spec["query_id"])

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=spec["timeout"],
        )
        timed_out = False
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        returncode = -1
    wall_ms = (time.perf_counter() - started) * 1000.0

    metal_path = run_dir / f"codegen_debug_{query_name}.metal"
    kernel_code = read_text(metal_path) if metal_path.exists() else ""
    result_text = parse_result_block(stdout, query_name)
    timing = parse_timing_csv(stdout)
    timing_block = parse_timing_block(stdout)

    ok = returncode == 0 and not timed_out
    message = "Run completed." if ok else "Run failed."
    if timed_out:
        message = f"Run timed out after {spec['timeout']} seconds."
    elif not ok and stderr.strip():
        message = stderr.strip().splitlines()[-1]

    if not result_text and not ok:
        result_text = "No result block was produced."

    return {
        "ok": ok,
        "message": message,
        "run_id": run_dir.name,
        "mode": spec["mode"],
        "query": query_name,
        "selected_query_id": spec["query_id"],
        "scale": spec["scale"],
        "command": " ".join(cmd),
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_ms": wall_ms,
        "built_backend": build["built"],
        "build_log": build["log"],
        "data_status": current_data_status,
        "kernel_code": kernel_code,
        "kernel_path": str(metal_path.relative_to(ROOT)) if metal_path.exists() else "",
        "query_result": result_text,
        "timing": timing,
        "timing_text": timing_block,
        "stdout": tail(stdout),
        "stderr": tail(stderr),
    }


def run_suite(payload: dict[str, Any]) -> dict[str, Any]:
    spec = validate_suite_payload(payload)
    current_data_status = ensure_data_ready(spec["scale"])
    build = ensure_binary()

    if not SUITE_LOCK.acquire(blocking=False):
        raise ApiError("Full suite is already running.", status=409)

    run_dir = make_run_dir()
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for n in range(1, 23):
            query_id = f"q{n}"
            query_name = f"Q{n}"
            cmd = [
                str(BIN_PATH),
                spec["scale"],
                "--warmup",
                str(spec["warmup"]),
                "--repeat",
                str(spec["repeat"]),
                "--dump-msl",
                str(run_dir),
                query_id,
            ]
            query_started = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=spec["timeout"],
                )
                timed_out = False
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""
                returncode = proc.returncode
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                returncode = -1

            wall_ms = (time.perf_counter() - query_started) * 1000.0
            timing = parse_timing_csv(stdout)
            ok = returncode == 0 and not timed_out and bool(timing)
            if timed_out:
                message = f"{query_name} timed out after {spec['timeout']} seconds."
            elif ok:
                message = f"{query_name} completed."
            elif stderr.strip():
                message = stderr.strip().splitlines()[-1]
            elif not timing:
                message = f"{query_name} produced no timing row."
            else:
                message = f"{query_name} failed."

            results.append(
                {
                    "query": query_name,
                    "query_id": query_id,
                    "ok": ok,
                    "message": message,
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "wall_ms": wall_ms,
                    "timing": timing,
                    "stderr": tail(stderr, 8000),
                }
            )
    finally:
        SUITE_LOCK.release()

    wall_ms = (time.perf_counter() - started) * 1000.0
    failures = [row for row in results if not row["ok"]]
    values = [
        row["timing"].get("query_execution_ms")
        for row in results
        if isinstance(row.get("timing"), dict)
        and isinstance(row["timing"].get("query_execution_ms"), (int, float))
    ]
    return {
        "ok": not failures and len(results) == 22,
        "message": (
            "Full suite completed."
            if not failures and len(results) == 22
            else f"Full suite finished with {len(failures)} failure(s)."
        ),
        "run_id": run_dir.name,
        "scale": spec["scale"],
        "warmup": spec["warmup"],
        "repeat": spec["repeat"],
        "wall_ms": wall_ms,
        "built_backend": build["built"],
        "build_log": build["log"],
        "data_status": current_data_status,
        "results": results,
        "summary": {
            "total": len(results),
            "ok": len(results) - len(failures),
            "failed": len(failures),
            "max_query_execution_ms": max(values) if values else 0.0,
            "min_query_execution_ms": min(values) if values else 0.0,
        },
    }


def generate_data(payload: dict[str, Any]) -> dict[str, Any]:
    scale = validate_scale(payload.get("scale"))
    timeout = clamp_int(payload.get("timeout"), 7200, 60, 86400)
    before = data_status(scale)
    if before["ready"]:
        return {
            "ok": True,
            "message": f"{scale.upper()} data is already ready.",
            "data_status": before,
            "command": "",
            "stdout": "",
            "stderr": "",
            "returncode": 0,
            "wall_ms": 0.0,
        }

    if not DATA_LOCK.acquire(blocking=False):
        raise ApiError(
            "Data generation is already running.",
            status=409,
            details={"data_status": before},
        )

    cmd = [str(PREPARE_DATA_SCRIPT), scale, "--prepare-data-only"]
    started = time.perf_counter()
    try:
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            timed_out = False
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            returncode = -1
    finally:
        DATA_LOCK.release()

    wall_ms = (time.perf_counter() - started) * 1000.0
    after = data_status(scale)
    ok = returncode == 0 and not timed_out and after["ready"]
    if timed_out:
        message = f"Data generation timed out after {timeout} seconds."
    elif ok:
        message = f"{scale.upper()} data is ready."
    elif returncode == 0:
        message = "Data generation finished, but required .colbin files are still missing."
    elif stderr.strip():
        message = stderr.strip().splitlines()[-1]
    else:
        message = "Data generation failed."

    return {
        "ok": ok,
        "message": message,
        "data_status": after,
        "command": " ".join(cmd),
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_ms": wall_ms,
        "stdout": tail(stdout),
        "stderr": tail(stderr),
    }


def json_bytes(data: Any) -> bytes:
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "GPUDBWeb/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, data: Any, status: int = 200) -> None:
        body = json_bytes(data)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            self.send_file(STATIC_DIR / "index.html")
            return
        if path == "/api/queries":
            self.send_json(
                {
                    "queries": load_queries(),
                    "scales": load_scales(),
                    "binary_exists": BIN_PATH.exists(),
                    "binary_path": str(BIN_PATH.relative_to(ROOT)),
                    "auto_build": CONFIG.auto_build,
                    "system_info": load_system_info(),
                }
            )
            return
        if path == "/api/data-status":
            query = dict(part.split("=", 1) for part in parsed.query.split("&") if "=" in part)
            try:
                self.send_json(data_status(query.get("scale", "sf1")))
            except ApiError as exc:
                self.send_json({"ok": False, "message": exc.message}, status=exc.status)
            return
        if path == "/api/health":
            self.send_json({"ok": True, "root": str(ROOT), "binary_exists": BIN_PATH.exists()})
            return
        if path == "/api/system-info":
            self.send_json(load_system_info())
            return
        if path.startswith("/static/"):
            rel = path.removeprefix("/static/")
            target = (STATIC_DIR / rel).resolve()
            try:
                target.relative_to(STATIC_DIR.resolve())
            except ValueError:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self.send_file(target)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/run", "/api/run-suite", "/api/generate-data"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length > 1024 * 1024:
            self.send_json({"ok": False, "message": "Request body is too large."}, status=413)
            return

        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ApiError("JSON body must be an object.")
            if parsed.path == "/api/run":
                result = run_codegen(payload)
            elif parsed.path == "/api/run-suite":
                result = run_suite(payload)
            else:
                result = generate_data(payload)
            self.send_json(result, status=200 if result["ok"] else 500)
        except ApiError as exc:
            response = {"ok": False, "message": exc.message}
            response.update(exc.details)
            self.send_json(response, status=exc.status)
        except json.JSONDecodeError:
            self.send_json({"ok": False, "message": "Invalid JSON body."}, status=400)
        except Exception as exc:  # pragma: no cover - defensive API boundary
            self.send_json({"ok": False, "message": str(exc)}, status=500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local GPUDB Metal web UI.")
    parser.add_argument("--host", default=os.environ.get("GPUDB_WEB_HOST", CONFIG.host))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GPUDB_WEB_PORT", CONFIG.port)))
    parser.add_argument("--timeout", type=int, default=CONFIG.default_timeout)
    parser.add_argument("--no-auto-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CONFIG.host = args.host
    CONFIG.port = args.port
    CONFIG.default_timeout = max(10, args.timeout)
    CONFIG.auto_build = not args.no_auto_build

    os.chdir(ROOT)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((CONFIG.host, CONFIG.port), Handler)
    print(f"GPUDB web UI: http://{CONFIG.host}:{CONFIG.port}")
    print(f"Repo root: {ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
