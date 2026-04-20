"""
Evaluate the trained UNet model on the full VoiceBank+DEMAND test set.

Results saved to:
    test/enhanced/metrics.csv     — one row per file, opens in Excel
    test/enhanced/metrics.xlsx    — same data, formatted Excel workbook
    test/enhanced/summary.json    — mean scores for the README

Usage:
    python -m src.evaluation.evaluate_testset
    python -m src.evaluation.evaluate_testset \
        --noisy      data/processed/voicebank_demand/test/noisy \
        --clean      data/processed/voicebank_demand/test/clean \
        --out_dir    data/processed/voicebank_demand/test/enhanced \
        --checkpoint models/unet_best.pt
"""

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

from tqdm import tqdm

from src.inference.denoise import denoise_audio
from src.evaluation.metrics import evaluate_file_pair


def _to_py(val):
    """Convert any numpy scalar to plain Python float for json.dump."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return val


def _write_xlsx(rows: list[dict], out_path: Path):
    """Write metrics rows to a formatted .xlsx file. Requires openpyxl."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
    except ImportError:
        print("  openpyxl not installed — skipping .xlsx export. Run: pip install openpyxl")
        return

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Metrics"

    # Header row
    headers = ["File", "SNR (dB)", "PESQ", "STOI"]
    header_fill = PatternFill("solid", fgColor="4472C4")
    header_font = Font(color="FFFFFF", bold=True)
    for col, header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")

    # Data rows
    for row_idx, row in enumerate(rows, start=2):
        ws.cell(row=row_idx, column=1, value=row["file"])
        for col, key in enumerate(["snr", "pesq", "stoi"], start=2):
            val = row[key]
            ws.cell(row=row_idx, column=col, value=val if val != "" else None)

    # Column widths
    ws.column_dimensions["A"].width = 28
    for col in ["B", "C", "D"]:
        ws.column_dimensions[col].width = 12

    # Summary rows at the bottom
    n = len(rows)
    ws.cell(row=n + 3, column=1, value="Mean").font = Font(bold=True)
    for col, key in enumerate(["snr", "pesq", "stoi"], start=2):
        vals = [r[key] for r in rows if r[key] != ""]
        if vals:
            mean_val = round(sum(vals) / len(vals), 4)
            cell = ws.cell(row=n + 3, column=col, value=mean_val)
            cell.font = Font(bold=True)

    # Write to temp then rename (avoids PermissionError if file is open)
    tmp = out_path.with_suffix(".tmp.xlsx")
    wb.save(tmp)
    if out_path.exists():
        out_path.unlink()
    tmp.rename(out_path)
    print(f"  {out_path}")


def run(noisy_dir: Path, clean_dir: Path, out_dir: Path, checkpoint: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    noisy_files = sorted(noisy_dir.glob("*.wav"))
    if not noisy_files:
        raise FileNotFoundError(f"No .wav files in {noisy_dir}")

    csv_path     = out_dir / "metrics.csv"
    xlsx_path    = out_dir / "metrics.xlsx"
    summary_path = out_dir / "summary.json"

    rows = []

    # Write CSV via a temp file to avoid PermissionError if metrics.csv is open
    tmp_csv = out_dir / "metrics.tmp.csv"
    with open(tmp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "snr", "pesq", "stoi"])
        writer.writeheader()

        for noisy_path in tqdm(noisy_files, desc="Evaluating"):
            enhanced_path = out_dir / noisy_path.name
            clean_path    = clean_dir / noisy_path.name

            if not clean_path.exists():
                print(f"  WARNING: no clean reference for {noisy_path.name}, skipping")
                continue

            # Skip inference if enhanced file already exists
            if not enhanced_path.exists():
                denoise_audio(
                    str(noisy_path),
                    output_path=str(enhanced_path),
                    ckpt_path=checkpoint,
                )

            result = evaluate_file_pair(str(clean_path), str(enhanced_path))

            def fmt(v):
                return round(_to_py(v), 3) if v is not None else ""

            row = {
                "file": noisy_path.name,
                "snr":  fmt(result["snr"]),
                "pesq": fmt(result["pesq"]),
                "stoi": fmt(result["stoi"]),
            }
            writer.writerow(row)
            rows.append(row)

    # Atomic rename: replaces metrics.csv only after writing is complete
    if csv_path.exists():
        csv_path.unlink()
    tmp_csv.rename(csv_path)

    # Excel export
    _write_xlsx(rows, xlsx_path)

    # Summary JSON
    def _mean(key):
        vals = [r[key] for r in rows if r[key] != ""]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "n_files":   len(rows),
        "snr_mean":  _to_py(_mean("snr")),
        "pesq_mean": _to_py(_mean("pesq")),
        "stoi_mean": _to_py(_mean("stoi")),
    }

    tmp_json = out_dir / "summary.tmp.json"
    with open(tmp_json, "w") as f:
        json.dump(summary, f, indent=2)
    if summary_path.exists():
        summary_path.unlink()
    tmp_json.rename(summary_path)

    print(f"\n{'─'*50}")
    print(f"Files evaluated : {summary['n_files']}")
    print(f"Mean SNR        : {summary['snr_mean']} dB")
    print(f"Mean PESQ       : {summary['pesq_mean']}")
    print(f"Mean STOI       : {summary['stoi_mean']}")
    print(f"\nResults saved to:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy",      default="data/processed/voicebank_demand/test/noisy")
    parser.add_argument("--clean",      default="data/processed/voicebank_demand/test/clean")
    parser.add_argument("--out_dir",    default="data/processed/voicebank_demand/test/enhanced")
    parser.add_argument("--checkpoint", default="models/unet_best.pt")
    args = parser.parse_args()

    run(Path(args.noisy), Path(args.clean), Path(args.out_dir), args.checkpoint)