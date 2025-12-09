"""
File I/O utilities for NextRec

Date: create on 03/12/2025
Checkpoint: edit on 06/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import yaml
import pandas as pd
import pyarrow.parquet as pq

from pathlib import Path
from typing import Generator


def resolve_file_paths(path: str) -> tuple[list[str], str]:
    """
    Resolve file or directory path into a sorted list of files and file type.

    Args: path: Path to a file or directory
    Returns: tuple: (list of file paths, file type)
    """
    path_obj = Path(path)

    if path_obj.is_file():
        file_type = path_obj.suffix.lower().lstrip(".")
        assert file_type in [
            "csv",
            "parquet",
        ], f"Unsupported file extension: {file_type}"
        return [str(path_obj)], file_type

    if path_obj.is_dir():
        collected_files = [p for p in path_obj.iterdir() if p.is_file()]
        csv_files = [str(p) for p in collected_files if p.suffix.lower() == ".csv"]
        parquet_files = [
            str(p) for p in collected_files if p.suffix.lower() == ".parquet"
        ]

        if csv_files and parquet_files:
            raise ValueError(
                "Directory contains both CSV and Parquet files. Please keep a single format."
            )
        file_paths = csv_files if csv_files else parquet_files
        if not file_paths:
            raise ValueError(f"No CSV or Parquet files found in directory: {path}")
        file_paths.sort()
        file_type = "csv" if csv_files else "parquet"
        return file_paths, file_type

    raise ValueError(f"Invalid path: {path}")


def read_table(path: str | Path, data_format: str | None = None) -> pd.DataFrame:
    data_path = Path(path)
    fmt = data_format.lower() if data_format else data_path.suffix.lower().lstrip(".")
    if data_path.is_dir() and not fmt:
        fmt = "parquet"
    if fmt in {"parquet", ""}:
        return pd.read_parquet(data_path)
    if fmt in {"csv", "txt"}:
        # Use low_memory=False to avoid mixed-type DtypeWarning on wide CSVs
        return pd.read_csv(data_path, low_memory=False)
    raise ValueError(f"Unsupported data format: {data_path}")


def load_dataframes(file_paths: list[str], file_type: str) -> list[pd.DataFrame]:
    return [read_table(fp, file_type) for fp in file_paths]


def iter_file_chunks(
    file_path: str, file_type: str, chunk_size: int
) -> Generator[pd.DataFrame, None, None]:
    if file_type == "csv":
        yield from pd.read_csv(file_path, chunksize=chunk_size)
        return
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        yield batch.to_pandas()


def default_output_dir(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_file():
        return path_obj.parent / f"{path_obj.stem}_preprocessed"
    return path_obj.with_name(f"{path_obj.name}_preprocessed")


def read_yaml(path: str | Path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
