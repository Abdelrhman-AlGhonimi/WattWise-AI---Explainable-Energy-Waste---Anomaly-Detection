from __future__ import annotations
import io
import logging
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
from .config import RAW_DATA, UCI_URL
from .validation import validate_frame

LOG = logging.getLogger(__name__)


def load_data(path: str | Path = RAW_DATA, download: bool = True) -> pd.DataFrame:
    """Load locally cached UCI data, downloading official archive if absent."""
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"Dataset not found at {path}; download from https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(UCI_URL, timeout=60)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                member = next(
                    n
                    for n in archive.namelist()
                    if n.endswith("energydata_complete.csv")
                )
                fd, temporary = tempfile.mkstemp(
                    prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
                )
                os.close(fd)
                try:
                    Path(temporary).write_bytes(archive.read(member))
                    os.replace(temporary, path)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".source.txt").write_text(
                f"Source: {UCI_URL}\nDownloaded UTC: {datetime.now(timezone.utc).isoformat()}\n",
                encoding="utf-8",
            )
        except Exception as exc:
            raise RuntimeError(
                f"UCI download failed: {exc}. Manually download the official archive and place energydata_complete.csv in {path.parent}."
            ) from exc
    frame = pd.read_csv(path)
    validate_frame(frame)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    LOG.info("Loaded %s rows x %s columns from %s", *frame.shape, path)
    # Keep duplicate rows until clean_data can remove them and record the
    # affected counts; silently dropping them here made cleaning reports blind.
    return frame.sort_values("date").reset_index(drop=True)
