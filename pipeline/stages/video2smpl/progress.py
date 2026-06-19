"""Terminal progress display for video2smpl batch runs."""

from __future__ import annotations

from typing import Optional

from tqdm import tqdm


class Video2SmplProgress:
    def __init__(self, total: int, *, desc: str = "video2smpl", enabled: bool = True) -> None:
        self._ok = 0
        self._err = 0
        self._pbar: Optional[tqdm] = None
        if enabled and total > 0:
            self._pbar = tqdm(
                total=total,
                desc=desc,
                unit="sample",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )

    def update(self, sample_id: str, note: str) -> None:
        if note == "ok":
            self._ok += 1
        else:
            self._err += 1
        if self._pbar is not None:
            self._pbar.set_postfix_str(
                f"ok={self._ok} err={self._err} last={sample_id}",
                refresh=False,
            )
            self._pbar.update(1)

    def write(self, msg: str) -> None:
        if self._pbar is not None:
            self._pbar.write(msg)
        else:
            print(msg, flush=True)

    def close(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
