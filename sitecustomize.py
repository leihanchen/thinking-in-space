"""Runtime compatibility shims for the evaluation container."""

from __future__ import annotations


def _patch_torchvision_read_video() -> None:
    try:
        import av
        import numpy as np
        import torch
        import torchvision.io as torchvision_io
    except Exception:
        return

    if hasattr(torchvision_io, "read_video"):
        return

    def _read_video(
        filename,
        start_pts=0.0,
        end_pts=None,
        pts_unit="sec",
        output_format="TCHW",
    ):
        if pts_unit != "sec":
            raise ValueError("sitecustomize read_video shim only supports pts_unit='sec'")
        if output_format != "TCHW":
            raise ValueError("sitecustomize read_video shim only supports output_format='TCHW'")

        if isinstance(filename, bytes):
            filename = filename.decode("utf-8")
        if isinstance(filename, str) and filename.startswith("file://"):
            filename = filename[7:]

        container = av.open(filename)
        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate) if video_stream.average_rate is not None else 0.0
        if fps <= 0 and video_stream.time_base is not None:
            fps = 1.0 / float(video_stream.time_base)

        frames = []
        for frame in container.decode(video=0):
            frame_time = frame.time
            if frame_time is None and fps > 0:
                frame_time = len(frames) / fps
            if frame_time is None:
                frame_time = 0.0
            if frame_time < start_pts:
                continue
            if end_pts is not None and frame_time > end_pts:
                break
            frames.append(torch.from_numpy(np.asarray(frame.to_ndarray(format="rgb24"))))

        if frames:
            video = torch.stack(frames).permute(0, 3, 1, 2)
        else:
            video = torch.empty((0, 3, 0, 0), dtype=torch.uint8)

        audio = torch.empty((0,), dtype=torch.float32)
        info = {"video_fps": fps if fps > 0 else None}
        return video, audio, info

    torchvision_io.read_video = _read_video


_patch_torchvision_read_video()