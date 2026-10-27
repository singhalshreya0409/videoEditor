"""Netflix-Inspired Stream-Copy Ingestion Chunker.

Slices long gameplay recordings (10 min - 1 hour, 1 GB - 5 GB) into contiguous
10-second windows using FFmpeg stream-copy (-c copy) in ~0.2s without re-encoding.
Uploads window chunks directly to Google Cloud Storage (GCS) for zero-transcode
Gemini multimodal analysis.
"""

import os
import subprocess
import logging
from typing import List, Dict, Any
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest_chunker")


class PlaytestStreamChunker:
    """Manages high-speed stream-copy windowing for gameplay playtests."""

    def __init__(self, gcs_bucket_name: str, window_duration_sec: int = 10):
        self.gcs_bucket_name = gcs_bucket_name
        self.window_duration_sec = window_duration_sec
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(gcs_bucket_name)

    def slice_and_upload_windows(
        self,
        input_video_path: str,
        session_id: str,
        total_duration_sec: float,
        output_dir: str = "/tmp/chunks"
    ) -> List[Dict[str, Any]]:
        """Slices video into 10s MP4 windows and stream-copies to GCS."""
        os.makedirs(output_dir, exist_ok=True)
        chunks = []

        start_time = 0.0
        window_idx = 0

        while start_time < total_duration_sec:
            chunk_filename = f"{session_id}_win_{window_idx:04d}.mp4"
            local_chunk_path = os.path.join(output_dir, chunk_filename)
            gcs_blob_name = f"playtests/{session_id}/windows/{chunk_filename}"

            # Stream-copy slice in ~0.2s (Zero Re-encoding)
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", input_video_path,
                "-t", str(self.window_duration_sec),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                local_chunk_path
            ]

            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                # Upload to Cloud Storage
                blob = self.bucket.blob(gcs_blob_name)
                blob.upload_from_filename(local_chunk_path)

                gcs_uri = f"gs://{self.gcs_bucket_name}/{gcs_blob_name}"
                logger.info(f"Stream-copied window {window_idx}: {start_time:.1f}s -> {gcs_uri}")

                chunks.append({
                    "window_index": window_idx,
                    "start_offset_sec": start_time,
                    "end_offset_sec": min(start_time + self.window_duration_sec, total_duration_sec),
                    "gcs_uri": gcs_uri,
                    "local_path": local_chunk_path
                })
            except Exception as e:
                logger.error(f"Failed to chunk window {window_idx} at {start_time}s: {e}")
            finally:
                # Clean up local temporary file
                if os.path.exists(local_chunk_path):
                    try:
                        os.remove(local_chunk_path)
                    except OSError:
                        pass

            start_time += self.window_duration_sec
            window_idx += 1

        return chunks
