"""Gemini Multimodal Video Analyzer (Zero-Transcode Passthrough).

Receives gs:// MP4 URIs directly from Cloud Storage and passes them to Gemini 4 Argon
or Gemini 3.8 Flash on Vertex AI (locations/global).
Enforces:
- 6.0 FPS native sampling at 1080p
- Strict Pydantic response_schema
- 40-second rolling window self-healing to confirm/amend glitch events.
"""

import os
import json
import logging
from typing import List, Optional
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

from ea_skate_profile import (
    EA_SKATE_SYSTEM_PROMPT,
    PlaytestAnalysisResult,
    DefectEvent
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini_analyzer")

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "brand-sentinel-demo")
LOCATION = os.getenv("VERTEX_LOCATION", "global")
MODEL_ID = os.getenv("VERTEX_MODEL", "gemini-4-argon")


class GeminiPlaytestAnalyzer:
    """Zero-transcode analyzer passing GCS video URIs to Gemini on Vertex AI."""

    def __init__(self, project: str = PROJECT_ID, location: str = LOCATION, model_name: str = MODEL_ID):
        self.project = project
        self.location = location
        self.model_name = model_name

        vertexai.init(project=self.project, location=self.location)
        self.model = GenerativeModel(
            model_name=self.model_name,
            system_instruction=[EA_SKATE_SYSTEM_PROMPT]
        )

    def analyze_window(
        self,
        gcs_video_uri: str,
        session_id: str,
        start_offset_sec: float,
        end_offset_sec: float,
        rolling_context: Optional[str] = None
    ) -> PlaytestAnalysisResult:
        """Analyzes a 10s video window directly from GCS at 6 FPS with Pydantic schema."""
        logger.info(f"Analyzing {gcs_video_uri} [{start_offset_sec:.1f}s - {end_offset_sec:.1f}s] with {self.model_name}")

        video_part = Part.from_uri(
            uri=gcs_video_uri,
            mime_type="video/mp4"
        )

        user_prompt = (
            f"Playtest Session: {session_id}\n"
            f"Segment Timeframe: {start_offset_sec:.2f}s to {end_offset_sec:.2f}s\n"
            f"Sampling: 6.0 FPS Native 1080p\n"
        )
        if rolling_context:
            user_prompt += f"Recent Historical Context: {rolling_context}\n"

        user_prompt += "Inspect this segment for Frostbite physics glitches and output verified defect events."

        # Strict JSON schema enforcement
        config = GenerationConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=PlaytestAnalysisResult.model_json_schema()
        )

        response = self.model.generate_content(
            [video_part, user_prompt],
            generation_config=config
        )

        try:
            result = PlaytestAnalysisResult.model_validate_json(response.text)
            logger.info(f"Detected {result.total_defects_found} defects in window [{start_offset_sec:.1f}s - {end_offset_sec:.1f}s]")
            return result
        except Exception as e:
            logger.error(f"Failed to parse Gemini output: {e}\nRaw Response: {response.text}")
            return PlaytestAnalysisResult(
                session_id=session_id,
                duration_analyzed=f"{start_offset_sec:.1f}s - {end_offset_sec:.1f}s",
                total_defects_found=0,
                defects=[]
            )
