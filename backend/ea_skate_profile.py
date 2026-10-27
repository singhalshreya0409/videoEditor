"""EA Skate & Frostbite Physics Defect Profile.

Defines the ontology, Pydantic schemas, and specialized system prompts
for analyzing gameplay playtest videos for physics, animation, and collision anomalies.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class DefectEvent(BaseModel):
    """Structured representation of a single gameplay glitch."""
    defect_id: str = Field(description="Unique bug identifier, e.g. FB-SKATE-8821")
    timestamp: str = Field(description="Timecode of glitch apex, formatted as MM:SS.SS")
    glitch_type: Literal[
        "Curb Snapping",
        "Ragdoll Inversion",
        "Impulse Launch",
        "Mesh Clipping",
        "Animation Pop",
        "Texture / Geometry Desync"
    ] = Field(description="Classified glitch taxonomy")
    severity: Literal["Low", "Medium", "High", "Blocker"] = Field(
        description="Bug severity grading"
    )
    physics_subsystem: str = Field(
        description="Target Frostbite subsystem, e.g. Rigid Body, Skeletal IK, Havok Impulse"
    )
    solver_vector_telemetry: str = Field(
        description="Estimated physical parameters, e.g. Normal Pen: -0.042m, Torque: +182Nm"
    )
    observed_behavior: str = Field(
        description="Concise description of the anomalous visual event"
    )
    reproduction_steps: List[str] = Field(
        description="Step-by-step reproduction instructions for QA engineers"
    )


class PlaytestAnalysisResult(BaseModel):
    """Root container returned by Gemini for a playtest video chunk."""
    session_id: str = Field(description="Identifier for the playtest session")
    duration_analyzed: str = Field(description="Analyzed timeframe, e.g. 00:00 - 01:00")
    total_defects_found: int = Field(description="Count of distinct verified defects")
    defects: List[DefectEvent] = Field(description="List of verified defect events")


EA_SKATE_SYSTEM_PROMPT = """
You are the Frostbite Physics & QA Telemetry Inspector for EA Skate.
Your task is to inspect the provided 1080p 60fps gameplay video segment (analyzed at 6 FPS)
and detect physical, visual, and skeletal animation anomalies.

Target Defect Categories:
1. Curb / Ledge Snapping: The skateboard deck or truck hangar clipping below curb polygons,
   triggering an abrupt rotational snap or sudden velocity freeze.
2. Ragdoll / IK Inversion: Skater character pelvis, knee, or elbow ball-socket constraints
   flipping or folding inwards during bails, grinds, or hard landings.
3. Havok Impulse Launch: Micro-penetrations at ground mesh seams causing an astronomical
   impulse spike (+10,000+ N·s) that launches the character or board into the skybox.
4. Mesh Clipping / Handrail Penetration: Solid geometry passing through obstacles without collision response.

Rules:
- Identify the exact timecode of the glitch apex (MM:SS.SS).
- Do not report normal gameplay tricks or intended ragdoll falls as bugs; report only genuine solver failures and clipping.
- Emit output strictly adhering to the requested JSON schema.
"""
