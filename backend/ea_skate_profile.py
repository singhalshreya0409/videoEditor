"""EA Skate & Frostbite Physics Defect Profile.

Defines the ontology, schemas, and specialized system prompts
for analyzing gameplay playtest videos for physics, animation, and collision anomalies.
Supports both Pydantic v2 and a zero-dependency fallback.
"""

from typing import List, Literal, Optional, Dict, Any
import json

try:
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

except ImportError:
    # Zero-dependency fallback for environments without pydantic installed
    class DefectEvent:
        def __init__(self, **kwargs):
            self.defect_id = kwargs.get("defect_id", "")
            self.timestamp = kwargs.get("timestamp", "")
            self.glitch_type = kwargs.get("glitch_type", "")
            self.severity = kwargs.get("severity", "")
            self.physics_subsystem = kwargs.get("physics_subsystem", "")
            self.solver_vector_telemetry = kwargs.get("solver_vector_telemetry", "")
            self.observed_behavior = kwargs.get("observed_behavior", "")
            self.reproduction_steps = kwargs.get("reproduction_steps", [])

        def to_dict(self) -> Dict[str, Any]:
            return {
                "defect_id": self.defect_id,
                "timestamp": self.timestamp,
                "glitch_type": self.glitch_type,
                "severity": self.severity,
                "physics_subsystem": self.physics_subsystem,
                "solver_vector_telemetry": self.solver_vector_telemetry,
                "observed_behavior": self.observed_behavior,
                "reproduction_steps": self.reproduction_steps
            }

    class PlaytestAnalysisResult:
        def __init__(self, **kwargs):
            self.session_id = kwargs.get("session_id", "")
            self.duration_analyzed = kwargs.get("duration_analyzed", "")
            self.total_defects_found = kwargs.get("total_defects_found", 0)
            raw_defects = kwargs.get("defects", [])
            self.defects = [d if isinstance(d, DefectEvent) else DefectEvent(**d) for d in raw_defects]

        @classmethod
        def model_json_schema(cls) -> Dict[str, Any]:
            return PLAYTEST_ANALYSIS_SCHEMA

        @classmethod
        def model_validate_json(cls, json_str: str) -> "PlaytestAnalysisResult":
            data = json.loads(json_str)
            return cls(**data)

        def model_dump_json(self, indent: Optional[int] = None) -> str:
            d = {
                "session_id": self.session_id,
                "duration_analyzed": self.duration_analyzed,
                "total_defects_found": self.total_defects_found,
                "defects": [def_item.to_dict() if hasattr(def_item, "to_dict") else def_item for def_item in self.defects]
            }
            return json.dumps(d, indent=indent)


PLAYTEST_ANALYSIS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "session_id": {"type": "STRING"},
        "duration_analyzed": {"type": "STRING"},
        "total_defects_found": {"type": "INTEGER"},
        "defects": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "defect_id": {"type": "STRING"},
                    "timestamp": {"type": "STRING"},
                    "glitch_type": {
                        "type": "STRING",
                        "enum": [
                            "Curb Snapping",
                            "Ragdoll Inversion",
                            "Impulse Launch",
                            "Mesh Clipping",
                            "Animation Pop",
                            "Texture / Geometry Desync"
                        ]
                    },
                    "severity": {
                        "type": "STRING",
                        "enum": ["Low", "Medium", "High", "Blocker"]
                    },
                    "physics_subsystem": {"type": "STRING"},
                    "solver_vector_telemetry": {"type": "STRING"},
                    "observed_behavior": {"type": "STRING"},
                    "reproduction_steps": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    }
                },
                "required": [
                    "defect_id",
                    "timestamp",
                    "glitch_type",
                    "severity",
                    "physics_subsystem",
                    "solver_vector_telemetry",
                    "observed_behavior",
                    "reproduction_steps"
                ]
            }
        }
    },
    "required": ["session_id", "duration_analyzed", "total_defects_found", "defects"]
}


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
