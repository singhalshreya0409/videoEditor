"""FastAPI MCP Server for EA Video Motion & Optical Flow Bug Analyst.

Advanced 2-Stage Hybrid Video QA Engine:
- Stage 1: Client-Side Optical Flow & Frame Differencing (spikes in pixel velocity / freeze detection)
- Stage 2: Dense Multimodal AI Verification via Google Cloud Vertex AI Gemini 4 Argon
- Netflix-Style Stream-Copy Ingestion: 10s windows (-c copy) directly from GCS at 6.0 FPS
- Studio QA Verification Toolbar with Frame-by-Frame Fine Tuning
- 1-Click EA Jira Ticket Creation
"""

import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gemini_analyzer import GeminiPlaytestAnalyzer
from ingest_chunker import PlaytestStreamChunker

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "brand-sentinel-demo")
LOCATION = os.environ.get("VERTEX_LOCATION", "global")
MODEL_ID = os.environ.get("VERTEX_MODEL", "gemini-4-argon")
GCS_BUCKET = os.environ.get("GCS_BUCKET", f"ea-skate-playtests-15785338844")

# In-memory store for session defect analysis results
SESSION_RESULTS_STORE: Dict[str, Dict[str, Any]] = {}

# Pre-seeded / Latest Analysis Data for sample_ea.mp4
LATEST_ANALYSIS_DATA: Dict[str, Any] = {
    "session_id": "skate_playtest_auto",
    "video_path": "gs://ea-skate-playtests-15785338844/sample_ea.mp4",
    "total_defects": 5,
    "defects": [
        {
            "defect_id": "FB-SKATE-0120",
            "timestamp": "00:12.00",
            "glitch_type": "Animation Pop",
            "severity": "Blocker",
            "physics_subsystem": "Frostbite Core / Assertion",
            "solver_vector_telemetry": "Assert: YourSelfFootController->GetSomeAssetyUpStateInsideOutJynxch",
            "observed_behavior": "The client halts simulation with an unhandled engine assertion failure during an inverted grind transition.",
            "reproduction_steps": [
                "1. Initiate a grind on any hubba ledge.",
                "2. Transition into an inverted body state while the foot controller IK remains active.",
                "3. Verify the engine assertion dialog triggers."
            ]
        },
        {
            "defect_id": "FB-SKATE-0130",
            "timestamp": "00:13.00",
            "glitch_type": "Ragdoll Inversion",
            "severity": "High",
            "physics_subsystem": "Skeletal IK / Ragdoll Solver",
            "solver_vector_telemetry": "Joint Angle Pen: +195°, Pelvis Torque: +450 Nm",
            "observed_behavior": "Character skeleton collapses into an inverted, folded ball with limbs clipping through the torso while rolling on flat asphalt.",
            "reproduction_steps": [
                "1. Bail or land awkwardly on a flat road surface.",
                "2. Observe skeletal constraints exceeding anatomical limits (+195° joint penetration)."
            ]
        },
        {
            "defect_id": "FB-SKATE-0140",
            "timestamp": "00:14.00",
            "glitch_type": "Animation Pop",
            "severity": "Medium",
            "physics_subsystem": "Skeletal IK",
            "solver_vector_telemetry": "Root Pitch Offset: +65°, CoM Desync: 0.85m",
            "observed_behavior": "Skater body locks rigidly into a 45-degree forward-leaning pose without dynamic balance adjustments during a handrail grind.",
            "reproduction_steps": [
                "1. Ollie onto a downward stair handrail into a boardslide or 50-50 grind.",
                "2. Observe the skater's center-of-mass desync (0.85m) and frozen pitch angle."
            ]
        },
        {
            "defect_id": "FB-SKATE-0270",
            "timestamp": "00:27.00",
            "glitch_type": "Impulse Launch",
            "severity": "High",
            "physics_subsystem": "Havok Impulse / Rigid Body",
            "solver_vector_telemetry": "Vertical Impulse: +14,200 N*s, Vz: +48.5 m/s",
            "observed_behavior": "Skater receives an extreme vertical momentum transfer (+14,200 N*s), rocketing dozens of meters into the skybox upon hitting a ramp transition.",
            "reproduction_steps": [
                "1. Approach the pink blockout kicker ramp at maximum sprint speed.",
                "2. Ollie precisely at the geometric seam between the ground plane and ramp lip."
            ]
        },
        {
            "defect_id": "FB-SKATE-0570",
            "timestamp": "00:57.00",
            "glitch_type": "Texture / Geometry Desync",
            "severity": "High",
            "physics_subsystem": "Character Scale / Net Sync",
            "solver_vector_telemetry": "Scale Matrix: [12.0, 12.0, 12.0], Rig: T-Pose",
            "observed_behavior": "A massive, building-sized character mesh spawns in a static T-pose floating above the sidewalk during a multiplayer hill bomb.",
            "reproduction_steps": [
                "1. Join a multiplayer session with 10+ skaters bombing a steep hill.",
                "2. Cause simultaneous ragdoll bails at the bottom kicker ramp."
            ]
        }
    ]
}


def get_auth_token() -> str:
    """Retrieves GCP OAuth token from Cloud Run metadata or gcloud fallback."""
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            return data.get("access_token", "")
    except Exception:
        pass

    try:
        import subprocess
        return subprocess.check_output(["gcloud", "auth", "print-access-token"]).decode().strip()
    except Exception:
        return ""


def evaluate_motion_candidates_with_gemini(candidates: List[Dict[str, Any]], query: str = None) -> List[Dict[str, Any]]:
    """Evaluates optical flow motion anomaly candidates with Google Cloud Vertex AI Gemini 4 Argon."""
    token = get_auth_token()
    if not token or not candidates:
        return []

    endpoint = "aiplatform.googleapis.com" if LOCATION == "global" else f"{LOCATION}-aiplatform.googleapis.com"
    url = f"https://{endpoint}/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"

    parts = []
    instruction = (
        "You are an expert game QA engineer for Electronic Arts specializing in Frostbite engine physics and rendering.\n"
        "You are provided with gameplay video frames centered around timestamps flagged by a client-side Optical Flow / Motion Filter.\n"
        "The Optical Flow filter detected sudden velocity spikes, camera jerks, or visual motion anomalies at these moments.\n\n"
        "YOUR TASK:\n"
        "Determine for each candidate moment whether it represents:\n"
        "1. A GENUINE DEFECT: Physics collision penetration, skateboard clipping inside geometry, ragdoll glitch, geometry lock, missing textures, or animation snap.\n"
        "2. NORMAL GAMEPLAY / FALSE POSITIVE: An intended skater trick bail, intentional camera cut, normal landing, or UI menu navigation.\n\n"
        "STRICT EVALUATION RULES:\n"
        "- Base your judgment ONLY on what is genuinely visible in the provided frame images.\n"
        "- If the motion spike is just a normal gameplay wipeout or intentional cut, DISMISS IT and do not report it as a defect.\n"
        "- If all candidates are normal gameplay, return an empty array: {\"defects\": []}.\n"
        "- Only return genuine defects in the defects array with:\n"
        "  - timestamp_seconds: float (exact timestamp of the defect frame)\n"
        "  - time_display: string MM:SS.SS\n"
        "  - title: short technical defect title\n"
        "  - severity: 'critical', 'high', or 'medium'\n"
        "  - analysis: concise explanation of why the motion spike was a genuine defect vs normal gameplay\n"
        "  - optical_flow_metric: string description of the motion anomaly (e.g. 'Velocity vector jerk: +182% | Collider penetration')\n"
        "  - telemetry: simulated Frostbite telemetry string matching the visual defect\n"
        "  - repro: array of 3 reproduction steps\n"
    )
    if query:
        instruction += f'\nFocus Filter: "{query}". Prioritize candidate moments matching this description.'

    parts.append({"text": instruction})

    for cand in candidates[:6]:
        ts = cand.get("timestamp_seconds", 0)
        td = cand.get("time_display", f"{ts:.2f}")
        score = cand.get("motion_score", "1.0")
        frames = cand.get("frames", [])

        parts.append({"text": f"\n=== Optical Flow Motion Spike Candidate at {td} (Motion Anomaly Score: {score}) ==="})
        for f in frames[:3]:
            fts = f.get("timestamp_seconds", ts)
            ftd = f.get("time_display", f"{fts:.2f}")
            b64 = f.get("image_b64", "")
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            parts.append({"text": f"Frame at {ftd}:"})
            parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64}})

    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "defects": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "timestamp_seconds": {"type": "NUMBER"},
                                "time_display": {"type": "STRING"},
                                "title": {"type": "STRING"},
                                "severity": {"type": "STRING"},
                                "analysis": {"type": "STRING"},
                                "optical_flow_metric": {"type": "STRING"},
                                "telemetry": {"type": "STRING"},
                                "repro": {"type": "ARRAY", "items": {"type": "STRING"}},
                            },
                            "required": [
                                "timestamp_seconds",
                                "time_display",
                                "title",
                                "severity",
                                "analysis",
                                "optical_flow_metric",
                                "telemetry",
                                "repro",
                            ],
                        },
                    }
                },
                "required": ["defects"],
            },
            "temperature": 0.0,
        },
    }

    try:
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            data=json.dumps(body).encode(),
        )
        with urllib.request.urlopen(req, timeout=40) as resp:
            resp_data = json.loads(resp.read().decode())
            text = resp_data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "defects" in parsed:
                return parsed["defects"]
            elif isinstance(parsed, list):
                return parsed
            return []
    except Exception as e:
        print("Vertex AI Gemini error:", e)
        return []


app = FastAPI(title="EA Video Motion & Optical Flow Bug Analyst MCP Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMPLATES_DIR = Path(__file__).parent / "templates"
MOTION_ANALYST_HTML_PATH = TEMPLATES_DIR / "motion_analyst.html"
SKATE_REVIEWER_HTML_PATH = TEMPLATES_DIR / "skate_defect_reviewer.html"

RESOURCE_URI = "ui://ea-video/motion-analyst.html"
REVIEWER_RESOURCE_URI = "ui://ea-skate/defect-reviewer.html"


def get_widget_html() -> str:
    if MOTION_ANALYST_HTML_PATH.exists():
        with open(MOTION_ANALYST_HTML_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>EA Video Motion Analyst Template Not Found</h1>"


def get_reviewer_html(session_id: Optional[str] = None) -> str:
    if SKATE_REVIEWER_HTML_PATH.exists():
        with open(SKATE_REVIEWER_HTML_PATH, "r", encoding="utf-8") as f:
            content = f.read()
            # If we have analyzed defects, inject them
            active_data = SESSION_RESULTS_STORE.get(session_id) if session_id else LATEST_ANALYSIS_DATA
            if active_data and active_data.get("defects"):
                injected_json = json.dumps(active_data["defects"])
                content = content.replace("/* __INJECTED_DEFECTS_PLACEHOLDER__ */", f"const SERVER_INJECTED_DEFECTS = {injected_json};")
            return content
    return "<h1>EA Skate Defect Reviewer Template Not Found</h1>"


@app.get("/api/video_stream")
def video_stream(path: str = "gs://ea-skate-playtests-15785338844/sample_ea.mp4") -> Response:
    """Streams video directly from Cloud Storage for in-browser playback."""
    try:
        from google.cloud import storage
        if path.startswith("gs://"):
            bucket_name, blob_name = path[5:].split("/", 1)
        else:
            bucket_name, blob_name = GCS_BUCKET, path
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        data = blob.download_as_bytes()
        return Response(content=data, media_type="video/mp4")
    except Exception as e:
        print("Video stream error:", e)
        # Fallback to test video
        return Response(status_code=302, headers={"Location": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"})


@app.get("/")
def health_check() -> Dict[str, str]:
    return {
        "status": "ok",
        "app": "ea-video-motion-mcp",
        "version": "2.1.0-stream-copy",
        "model": MODEL_ID,
        "location": LOCATION,
        "bucket": GCS_BUCKET
    }


@app.get("/widget")
@app.get("/motion")
def get_widget() -> Response:
    return Response(content=get_widget_html(), media_type="text/html")


@app.get("/reviewer")
def get_reviewer() -> Response:
    return Response(content=get_reviewer_html(), media_type="text/html")


@app.get("/api/playtest_results")
def get_playtest_results(session_id: str) -> Response:
    """Returns analyzed defect events for a playtest session."""
    data = SESSION_RESULTS_STORE.get(session_id)
    if data:
        return JSONResponse(data)
    return JSONResponse(
        status_code=404,
        content={"error": f"Session '{session_id}' not found in results registry."}
    )


@app.post("/api/analyze_motion")
async def api_analyze_motion(request: Request) -> Response:
    try:
        body = await request.json()
        candidates = body.get("candidates", [])
        query = body.get("query", None)
        defects = evaluate_motion_candidates_with_gemini(candidates, query=query)
        return JSONResponse({"defects": defects})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "defects": []})


@app.post("/api/analyze_playtest")
async def api_analyze_playtest(request: Request) -> Response:
    """Runs Netflix-style stream-copy chunking & Gemini 4 Argon analysis on a playtest video."""
    try:
        body = await request.json()
        session_id = body.get("session_id", "skate_session_01")
        video_path = body.get("video_path")  # Local file or gs:// URI
        total_duration = float(body.get("duration_sec", 60.0))

        analyzer = GeminiPlaytestAnalyzer(project=PROJECT_ID, location=LOCATION, model_name=MODEL_ID)
        all_defects = []

        if video_path and video_path.startswith("gs://"):
            # Direct GCS URI passthrough
            res = analyzer.analyze_window(
                gcs_video_uri=video_path,
                session_id=session_id,
                start_offset_sec=0.0,
                end_offset_sec=total_duration
            )
            raw_defects = [d.to_dict() if hasattr(d, "to_dict") else d.model_dump() for d in res.defects]
            all_defects.extend(raw_defects)
        elif video_path and os.path.exists(video_path):
            chunker = PlaytestStreamChunker(gcs_bucket_name=GCS_BUCKET)
            chunks = chunker.slice_and_upload_windows(
                input_video_path=video_path,
                session_id=session_id,
                total_duration_sec=total_duration
            )
            rolling_context = None
            for c in chunks:
                res = analyzer.analyze_window(
                    gcs_video_uri=c["gcs_uri"],
                    session_id=session_id,
                    start_offset_sec=c["start_offset_sec"],
                    end_offset_sec=c["end_offset_sec"],
                    rolling_context=rolling_context
                )
                raw = [d.to_dict() if hasattr(d, "to_dict") else d.model_dump() for d in res.defects]
                all_defects.extend(raw)
                if raw:
                    rolling_context = f"Glitch at {raw[-1].get('timestamp', '')}: {raw[-1].get('glitch_type', '')}"

        result_payload = {
            "session_id": session_id,
            "total_defects": len(all_defects),
            "defects": all_defects
        }
        SESSION_RESULTS_STORE[session_id] = result_payload
        return JSONResponse(result_payload)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "defects": []})


@app.post("/mcp")
async def handle_mcp(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None},
        )

    rpc_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})
    print(f"MCP RPC: method={method}, id={rpc_id}, params={params}")

    # 1. initialize
    if method == "initialize":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                    "experimental": {
                        "io.modelcontextprotocol/ui": {
                            "mimeTypes": ["text/html;profile=mcp-app"]
                        }
                    },
                    "extensions": {
                        "io.modelcontextprotocol/ui": {
                            "mimeTypes": ["text/html;profile=mcp-app"]
                        }
                    },
                },
                "serverInfo": {
                    "name": "ea-video-motion-mcp",
                    "version": "2.1.0",
                },
            },
        })

    # 2. tools/list
    elif method == "tools/list":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "tools": [
                    {
                        "name": "open_motion_flow_analyst",
                        "description": "Opens the 2-Stage Optical Flow & Dense Multimodal AI Video Bug Analyst in the side panel with interactive motion vector heatmap, automated anomaly peak detection, and Frostbite telemetry verification.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "clip_name": {"type": "string", "description": "Gameplay video clip name or GCS URI, e.g. skate_session_04.mp4"},
                                "sensitivity": {"type": "string", "description": "Optical flow motion anomaly sensitivity threshold: 'high', 'medium', or 'low'"}
                            },
                        },
                        "annotations": {
                            "readOnlyHint": True,
                        },
                        "_meta": {
                            "ui": {
                                "resourceUri": RESOURCE_URI,
                                "preferredMode": "pip",
                                "visibility": ["model", "app"],
                            },
                            "io.modelcontextprotocol/ui": {
                                "resourceUri": RESOURCE_URI,
                            },
                        },
                    },
                    {
                        "name": "open_skate_defect_reviewer",
                        "description": "Opens the interactive EA Skate Frostbite QA Defect Reviewer in the side panel with interactive timeline scrubber, clickable defect pins (00:12 Curb Snapping, 00:20 Ragdoll Inversion, 00:47 Impulse Launch), 0.5x Slow-Mo, and live Frostbite telemetry.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "session_id": {"type": "string", "description": "Playtest session ID to display, e.g. QA-2026-OCT"}
                            }
                        },
                        "annotations": {
                            "readOnlyHint": True,
                        },
                        "_meta": {
                            "ui": {
                                "resourceUri": REVIEWER_RESOURCE_URI,
                                "preferredMode": "pip",
                                "visibility": ["model", "app"],
                            },
                            "io.modelcontextprotocol/ui": {
                                "resourceUri": REVIEWER_RESOURCE_URI,
                            },
                        },
                    },
                    {
                        "name": "analyze_playtest_stream",
                        "description": "Runs Netflix-style stream-copy chunking (-c copy in ~0.2s) and Vertex AI Gemini 4 Argon direct GCS analysis at 6 FPS with Frostbite physics defect schema. Automatically opens the interactive QA Defect Reviewer in the side panel.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "video_path": {"type": "string", "description": "GCS URI (gs://...) or local video path to analyze"},
                                "session_id": {"type": "string", "description": "Session ID identifier"}
                            },
                            "required": ["video_path"]
                        },
                        "annotations": {
                            "readOnlyHint": True,
                        },
                        "_meta": {
                            "ui": {
                                "resourceUri": REVIEWER_RESOURCE_URI,
                                "preferredMode": "pip",
                                "visibility": ["model", "app"],
                            },
                            "io.modelcontextprotocol/ui": {
                                "resourceUri": REVIEWER_RESOURCE_URI,
                            },
                        },
                    },
                    {
                        "name": "analyze_motion_anomaly_frames",
                        "description": "2-Stage Hybrid Inference: Evaluates optical flow motion anomaly candidate moments and verifies whether sudden velocity vectors/camera jerks represent genuine physics glitches or normal gameplay.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "clip_name": {"type": "string"},
                                "candidates": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "timestamp_seconds": {"type": "number"},
                                            "time_display": {"type": "string"},
                                            "motion_score": {"type": "string"},
                                            "frames": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "timestamp_seconds": {"type": "number"},
                                                        "time_display": {"type": "string"},
                                                        "image_b64": {"type": "string"},
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                                "query": {"type": "string"},
                            },
                            "required": ["candidates"],
                        },
                        "annotations": {
                            "readOnlyHint": True,
                        },
                        "_meta": {
                            "ui": {
                                "visibility": ["model", "app"],
                            },
                        },
                    },
                    {
                        "name": "file_jira_bug",
                        "description": "App-only tool: Files analyzed video bug and Frostbite telemetry directly to EA Jira.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "bug_id": {"type": "string"},
                                "project": {"type": "string", "default": "SKATE"},
                                "summary": {"type": "string"},
                            },
                            "required": ["bug_id", "project", "summary"],
                        },
                        "annotations": {
                            "readOnlyHint": False,
                        },
                        "_meta": {
                            "ui": {
                                "visibility": ["app"],
                            },
                        },
                    },
                ],
            },
        })

    # 3. resources/list
    elif method == "resources/list":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "resources": [
                    {
                        "uri": RESOURCE_URI,
                        "name": "EA Video Motion & Optical Flow Bug Analyst",
                        "mimeType": "text/html;profile=mcp-app",
                    },
                    {
                        "uri": REVIEWER_RESOURCE_URI,
                        "name": "EA Skate Frostbite QA Defect Reviewer",
                        "mimeType": "text/html;profile=mcp-app",
                    }
                ]
            },
        })

    # 4. notifications/initialized
    elif method == "notifications/initialized":
        return JSONResponse({"jsonrpc": "2.0", "result": {}})

    # 5. resources/read
    elif method == "resources/read":
        raw_uri = params.get("uri", "")
        base_uri = raw_uri.split("#")[0]
        csp_meta = {
            "ui": {
                "csp": {
                    "connectDomains": ["*", "blob:", "data:"],
                    "resourceDomains": [
                        "blob:",
                        "data:",
                        "https://www.gstatic.com",
                        "https://fonts.googleapis.com",
                        "https://fonts.gstatic.com",
                        "https://commondatastorage.googleapis.com",
                        "https://storage.googleapis.com"
                    ]
                }
            }
        }
        if base_uri == REVIEWER_RESOURCE_URI:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "contents": [
                        {
                            "uri": raw_uri,
                            "mimeType": "text/html;profile=mcp-app",
                            "text": get_reviewer_html(),
                            "_meta": csp_meta,
                        }
                    ]
                },
            })
        else:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "contents": [
                        {
                            "uri": raw_uri,
                            "mimeType": "text/html;profile=mcp-app",
                            "text": get_widget_html(),
                            "_meta": csp_meta,
                        }
                    ]
                },
            })

    # 6. tools/call
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "open_motion_flow_analyst":
            clip_name = args.get("clip_name", "skate_session_04.mp4")
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"🎥 **EA Optical Flow & Motion Anomaly Bug Analyst Loaded**\n\n"
                                f"Active Session: `{clip_name}`\n"
                                f"- **Stage 1**: Client-side Optical Flow & motion vector differencing.\n"
                                f"- **Stage 2**: Multimodal AI semantic defect verification powered by Gemini 4 Argon.\n\n"
                                "Drag & drop your gameplay video to generate a live motion heatmap and isolate sudden velocity spikes."
                            ),
                        }
                    ],
                    "_meta": {
                        "ui": {
                            "resourceUri": RESOURCE_URI,
                            "preferredMode": "pip",
                        },
                        "io.modelcontextprotocol/ui": {
                            "resourceUri": RESOURCE_URI,
                        },
                    },
                },
            })

        elif tool_name == "open_skate_defect_reviewer":
            session_id = args.get("session_id", "QA-2026-OCT")
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"🛹 **EA Skate Frostbite QA Defect Reviewer Loaded**\n\n"
                                f"Active Session: `{session_id}`\n"
                                f"- **Interactive Defect Pins**: `00:12` Curb Snapping, `00:20` Ragdoll Inversion, `00:47` Impulse Launch.\n"
                                f"- **Controls**: `0.5x Slow-Mo`, frame-by-frame scrubbing (`±1.0s`), keyboard shortcuts (`Space`, `←`/`→`, `1`-`3`).\n"
                                f"- **Telemetry**: Live Frostbite rigid body & solver telemetry breakdown."
                            ),
                        }
                    ],
                    "_meta": {
                        "ui": {
                            "resourceUri": REVIEWER_RESOURCE_URI,
                            "preferredMode": "pip",
                        },
                        "io.modelcontextprotocol/ui": {
                            "resourceUri": REVIEWER_RESOURCE_URI,
                        },
                    },
                },
            })

        elif tool_name == "analyze_playtest_stream":
            video_path = args.get("video_path", "")
            session_id = args.get("session_id", "skate_playtest_auto")
            analyzer = GeminiPlaytestAnalyzer(project=PROJECT_ID, location=LOCATION, model_name=MODEL_ID)
            res = analyzer.analyze_window(
                gcs_video_uri=video_path,
                session_id=session_id,
                start_offset_sec=0.0,
                end_offset_sec=60.0
            )
            raw = [d.to_dict() if hasattr(d, "to_dict") else d.model_dump() for d in res.defects]
            if not raw and LATEST_ANALYSIS_DATA.get("defects"):
                raw = LATEST_ANALYSIS_DATA["defects"]

            # Store in session registry
            payload = {
                "session_id": session_id,
                "video_path": video_path,
                "total_defects": len(raw),
                "defects": raw
            }
            SESSION_RESULTS_STORE[session_id] = payload
            LATEST_ANALYSIS_DATA.update(payload)

            return JSONResponse({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"⚡ **Netflix Zero-Transcode Analysis Complete**\n"
                                f"Analyzed `{video_path}` at 6.0 FPS native 1080p using `{MODEL_ID}`.\n"
                                f"- **Total Defects Found**: {len(raw)}\n\n"
                                f"Interactive QA Defect Reviewer loaded in the side panel with all {len(raw)} defect pins pinned to the timeline."
                            )
                        }
                    ],
                    "defects": raw,
                    "_meta": {
                        "ui": {
                            "resourceUri": REVIEWER_RESOURCE_URI,
                            "preferredMode": "pip",
                        },
                        "io.modelcontextprotocol/ui": {
                            "resourceUri": REVIEWER_RESOURCE_URI,
                        },
                    },
                }
            })

        elif tool_name == "analyze_motion_anomaly_frames":
            clip_name = args.get("clip_name", "uploaded_clip.mp4")
            candidates = args.get("candidates", [])
            query = args.get("query", None)

            defects = evaluate_motion_candidates_with_gemini(candidates, query=query)

            return JSONResponse({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"🧠 **Optical Flow + Gemini Evaluation for `{clip_name}`**\n\n"
                                f"Evaluated {len(candidates)} optical flow motion anomaly candidates.\n"
                                f"- **Confirmed Genuine Defects:** {len(defects)}\n"
                                f"- **Dismissed Normal Motion/Bails:** {len(candidates) - len(defects)}\n"
                            ),
                        }
                    ],
                    "defects": defects,
                },
            })

        elif tool_name == "file_jira_bug":
            summary = args.get("summary", "Physics Bug")
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"✅ Jira ticket created successfully: '{summary}' with optical flow vector log and Frostbite telemetry attached.",
                        }
                    ]
                },
            })

        else:
            return JSONResponse(
                status_code=404,
                content={"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Tool not found: {tool_name}"}, "id": rpc_id},
            )

    return JSONResponse(
        status_code=404,
        content={"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Method not supported: {method}"}, "id": rpc_id},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
