# 🛹 EA Skate // Multimodal Video Defect Reviewer & Ingest Pipeline

An enterprise-grade, multimodal gameplay video analysis and QA triage system designed for **Electronic Arts (EA) Skate**.

Built using **Google Cloud Vertex AI (Gemini 4 Argon & Gemini 3.8 Flash)**, **Cloud Storage**, **Google Drive API**, **Cloud Scheduler**, and **FastMCP on Cloud Run**, paired with an ultra-lightweight, zero-package **HTML5 Single Page Application (SPA)** that docks natively into the **Gemini Enterprise side panel (`<ucs-mcp-pip-panel>`)** or runs in standalone browsers.

---

## 🏛️ End-to-End System Architecture

```
                                  [ EA QA Engineers / Testers ]
                                                 │
                                                 ▼
                       [ Google Drive Folder: "Skate-QA-Ingest" ]
                               (10 min – 1 hr MP4s, 100 MB – 2 GB)
                                                 │
                                                 │  Every 5 Minutes
                     ┌───────────────────────────┴───────────────────────────┐
                     │                                                       │
                     ▼                                                       ▼
       [ Cloud Scheduler Cron ]                             [ Gemini Enterprise Chat ]
       • Job: skate-drive-folder-sync                       • "Analyze easakte4min.mp4"
       • Schedule: */5 * * * *                              • "Analyze the latest video"
       • OIDC Service Account Auth                          • Pasting direct Drive/GCS link
                     │                                                       │
                     └───────────────────────────┬───────────────────────────┘
                                                 │
                                                 ▼
                               [ Cloud Run FastMCP Server ]
                              (ea-skate-mcp-app: 1 GB RAM)
                                                 │
               ┌─────────────────────────────────┴─────────────────────────────────┐
               │                                                                   │
               ▼                                                                   ▼
   [ Stream-Copy Ingest Pipeline ]                                     [ Drive Video Catalog ]
   • Queries Drive API v3 (orderBy: modifiedTime desc)                 • list_drive_playtest_videos
   • 4MB Chunked Streaming Direct into GCS                             • Instant fuzzy name matching
   • Deduplication: blob.exists() caching                              • Status: ⚡ READY / ⏳ SYNCING
               │                                                                   │
               ▼                                                                   ▼
   [ Google Cloud Storage ] (gs://ea-skate-playtests-...) ─────────────────────────┘
               │
               │  Zero-Transcode 6.0 FPS Stream Direct Passthrough
               ▼
   [ Vertex AI Gemini 4 Argon / 3.8 Flash ] (Global Endpoint)
   • 6.0 FPS native 1080p multimodal temporal grounding
   • Strict Pydantic JSON defect schema (PlaytestAnalysisResult)
   • Frostbite Physics Taxonomy:
     - Rigid Body & Mesh Collider clipping
     - Skeletal IK & Ragdoll Contortions / Joint Inversions (>360°)
     - Engine assertions (e.g. YourSelfFaceController::GetSomeAccents())
     - Havok Impulse Explosion (+14,000 N·s)
               │
               │  JSON-RPC 2.0 (preferredMode: "pip")
               ▼
   [ Interactive Reviewer SPA in Gemini Enterprise Side Panel ]
   • AppBridge postMessage Handshake (ui/request-display-mode: pip)
   • Clickable Timeline Defect Pins & Dual-Mode Scrubber
   • Sub-second frame-stepping (±1s), 0.5x Slow-Mo, and keyboard hotkeys (1, 2, 3, Space)
   • Live Frostbite telemetry diagnostics card & 1-Click Jira Export
   • Truthful 0-Defect All-Clear Display (Zero false fallback pins)
   • Toolbar Controls: ⛶ Fullscreen, ↗ Full Tab (/reviewer), 📂 Drag & Drop Dropzone
```

---

## 🚀 Key Advantages Over Traditional Approaches

| Challenge | Traditional Approach | Our Pipeline Solution |
| :--- | :--- | :--- |
| **Large Playtests (10m – 1hr)** | Loading 2 GB video files into chat hits 30s gateway proxy timeouts and crashes Chrome (OOM) | **Drive-to-GCS background streaming:** Cloud Scheduler auto-ingests in 4MB chunks; Gemini analyzes direct `gs://` URI. |
| **Manual Video Copying** | QA engineers must manually find GCS bucket paths and run shell upload commands | **Natural Language Referencing:** Simply ask `"Analyze easakte4min.mp4"` or `"Analyze the latest video in the Drive folder"`. |
| **False Positives on Fast Action** | 1 FPS Gemini Live API misses fast glitches or misidentifies normal wipeouts | Standard `generateContent` at **6.0 FPS & 1080p** provides frame-accurate temporal grounding. |
| **UI Trapped in Chat** | MCP App iframes render inside narrow chat bubbles without fullscreen | **AppBridge Handshake:** Fired on load to immediately dock into Gemini Enterprise's `<ucs-mcp-pip-panel>` side drawer. |
| **Truthful Zero-Defect Reporting** | Hardcoded sample bugs injected when clean videos return 0 defects | **Zero False Injections:** Accurately renders green `✅ Frostbite Stream Analysis Clean` all-clear badge when footage is clean. |

---

## 🛠️ Implemented Features & Components

### 1. Autonomous Google Drive Ingest & Cloud Scheduler
* **Watched Folder:** `Skate-QA-Ingest` (`1k1vdra7b2M0tYLIBM6yFNPfJa9gsS2gH`).
* **Cloud Scheduler Job:** `skate-drive-folder-sync` running every 5 minutes (`*/5 * * * *`) in `us-central1`.
* **Zero-RAM Streaming:** 4MB chunked stream transfer directly from Google Drive API into GCS (`blob.open("wb")`) with a 300-second stream deadline.
* **Instant Deduplication:** Checks GCS blob existence before downloading; cached files return in `< 50ms`.

### 2. Natural Language Playtest Resolution
Users in Gemini Enterprise do not need to know bucket paths or copy long URLs:
* **By Name:** `"Run Frostbite stream analysis on easakte4min.mp4"` (fuzzy matches against Drive catalog).
* **By Recency:** `"Analyze the latest playtest video in the Skate Drive folder"` (resolves newest by `modifiedTime`).
* **By Link:** Paste any Google Drive file link (`https://drive.google.com/file/d/...`) or folder link.
* **By GCS URI:** Direct `gs://ea-skate-playtests-...` path passthrough.

### 3. Native MCP Tools Suite
1. `analyze_playtest_stream`: Runs 6.0 FPS Frostbite stream analysis with Gemini 4 Argon and opens side panel reviewer.
2. `list_drive_playtest_videos`: Lists all videos in `Skate-QA-Ingest` with sizes, modified timestamps, and `⚡ READY` / `⏳ SYNCING` caching status.
3. `sync_drive_folder`: On-demand or scheduled Drive-to-GCS folder synchronization.
4. `open_skate_defect_reviewer`: Directly opens the interactive defect reviewer in the side panel.
5. `open_motion_flow_analyst`: Client-side optical flow motion vector analyzer.
6. `file_jira_bug`: 1-click Jira ticket filing with Frostbite telemetry and timecode.

### 4. Interactive Reviewer SPA (`skate_defect_reviewer.html`)
* **Gemini Enterprise Side-Panel Docking:** Uses the official Dave Wang MCP AppBridge postMessage handshake:
  ```javascript
  window.parent.postMessage({ jsonrpc: "2.0", id: 1, method: "ui/initialize", params: {} }, "*");
  window.parent.postMessage({ jsonrpc: "2.0", method: "ui/notifications/initialized", params: {} }, "*");
  window.parent.postMessage({ jsonrpc: "2.0", id: 2, method: "ui/request-display-mode", params: { mode: "pip" } }, "*");
  ```
* **Toolbar Controls:**
  - `⛶ Fullscreen`: Toggles fullscreen and dispatches `ui/request-display-mode: pip`.
  - `↗ Full Tab`: Direct one-click popout to `/reviewer` in a dedicated browser tab.
  - `📂 Drag & Drop`: Dropzone to inspect local files on the fly.
* **True 0-Defect State:** Clean runs display `✅ Frostbite Stream Analysis Clean: 0 Physics Defects Detected` with all nominal solver readings.
* **Playback Controls:** `Spacebar` (play/pause), `←`/`→` (±1s frame step), `0.5x Slow-Mo`, and number hotkeys (`1`, `2`, `3`).

---

## 📂 Repository Structure

```text
videoEditor/
├── frontend/
│   ├── skate_defect_reviewer.html   # Reviewer SPA frontend
│   └── motion_analyst.html          # Optical flow client-side viewer
├── backend/
│   ├── server.py                    # Cloud Run FastMCP server & Drive ingest pipeline
│   ├── templates/
│   │   ├── skate_defect_reviewer.html # Live server-rendered reviewer template
│   │   └── motion_analyst.html        # Live motion analyst template
│   ├── ingest_chunker.py            # 10s stream-copy (-c copy) chunker
│   ├── gemini_analyzer.py           # 6.0 FPS Vertex AI Gemini 4 Argon inspection engine
│   └── ea_skate_profile.py          # Frostbite physics rules & bug taxonomy
├── deploy/
│   ├── Dockerfile                   # Cloud Run container definition with FFmpeg
│   ├── deploy.sh                    # Automated GCP deploy script
│   └── requirements.txt             # Python dependencies
├── Dockerfile                       # Root container build definition
└── README.md                        # Documentation & architecture guide
```

---

## 🧪 Testing in Gemini Enterprise

1. **Check Available Playtest Videos:**
   > *"What playtest videos are in the Skate Drive folder?"*

2. **Analyze a Specific Playtest by Name:**
   > *"Run Frostbite stream analysis on easakte4min.mp4"*

3. **Analyze the Latest Upload:**
   > *"Analyze the latest video in the Skate Drive folder"*

4. **Review in Side Panel:**
   The interactive reviewer will immediately dock into the `<ucs-mcp-pip-panel>` side panel with all detected physics defect pins, Frostbite telemetry signatures, and frame scrubbing controls.
