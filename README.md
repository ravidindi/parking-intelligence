# 🚦 AI-Driven Parking Intelligence System

## Problem Statement

### Poor Visibility on Parking-Induced Congestion

On-street illegal parking and spillover parking near commercial areas,
metro stations, junctions, and events choke carriageways and
intersections.

## Operational Challenge

-   Enforcement is patrol-based and reactive.
-   No visibility into recurring obstruction hotspots.
-   No heatmap connecting parking/stationary vehicle events with
    congestion impact.
-   Difficult to prioritize enforcement zones.

## Objective

Build an AI-driven parking intelligence platform that:

-   Detects recurring stationary vehicle / illegal parking hotspots.
-   Quantifies congestion impact.
-   Predicts high-risk junctions and timings.
-   Generates enforcement recommendations.

------------------------------------------------------------------------

# Solution Architecture

Two-phase approach:

## Phase 1: Historical Parking Intelligence Engine (Current Implementation)

Uses the police parking-violation dataset:

    data/jan to may police violation_anonymized791b166.csv

Pipeline:

    Police Parking Violation Dataset
          |
          v
    Data Cleaning
          |
          v
    Feature Engineering
          |
          v
    Risk Score Generation
          |
          v
    Parking Risk Training Table
          |
          v
    Recommendation Engine

------------------------------------------------------------------------

# Dataset Understanding

Important columns:

-   latitude
-   longitude
-   location
-   vehicle_type
-   updated_vehicle_type
-   violation_type
-   offence_code
-   created_datetime
-   modified_datetime
-   validation_timestamp
-   police_station
-   junction_name
-   validation_status
-   data_sent_to_scita

Dataset exploration:

    rows: 298,450
    duplicate ids: 0
    lat/lon missing: 0

Top violations:

    WRONG PARKING          : 164,977
    NO PARKING             : 139,050
    PARKING IN A MAIN ROAD : 23,943
    PARKING ON FOOTPATH    : 3,757
    DOUBLE PARKING         : 2,037

Key modeling decision:

`junction_name` is used when present. Rows marked `No Junction` are
clustered using rounded latitude/longitude grid cells so the heatmap is
not dependent on place-name matching.

Detection-time offence codes are not taken from the dataset's internal
`offence_code` labels. The live detector assigns public Bengaluru Traffic
Police spot-fine categories from
https://btp.karnataka.gov.in/117/spot-fines/en after YOLO confirms a
stationary vehicle alert. API codes use legal references such as
`MV_ACT_177` and `190_CLAUSE_117`, plus the fine amount and public source link.

------------------------------------------------------------------------

# Completed Phase 1 Work

## 1. Obstruction Detection

Created:

    is_obstruction_event

Logic:

    violation_type contains parking-related offences

Meaning:

1 = parking violation / likely road-space obstruction\
0 = other event

The current police dataset is parking-specific, so all rows become usable
historical parking-obstruction signals.

------------------------------------------------------------------------

## 2. Feature Engineering

Generated:

### Duration

Using:

    first available action/validation/closed/modified timestamp - created_datetime

Output:

    duration_minutes

This is treated as an enforcement/clearance delay proxy, not a guaranteed
live obstruction duration.

### Parking Violation Flags

Generated:

-   wrong_parking
-   no_parking
-   main_road_parking
-   footpath_parking
-   double_parking
-   near_crossing
-   near_bus_stop
-   near_traffic_light
-   congestion_violation
-   heavy_vehicle
-   two_wheeler
-   auto_rickshaw
-   car_vehicle
-   approved_violation
-   sent_to_scita

### Time Features

Generated:

-   hour
-   day_of_week
-   month
-   peak_hour
-   parking_surge_hour
-   weekend

Peak hour:

Morning: 8 AM - 11 AM

Evening: 5 PM - 9 PM

Parking surge hour:

Late night / early morning: 12 AM - 6 AM

Evening: 7 PM - 11 PM

------------------------------------------------------------------------

## 3. Hotspot Intelligence

Aggregated data by:

-   junction
-   zone
-   hour

Generated:

-   event_frequency
-   avg_duration / enforcement delay proxy
-   congestion_violation_count
-   vehicle_impact
-   parking_surge_hour
-   vehicle mix
-   approved_count
-   unique_devices

Example:

  Junction                        Hour   Events
  ------------------------------- ------ --------
  BTP051 - Safina Plaza Junction  5      2947
  BTP051 - Safina Plaza Junction  4      2677
  BTP082 - KR Market Junction     19     1523

------------------------------------------------------------------------

## 4. Risk Score Engine

Risk score formula:

    Risk =
    42% parking violation frequency
    +
    18% congestion-specific violation types
    +
    12% vehicle impact
    +
    10% enforcement delay proxy
    +
    10% parking surge hour
    +
    5% approved violations
    +
    3% unique device coverage

Output range:

    0 - 100

Example:

    BTP051 - Safina Plaza Junction

    Risk: 86.33%

    Peak: 5 AM

    Recommendation:
    Deploy response team at 04:30

------------------------------------------------------------------------

## 5. Recommendation Engine

Rules:

High Risk:

    risk >= 70

Action:

Deploy response team before peak.

Medium Risk:

    risk >= 40

Action:

Increase monitoring.

Low Risk:

Normal patrol.

------------------------------------------------------------------------

# Machine Learning

Model:

    XGBoost Regressor trained on the new police-violation training table.

Features:

-   junction
-   zone
-   hour
-   day_of_week
-   event_frequency
-   avg_duration
-   vehicle_types
-   congestion_violation_count
-   vehicle_impact
-   parking_surge_hour
-   approved_count
-   unique_devices

Target:

    risk_score

Artifacts:

    models/police_violation/
        risk_lookup.csv
        training_features.csv
        feature_summary.json
        risk_model.pkl
        model_columns.pkl
        model_metrics.json

Old model artifacts in `models/` are no longer used by default.

Current model metrics:

    training rows: 52,495
    features: 26
    MAE: 0.043

Usage:

-   `POST /predict` uses `models/police_violation/risk_model.pkl`.
-   `POST /predict/location` uses `risk_lookup.csv` for fast latitude /
    longitude matching and returns police-station dispatch context.
-   `GET /hotspots` returns deduplicated corridor hotspots. Multiple
    hourly rows for the same junction and police station are merged into
    one corridor with `active_hours` and `records_merged`.

------------------------------------------------------------------------

# End-To-End Demo Flow

## 1. Upload Captured Camera Frame

The dashboard user selects a camera source and uploads an image frame.

Camera source policy supplies:

-   camera_id
-   zone type
-   restricted-zone flag
-   observed stationary seconds
-   alert threshold
-   offence context such as `no_parking` or `main_road`

Request:

    POST /detect/frame

What happens:

1.  FastAPI receives the uploaded JPEG/PNG/HEIC/HEIF frame.
2.  YOLO detects vehicles in the image.
3.  The stationary tracker updates state by `camera_id`.
4.  The alert rule checks:

        restricted_zone == true
        and stationary_seconds >= alert_threshold

5.  If the rule is true, the frame becomes an illegal-parking alert.
6.  The offence rules assign an official public Bengaluru Traffic Police
    category:

        MV_ACT_177       -> No Parking
        190_CLAUSE_117   -> Wrong Parking

7.  The response includes vehicle boxes, alert status, legal section,
    spot-fine amount, and source URL.

## 2. Forecast Historical Risk By Location

The dashboard also tracks the selected latitude, longitude, and hour.

Request:

    POST /predict/location

What happens:

1.  The backend searches historical parking-violation hotspots by
    latitude/longitude distance, not by place-name text.
2.  It prefers the requested hour when possible.
3.  It returns the matched hotspot, risk score, risk level, explanation,
    recommendation, police station, and dispatch station.

Example dispatch fields:

    police_station: Shivajinagar
    dispatch_station: Shivajinagar Traffic Police Station

## 3. Fuse Detection + Historical Risk

After both live detection and historical risk are available, the frontend
creates an incident decision.

Request:

    POST /enforcement/evaluate

Inputs:

-   camera policy
-   YOLO/stationary detection result
-   coordinate-based historical risk
-   matched police station

Output:

-   priority: Critical / High / Watch / Monitor
-   fused score
-   dispatch instruction
-   tactical dispatch plan:

        from_station
        target_stop
        personnel_count
        unit_count
        eta_minutes

-   recommended police station
-   congestion impact estimate
-   auditable evidence package

Example:

    Dispatch 6 personnel in 3 unit(s) from Shivajinagar Traffic Police Station
    to BTP051 - Safina Plaza Junction, Shivajinagar within 8 minutes.

The incident is persisted in:

    data/incidents.jsonl

## 4. City Risk Heatmap

Request:

    GET /hotspots

What happens:

1.  The API reads the precomputed model/risk lookup artifact.
2.  Hourly rows are merged by junction and police station.
3.  The dashboard receives unique corridors with:

        peak hour
        active hours
        event frequency
        average delay proxy
        risk score
        dispatch station

4.  The heatmap renders those corridors as map pins and a priority list.

## 5. Video / Live Relay Story

The current demo UI is optimized for uploaded still frames. For the
hackathon presentation, video should be described as a live relay that
samples frames periodically:

    CCTV / relay video
        |
        v
    capture frame every N seconds
        |
        v
    POST /detect/frame
        |
        v
    stationary tracker updates by camera_id
        |
        v
    alert when threshold is crossed

So the same image endpoint supports the live-camera design; the current
dashboard tests it with captured frames.

------------------------------------------------------------------------

# Current Structure

    parking-intelligence/

    ├── data/
    │   ├── jan to may police violation_anonymized791b166.csv
    │   └── traffic_events.csv  # legacy, no longer default
    │
    ├── notebooks/
    │   └── exploration.ipynb
    │
    ├── src/
    │   ├── preprocessing.py
    │   ├── feature_engineering.py
    │   ├── train.py
    │   └── predict.py
    │
    ├── models/
    │   └── police_violation/
    │
    └── README.md

------------------------------------------------------------------------

# Implemented Usage

Install dependencies:

    pip install -r requirements.txt

Train and refresh artifacts:

    python -m src.feature_engineering
    python -m src.train

Run a prediction from historical/model signals:

    python src/predict.py --junction SilkBoardJunc --hour 19

Start the API:

    uvicorn src.api:app --reload

Example request:

    curl -X POST http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"junction":"SilkBoardJunc","hour":19}'

Current modules:

-   preprocessing.py: CSV loading, cleaning, duration calculation,
    obstruction labels.
-   feature_engineering.py: time features, hotspot aggregation, risk
    scoring, reasons, recommendations.
-   train.py: XGBoost training, evaluation, artifact export.
-   predict.py: CLI/programmatic prediction with SHAP reasons when
    available and heuristic explanations as fallback.
-   intelligence.py: congestion impact proxy and before/after enforcement
    benefit analytics.
-   api.py: FastAPI prediction, detection, impact, incident, and health
    endpoints.

------------------------------------------------------------------------

# Phase 2 Implementation: Real-Time Detection

Implemented a first real-time detection API:

-   YOLO vehicle detection using Ultralytics.
-   OpenCV image decoding for uploaded camera frames.
-   Camera-level stationary vehicle tracking.
-   Restricted-zone illegal parking alerts.
-   Swagger-testable upload endpoint.

Endpoints:

-   `POST /detect/frame`
-   `POST /detect/reset`
-   `GET /offences`
-   `POST /impact/estimate`
-   `GET /analytics/before-after`
-   `POST /enforcement/evaluate`
-   `GET /incidents`
-   `PATCH /incidents/{incident_id}/status`

Swagger:

    http://127.0.0.1:8000/docs

To test the alert flow without downloading YOLO weights, use
`mock_detection=true` in Swagger and set:

    restricted_zone=true
    observed_seconds=360
    stationary_threshold_seconds=300

For real inference, upload a JPEG/PNG/HEIC/HEIF frame with `mock_detection=false`.
The first real inference may download the default `yolov8n.pt` model.
The detector returns the official offence category, legal section, spot-fine
amount, and source URL when a stationary restricted-zone alert is raised.

------------------------------------------------------------------------

# Dashboard

React dashboard:

Features:

-   City risk heatmap with latitude/longitude hotspot pins.
-   Camera source policy selection.
-   Still-frame vehicle detection and stationary alerting.
-   Coordinate-based historical risk intelligence.
-   Congestion impact quantification.
-   Incident workflow: open, dispatched, resolved, dismissed.
-   Before/after enforcement benefit analytics.

------------------------------------------------------------------------

# Local Run

API:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload

Dashboard:

    cd dashboard
    npm install
    npm run dev -- --host 127.0.0.1 --port 5173

Open:

    http://127.0.0.1:5173

Swagger:

    http://127.0.0.1:8000/docs

------------------------------------------------------------------------

# Deployment

## What To Commit

The deployed API runs from the compact trained artifacts in:

    models/police_violation/

Do not commit local runtime folders or the 105 MB raw police CSV. The
`.gitignore` excludes:

    .venv/
    dashboard/node_modules/
    dashboard/dist/
    data/*.csv
    data/incidents.jsonl

Keep these files in Git:

    src/
    dashboard/src/
    dashboard/package.json
    dashboard/package-lock.json
    models/police_violation/
    requirements-api.txt
    render.yaml
    netlify.toml

If this folder is not already a Git repository:

    git init
    git add .
    git commit -m "Deploy parking intelligence app"
    git branch -M main
    git remote add origin https://github.com/<you>/<repo>.git
    git push -u origin main

## Backend on Render

This repo includes `render.yaml`.

Recommended path:

1.  Push the repo to GitHub.
2.  In Render, create a new Blueprint or Web Service from the repo.
3.  If Render reads `render.yaml`, it will use the settings below.
4.  After deploy, open:

        https://your-render-service.onrender.com/health
        https://your-render-service.onrender.com/docs

Render settings if entering manually:

-   Build command:

        pip install --upgrade pip && pip install -r requirements-api.txt

-   Start command:

        uvicorn src.api:app --host 0.0.0.0 --port $PORT

-   Health check:

        /health

After creating the Render service, set:

    BACKEND_CORS_ORIGINS=https://your-netlify-site.netlify.app
    BACKEND_CORS_REGEX=https://.*\.netlify\.app
    YOLO_CONFIG_DIR=/tmp/.ultralytics
    MPLCONFIGDIR=/tmp/.mplconfig

The deploy requirements are intentionally in `requirements-api.txt` so
Render does not install notebook-only dependencies.

## Frontend on Netlify

This repo includes `netlify.toml`.

Netlify settings:

-   Base directory: `dashboard`
-   Build command: `npm ci && npm run build`
-   Publish directory: `dashboard/dist`
-   Environment variable:

        VITE_API_BASE=https://your-render-service.onrender.com

Important: `VITE_API_BASE` is baked into the frontend during build. If
you change the Render URL, update the Netlify environment variable and
redeploy the frontend.

Local dev can omit `VITE_API_BASE` because Vite proxies `/api` to the
local FastAPI server.

After Netlify deploys, copy the Netlify URL back into Render:

    BACKEND_CORS_ORIGINS=https://your-netlify-site.netlify.app

Then redeploy or restart the Render service.

Production smoke tests:

    curl https://your-render-service.onrender.com/health
    curl https://your-render-service.onrender.com/hotspots?limit=3

Open the Netlify URL and run a dashboard forecast/frame analysis.

------------------------------------------------------------------------

# Phase 2: Real-Time AI Detection

Goal:

Detect current illegal parking using cameras.

Pipeline:

    CCTV Feed
        |
        v
    YOLO Vehicle Detection
        |
        v
    Vehicle Tracking
        |
        v
    Stationary Vehicle Detection
        |
        v
    Alert Generation

Tools:

-   YOLO
-   OpenCV
-   ByteTrack / DeepSORT

Detection logic:

Vehicle stationary + long duration + restricted zone = illegal parking
alert.

------------------------------------------------------------------------

# Final Vision

Move traffic enforcement:

Reactive → Predictive → Preventive
