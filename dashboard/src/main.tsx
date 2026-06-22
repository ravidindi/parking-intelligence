import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  BarChart3,
  Camera,
  CheckCircle2,
  Clock3,
  Crosshair,
  Download,
  Gauge,
  ImageOff,
  ListChecks,
  MapPinned,
  Play,
  Radar,
  RefreshCw,
  ShieldAlert,
  Siren,
  Square,
  Video,
  Zap,
} from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

type PredictionResponse = {
  junction: string;
  zone: string | null;
  police_station: string | null;
  dispatch_station: string | null;
  hour: number;
  risk_score: number;
  risk_level: string;
  reason: string;
  recommendation: string;
  features: {
    event_frequency: number;
    avg_duration: number;
    vehicle_types: number;
    peak_hour: number;
  };
};

type LocationPredictionResponse = {
  requested_latitude: number;
  requested_longitude: number;
  matched_latitude: number | null;
  matched_longitude: number | null;
  distance_km: number | null;
  junction: string;
  zone: string;
  police_station: string;
  dispatch_station: string;
  hour: number;
  risk_score: number;
  risk_level: string;
  reason: string;
  recommendation: string;
  match_note: string;
  features: {
    event_frequency: number;
    avg_duration: number;
    vehicle_types: number;
    peak_hour: number;
  };
};

type Hotspot = {
  junction: string;
  zone: string;
  police_station: string;
  dispatch_station: string;
  hour: number;
  latitude: number | null;
  longitude: number | null;
  risk_score: number;
  risk_level: string;
  event_frequency: number;
  avg_duration: number;
  recommendation: string;
  reason: string;
  active_hours?: number[];
  records_merged?: number;
};

type Detection = {
  class_name: string;
  confidence: number;
  bbox: number[];
  centroid: number[];
  stationary_seconds: number;
  alert: boolean;
  alert_reason: string | null;
  offence_code: string | null;
  offence_label: string | null;
  offence_fine_amount: number | null;
  offence_source: string | null;
  offence_legal_section: string | null;
  offence_subtype: string | null;
};

type DetectionResponse = {
  camera_id: string;
  restricted_zone: boolean;
  vehicle_count: number;
  alert_count: number;
  detections: Detection[];
  recommendation: string;
};

type CombinedDecisionResponse = {
  incident_id: string;
  created_at: string;
  status: string;
  updated_at: string | null;
  priority: string;
  decision: string;
  fused_score: number;
  dispatch: string;
  dispatch_plan: DispatchPlan | null;
  reasons: string[];
  congestion_impact: CongestionImpactResponse | null;
  evidence: Record<string, unknown>;
};

type DispatchPlan = {
  from_station: string;
  target_stop: string;
  target_latitude: number | null;
  target_longitude: number | null;
  personnel_count: number;
  unit_count: number;
  eta_minutes: number;
  instruction: string;
  rationale: string[];
};

type CongestionImpactResponse = {
  impact_score: number;
  severity: string;
  flow_state: string;
  lane_capacity_loss_pct: number;
  throughput_loss_pct: number;
  estimated_delay_minutes: number;
  queue_risk_meters: number;
  vehicle_count: number;
  alert_count: number;
  action: string;
  assumptions: string[];
};

type CorridorBenefit = {
  junction: string;
  zone: string;
  hour: number;
  risk_before: number;
  risk_after: number;
  delay_before_minutes: number;
  delay_after_minutes: number;
  delay_saved_minutes: number;
  expected_reduction_pct: number;
  action: string;
};

type BeforeAfterResponse = {
  summary: {
    hotspots_analyzed: number;
    delay_before_minutes: number;
    delay_after_minutes: number;
    delay_saved_minutes: number;
    risk_reduction_pct: number;
    patrol_hours_saved: number;
    average_risk_before: number;
    average_risk_after: number;
  };
  corridors: CorridorBenefit[];
};

const cameraSources = [
  {
    id: "kr-market",
    label: "KR Market",
    cameraId: "cam-kr-market-01",
    zoneType: "Restricted commercial corridor",
    enforcementLevel: "High",
    restrictedZone: true,
    observedSeconds: 210,
    alertThreshold: 90,
    offenceContext: "main_road",
    note: "Loading bays and bus movement make short stops risky.",
  },
  {
    id: "koramangala",
    label: "Koramangala",
    cameraId: "cam-koramangala-05",
    zoneType: "Mixed / mid zone",
    enforcementLevel: "Medium",
    restrictedZone: true,
    observedSeconds: 270,
    alertThreshold: 240,
    offenceContext: "no_parking",
    note: "Mixed retail and residential traffic gets a moderate grace window.",
  },
  {
    id: "indiranagar-residential",
    label: "Indiranagar Residential",
    cameraId: "cam-indiranagar-res-02",
    zoneType: "Residential lane",
    enforcementLevel: "Low",
    restrictedZone: false,
    observedSeconds: 360,
    alertThreshold: 600,
    offenceContext: "wrong_parking",
    note: "Residential streets tolerate longer stationary time before escalation.",
  },
];

function App() {
  const [latitude, setLatitude] = useState(12.9177);
  const [longitude, setLongitude] = useState(77.6239);
  const [hour, setHour] = useState(19);
  const [prediction, setPrediction] = useState<LocationPredictionResponse | null>(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState("");
  const [hotspots, setHotspots] = useState<Hotspot[]>([]);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedFileLabel, setSelectedFileLabel] = useState("");
  const [previewUrl, setPreviewUrl] = useState("");
  const [previewError, setPreviewError] = useState("");
  const [testVideoFile, setTestVideoFile] = useState<File | null>(null);
  const [testVideoLabel, setTestVideoLabel] = useState("");
  const [testVideoUrl, setTestVideoUrl] = useState("");
  const [testVideoError, setTestVideoError] = useState("");
  const [autoScanning, setAutoScanning] = useState(false);
  const [autoScanCount, setAutoScanCount] = useState(0);
  const [lastAutoScanTime, setLastAutoScanTime] = useState("");
  const [autoScanError, setAutoScanError] = useState("");
  const [detection, setDetection] = useState<DetectionResponse | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [detectionError, setDetectionError] = useState("");
  const [combinedDecision, setCombinedDecision] = useState<CombinedDecisionResponse | null>(null);
  const [combinedLoading, setCombinedLoading] = useState(false);
  const [combinedError, setCombinedError] = useState("");
  const [congestionImpact, setCongestionImpact] = useState<CongestionImpactResponse | null>(null);
  const [analytics, setAnalytics] = useState<BeforeAfterResponse | null>(null);
  const [analyticsError, setAnalyticsError] = useState("");
  const [incidents, setIncidents] = useState<CombinedDecisionResponse[]>([]);
  const [incidentError, setIncidentError] = useState("");
  const [incidentUpdating, setIncidentUpdating] = useState("");
  const [imageSize, setImageSize] = useState({ width: 1536, height: 1024 });
  const [selectedCameraId, setSelectedCameraId] = useState(cameraSources[0].id);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const testVideoRef = useRef<HTMLVideoElement | null>(null);
  const autoScanTimerRef = useRef<number | null>(null);
  const autoScanCountRef = useRef(0);
  const historicalPanelRef = useRef<HTMLDivElement | null>(null);

  const currentCamera = cameraSources.find((camera) => camera.id === selectedCameraId) ?? cameraSources[0];
  const primaryAlert = detection?.detections.find((item) => item.alert);
  const maxStationarySeconds = Math.round(detection?.detections.reduce((max, item) => Math.max(max, item.stationary_seconds ?? 0), 0) ?? 0);
  const decisionEvents = [
    `Frame received from ${currentCamera.label}`,
    "Vehicle classified by YOLO pipeline",
    `${currentCamera.zoneType} policy applied`,
    `${currentCamera.alertThreshold}s threshold evaluated`,
    "Dispatch recommendation generated",
  ];
  const riskScore = prediction?.risk_score ?? 0;
  const cityReadiness = Math.round(
    ((prediction ? Math.min(100, prediction.features.event_frequency * 18 + 35) : 45) +
      (detection ? Math.min(100, detection.vehicle_count * 38 + detection.alert_count * 42) : 50)) /
      2,
  );

  useEffect(() => {
    loadHotspots();
    loadAnalytics();
    loadIncidents();
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    return () => {
      if (testVideoUrl) URL.revokeObjectURL(testVideoUrl);
    };
  }, [testVideoUrl]);

  useEffect(() => {
    return () => stopAutoScan(false);
  }, []);

  async function loadHotspots() {
    try {
      const response = await fetch(`${API_BASE}/hotspots?limit=24`);
      if (!response.ok) throw new Error(`Hotspots failed with ${response.status}`);
      const rows = (await response.json()) as Hotspot[];
      setHotspots(rows);
      const first = rows.find((row) => row.latitude !== null && row.longitude !== null);
      if (first?.latitude !== null && first?.longitude !== null && first?.latitude !== undefined && first?.longitude !== undefined) {
        setLatitude(first.latitude);
        setLongitude(first.longitude);
        setHour(first.hour);
        await runPrediction({ latitude: first.latitude, longitude: first.longitude, hour: first.hour });
      }
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : "Unable to load hotspots");
      await runPrediction();
    }
  }

  async function loadAnalytics() {
    setAnalyticsError("");
    try {
      const response = await fetch(`${API_BASE}/analytics/before-after?limit=12`);
      if (!response.ok) throw new Error(`Before/after analytics failed with ${response.status}`);
      setAnalytics((await response.json()) as BeforeAfterResponse);
    } catch (error) {
      setAnalyticsError(error instanceof Error ? error.message : "Unable to load before/after analytics");
    }
  }

  async function loadIncidents() {
    setIncidentError("");
    try {
      const response = await fetch(`${API_BASE}/incidents?limit=8`);
      if (!response.ok) throw new Error(`Incidents failed with ${response.status}`);
      setIncidents((await response.json()) as CombinedDecisionResponse[]);
    } catch (error) {
      setIncidentError(error instanceof Error ? error.message : "Unable to load incidents");
    }
  }

  async function runPrediction(next?: { latitude?: number; longitude?: number; hour?: number }) {
    const payload = {
      latitude: next?.latitude ?? latitude,
      longitude: next?.longitude ?? longitude,
      hour: next?.hour ?? hour,
      radius_km: 2,
    };

    setPredictionLoading(true);
    setPredictionError("");
    try {
      const response = await fetch(`${API_BASE}/predict/location`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(`Prediction failed with ${response.status}`);
      const riskPayload = (await response.json()) as LocationPredictionResponse;
      setPrediction(riskPayload);
      if (detection) void evaluateIncident(detection, riskPayload, selectedFileLabel || null);
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : "Prediction failed");
    } finally {
      setPredictionLoading(false);
    }
  }

  async function runDetection(fileOverride?: File, options?: { observedSeconds?: number; frameName?: string }) {
    const frame = fileOverride ?? selectedFile;
    if (!frame) {
      setDetectionError("Upload a camera frame first.");
      return;
    }
    if (!(frame instanceof File)) {
      setDetectionError("Selected frame is not a valid image file.");
      return;
    }

    const form = new FormData();
    form.append("image", frame, frame.name || "camera-frame.png");
    form.append("camera_id", currentCamera.cameraId);
    form.append("restricted_zone", String(currentCamera.restrictedZone));
    form.append("stationary_threshold_seconds", String(currentCamera.alertThreshold));
    form.append("observed_seconds", String(options?.observedSeconds ?? currentCamera.observedSeconds));
    form.append("offence_context", currentCamera.offenceContext);
    form.append("mock_detection", "false");

    setDetecting(true);
    setDetectionError("");
    try {
      const response = await fetch(`${API_BASE}/detect/frame`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(formatApiError(detail, `Detection failed with ${response.status}`));
      }
      const detectionPayload = (await response.json()) as DetectionResponse;
      setDetection(detectionPayload);
      if (prediction) await evaluateIncident(detectionPayload, prediction, options?.frameName ?? frame.name ?? selectedFileLabel ?? null);
    } catch (error) {
      setDetectionError(error instanceof Error ? error.message : "Detection failed");
    } finally {
      setDetecting(false);
    }
  }

  function chooseHotspot(item: Hotspot) {
    if (item.latitude === null || item.longitude === null) return;
    setLatitude(item.latitude);
    setLongitude(item.longitude);
    setHour(item.hour);
    runPrediction({ latitude: item.latitude, longitude: item.longitude, hour: item.hour });
    window.setTimeout(() => historicalPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
  }

  function chooseCityCorridor(item: Hotspot) {
    if (item.latitude === null || item.longitude === null) return;
    setLatitude(item.latitude);
    setLongitude(item.longitude);
    setHour(item.hour);
    runPrediction({ latitude: item.latitude, longitude: item.longitude, hour: item.hour });
  }

  function chooseCamera(cameraId: string) {
    setSelectedCameraId(cameraId);
    setDetection(null);
    setDetectionError("");
    setCombinedDecision(null);
    setCombinedError("");
    setCongestionImpact(null);
    stopAutoScan();
  }

  function chooseMapLocation(nextLatitude: number, nextLongitude: number) {
    setLatitude(Number(nextLatitude.toFixed(6)));
    setLongitude(Number(nextLongitude.toFixed(6)));
    runPrediction({ latitude: nextLatitude, longitude: nextLongitude, hour });
  }

  async function prepareUploadFile(file: File) {
    const extension = file.name.split(".").pop()?.toLowerCase();
    const looksLikeHeic = ["heic", "heif"].includes(extension ?? "") || ["image/heic", "image/heif"].includes(file.type);
    if (!looksLikeHeic) return file;

    const heic2any = (await import("heic2any")).default;
    const converted = await heic2any({
      blob: file,
      toType: "image/jpeg",
      quality: 0.92,
    });
    const blob = Array.isArray(converted) ? converted[0] : converted;
    return new File([blob], file.name.replace(/\.(heic|heif)$/i, ".jpg"), { type: "image/jpeg" });
  }

  async function onFileChange(file: File | null) {
    setPreviewError("");
    setSelectedFile(file);
    setSelectedFileLabel(file ? file.name : "");
    setDetection(null);
    setDetectionError("");
    setCombinedDecision(null);
    setCombinedError("");
    setCongestionImpact(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl("");

    if (!file) return;

    try {
      const prepared = await prepareUploadFile(file);
      setSelectedFile(prepared);
      setSelectedFileLabel(prepared.name === file.name ? file.name : `${file.name} converted to JPEG`);
      setPreviewUrl(URL.createObjectURL(prepared));
    } catch (error) {
      setSelectedFile(file);
      setPreviewError(error instanceof Error ? error.message : "Preview is unavailable for this file.");
      setSelectedFileLabel(file.name);
    }
  }

  function onTestVideoChange(file: File | null) {
    stopAutoScan();
    setAutoScanError("");
    setTestVideoError("");
    setAutoScanCount(0);
    autoScanCountRef.current = 0;
    setLastAutoScanTime("");
    setTestVideoFile(file);
    setTestVideoLabel(file ? file.name : "");
    if (testVideoUrl) URL.revokeObjectURL(testVideoUrl);
    setTestVideoUrl(file ? URL.createObjectURL(file) : "");
  }

  async function captureTestVideoFrame() {
    const video = testVideoRef.current;
    if (!video || video.readyState < 2) {
      throw new Error("Test video is still loading. Wait until the preview is visible.");
    }
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) throw new Error("Test video frame size is unavailable.");

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Unable to capture test video frame.");
    context.drawImage(video, 0, 0, width, height);
    setImageSize({ width, height });
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    if (!blob) throw new Error("Unable to encode test video frame.");
    const safeName = (testVideoFile?.name || "auto-scan-test").replace(/\.[^.]+$/, "");
    return new File([blob], `${safeName}-scan-${autoScanCountRef.current + 1}.jpg`, { type: "image/jpeg" });
  }

  async function runAutoScanOnce() {
    setAutoScanError("");
    try {
      const frame = await captureTestVideoFrame();
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setSelectedFile(frame);
      setSelectedFileLabel(frame.name);
      setPreviewUrl(URL.createObjectURL(frame));
      const videoSeconds = Math.round((testVideoRef.current?.currentTime ?? 0) * 30);
      const observedSeconds = Math.max(videoSeconds, autoScanCountRef.current * 45);
      autoScanCountRef.current += 1;
      setAutoScanCount(autoScanCountRef.current);
      setLastAutoScanTime(new Date().toLocaleTimeString());
      await runDetection(frame, {
        observedSeconds,
        frameName: `${testVideoLabel || "test-video"} auto-scan frame ${autoScanCountRef.current}`,
      });
    } catch (error) {
      setAutoScanError(error instanceof Error ? error.message : "Auto scan failed");
      stopAutoScan();
    }
  }

  function startAutoScan() {
    if (!testVideoFile) {
      setAutoScanError("Upload a test video first.");
      return;
    }
    if (autoScanTimerRef.current !== null) return;
    setAutoScanError("");
    setAutoScanning(true);
    void testVideoRef.current?.play().catch(() => undefined);
    void runAutoScanOnce();
    autoScanTimerRef.current = window.setInterval(() => {
      void runAutoScanOnce();
    }, 5000);
  }

  function stopAutoScan(updateState = true) {
    if (autoScanTimerRef.current !== null) {
      window.clearInterval(autoScanTimerRef.current);
      autoScanTimerRef.current = null;
    }
    if (updateState) setAutoScanning(false);
  }

  async function evaluateIncident(nextDetection: DetectionResponse, nextPrediction: LocationPredictionResponse, frameName: string | null) {
    setCombinedLoading(true);
    setCombinedError("");
    try {
      const response = await fetch(`${API_BASE}/enforcement/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          camera: {
            camera_id: currentCamera.cameraId,
            label: currentCamera.label,
            zone_type: currentCamera.zoneType,
            enforcement_level: currentCamera.enforcementLevel,
            restricted_zone: currentCamera.restrictedZone,
            observed_seconds: nextDetection.detections.reduce((max, item) => Math.max(max, item.stationary_seconds ?? 0), 0),
            alert_threshold: currentCamera.alertThreshold,
            offence_context: currentCamera.offenceContext,
          },
          detection: nextDetection,
          risk: nextPrediction,
          frame_name: frameName,
        }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(formatApiError(detail, `Incident evaluation failed with ${response.status}`));
      }
      const decisionPayload = (await response.json()) as CombinedDecisionResponse;
      setCombinedDecision(decisionPayload);
      setCongestionImpact(decisionPayload.congestion_impact);
      void loadIncidents();
    } catch (error) {
      setCombinedError(error instanceof Error ? error.message : "Incident evaluation failed");
    } finally {
      setCombinedLoading(false);
    }
  }

  async function updateIncidentStatus(incidentId: string, status: string) {
    setIncidentUpdating(`${incidentId}:${status}`);
    setIncidentError("");
    try {
      const response = await fetch(`${API_BASE}/incidents/${incidentId}/status`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(formatApiError(detail, `Incident status failed with ${response.status}`));
      }
      const updated = (await response.json()) as CombinedDecisionResponse;
      setIncidents((rows) => rows.map((incident) => (incident.incident_id === updated.incident_id ? updated : incident)));
      if (combinedDecision?.incident_id === updated.incident_id) setCombinedDecision(updated);
    } catch (error) {
      setIncidentError(error instanceof Error ? error.message : "Incident status update failed");
    } finally {
      setIncidentUpdating("");
    }
  }

  function exportEvidence() {
    if (!combinedDecision) return;
    const blob = new Blob([JSON.stringify(combinedDecision.evidence, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${combinedDecision.incident_id}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  const imageOverlay = useMemo(() => {
    if (!detection || !previewUrl || detection.detections.length === 0) return null;
    return detection.detections.map((item, index) => {
      const [x1, y1, x2, y2] = item.bbox;
      const left = `${(x1 / imageSize.width) * 100}%`;
      const top = `${(y1 / imageSize.height) * 100}%`;
      const width = `${((x2 - x1) / imageSize.width) * 100}%`;
      const height = `${((y2 - y1) / imageSize.height) * 100}%`;
      return (
        <div
          key={`${item.class_name}-${index}`}
          className={`absolute border-2 ${item.alert ? "border-red-500" : "border-emerald-400"} bg-black/10`}
          style={{ left, top, width, height }}
        >
          <span className={`absolute -top-7 left-0 whitespace-nowrap px-2 py-1 text-xs font-semibold text-white ${item.alert ? "bg-red-600" : "bg-emerald-600"}`}>
            {item.class_name} {(item.confidence * 100).toFixed(0)}%
          </span>
        </div>
      );
    });
  }, [detection, previewUrl, imageSize]);
  const mediaFrameStyle: React.CSSProperties =
    imageSize.height >= imageSize.width
      ? {
          aspectRatio: `${imageSize.width} / ${imageSize.height}`,
          height: "100%",
        }
      : {
          aspectRatio: `${imageSize.width} / ${imageSize.height}`,
          width: "100%",
        };

  return (
    <main className="min-h-screen bg-[#edf1f4] text-[#10151f]">
      <div className="mx-auto flex max-w-[1500px] flex-col gap-5 px-4 py-4 lg:px-6">
        <header className="border border-[#d5dde6] bg-white px-5 py-4 shadow-panel">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.24em] text-[#0f766e]">Bengaluru Mobility AI</p>
              <h1 className="mt-1 text-2xl font-black tracking-normal text-[#111827] md:text-4xl">
                Parking Intelligence Command Center
              </h1>
            </div>
            <div className="grid grid-cols-3 gap-2 text-sm">
              <Metric icon={<Radar size={18} />} label="City Readiness" value={`${cityReadiness}%`} tone="teal" />
              <Metric icon={<Siren size={18} />} label="Active Alerts" value={String(detection?.alert_count ?? 0)} tone="red" />
              <Metric icon={<Clock3 size={18} />} label="Response Window" value={primaryAlert ? "Now" : "Watch"} tone="amber" />
            </div>
          </div>
        </header>

        <Panel title="City Risk Heatmap" icon={<MapPinned size={20} />} action="Operational overview">
          <CityRiskHeatmapHeader prediction={prediction} primaryAlert={Boolean(primaryAlert)} activeHotspots={hotspots.filter((item) => item.risk_score >= 60).length} />
          <CityRiskHeatmap hotspots={hotspots} prediction={prediction} primaryAlert={Boolean(primaryAlert)} onHotspotSelect={chooseCityCorridor} />
        </Panel>

        <section className="grid items-stretch gap-5 xl:grid-cols-[2fr_1.25fr]">
          <div className="border-2 border-[#0f766e] bg-[#f8fffd] p-4 shadow-panel">
            <div className="mb-4 flex flex-col gap-2 border-b border-[#99f6e4] pb-4 md:flex-row md:items-end md:justify-between">
              <div>
                <p className="text-xs font-black uppercase tracking-[0.22em] text-[#0f766e]">Camera enforcement stack</p>
                <h2 className="mt-1 text-2xl font-black text-[#111827]">Camera Feed + Illegal Parking Verdict</h2>
              </div>
              <p className="text-sm font-bold text-[#475569]">Demo uses captured camera frames; production connects the same pipeline to live relay frames.</p>
            </div>
            <div className="grid gap-4 lg:grid-cols-2">
              <Panel title="Live Camera Frame" icon={<Camera size={20} />} action="YOLO + stationary tracking" className="h-[860px]">
                <div className="h-full space-y-4 overflow-y-auto pr-1 command-scroll">
                  <label className="flex min-h-[340px] cursor-pointer flex-col items-center justify-center border-2 border-dashed border-[#98a9ba] bg-[#f8fafc] p-4 text-center transition hover:border-[#0f766e] hover:bg-white">
                  {previewUrl ? (
                    <div className="relative flex h-[330px] w-full items-center justify-center overflow-hidden bg-black">
                      <div className="relative max-h-full max-w-full" style={mediaFrameStyle}>
                        <img
                          key={previewUrl}
                          ref={imageRef}
                          src={previewUrl}
                          alt="Uploaded camera frame"
                          className="h-full w-full object-fill"
                          onLoad={(event) =>
                            setImageSize({
                              width: event.currentTarget.naturalWidth || 1536,
                              height: event.currentTarget.naturalHeight || 1024,
                            })
                          }
                          onError={() => setPreviewError("Preview is unavailable, but the file can still be analyzed.")}
                        />
                        {imageOverlay}
                      </div>
                    </div>
                  ) : previewError ? (
                    <div className="flex flex-col items-center gap-3">
                      <ImageOff className="text-[#0f766e]" size={34} />
                      <div>
                        <p className="font-bold">Preview unavailable</p>
                        <p className="mt-1 text-sm text-[#617186]">{previewError}</p>
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center gap-3">
                      <Camera className="text-[#0f766e]" size={34} />
                      <div>
                        <p className="font-bold">Upload captured camera frame</p>
                        <p className="mt-1 text-sm text-[#617186]">Use a still frame from the selected camera source.</p>
                      </div>
                    </div>
                  )}
                  <input
                    className="hidden"
                    type="file"
                    accept="image/png,image/jpeg,image/jpg,image/heic,image/heif,.png,.jpg,.jpeg,.heic,.heif"
                    onChange={(event) => {
                      const file = event.target.files?.[0] ?? null;
                      void onFileChange(file);
                      event.currentTarget.value = "";
                    }}
                  />
                </label>
                <div className="border border-[#d5dde6] bg-white px-3 py-2 text-sm font-bold text-[#334155]">
                  Selected frame: <span className="text-[#0f766e]">{selectedFileLabel || "None"}</span>
                </div>

                <label className="space-y-1">
                  <span className="text-xs font-black uppercase tracking-[0.16em] text-[#617186]">Camera source</span>
                  <select value={selectedCameraId} onChange={(event) => chooseCamera(event.target.value)} className="input">
                    {cameraSources.map((camera) => (
                      <option key={camera.id} value={camera.id}>
                        {camera.label}
                      </option>
                    ))}
                  </select>
                </label>

                <div className="grid gap-3 md:grid-cols-3">
                  <PolicyStat label="Zone type" value={currentCamera.zoneType} />
                  <PolicyStat
                    label="Observed seconds"
                    value={`${currentCamera.observedSeconds}s`}
                  />
                  <PolicyStat label="Alert threshold" value={`${currentCamera.alertThreshold}s`} />
                </div>

                <div className={`border p-3 text-sm font-semibold ${currentCamera.enforcementLevel === "High" ? "border-red-200 bg-red-50 text-red-800" : currentCamera.enforcementLevel === "Medium" ? "border-amber-200 bg-amber-50 text-amber-900" : "border-teal-200 bg-teal-50 text-teal-800"}`}>
                  {currentCamera.enforcementLevel} enforcement · {currentCamera.restrictedZone ? "restricted rule active" : "monitor-only residential rule"}
                  <p className="mt-1 text-xs leading-5 opacity-80">{currentCamera.note}</p>
                </div>

                <button
                  onClick={() => runDetection()}
                  disabled={detecting || !selectedFile}
                  className="inline-flex w-full items-center justify-center gap-2 bg-[#0f766e] px-4 py-3 text-sm font-black uppercase tracking-[0.16em] text-white transition hover:bg-[#115e59] disabled:cursor-wait disabled:opacity-60"
                >
                  {detecting ? <RefreshCw className="animate-spin" size={18} /> : <Play size={18} />}
                  Analyze Frame
                </button>
                  {detectionError && <p className="border border-red-200 bg-red-50 p-3 text-sm font-semibold text-red-700">{detectionError}</p>}

                <div className="border border-[#d5dde6] bg-[#f8fafc] p-3">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-xs font-black uppercase tracking-[0.16em] text-[#0f766e]">Auto scan test video</p>
                      <p className="mt-1 text-xs font-semibold text-[#617186]">Demo harness: samples one frame every 5s to simulate live relay scanning.</p>
                    </div>
                    <Video className="shrink-0 text-[#0f766e]" size={22} />
                  </div>

                  {testVideoUrl && (
                    <>
                      <video
                        ref={testVideoRef}
                        src={testVideoUrl}
                        className="mt-3 h-[150px] w-full bg-black object-contain"
                        controls
                        muted
                        playsInline
                        loop
                        preload="metadata"
                        onError={() => {
                          setTestVideoError("Browser preview could not decode this clip. Try an H.264 MP4 or WebM test video.");
                        }}
                        onLoadedMetadata={() => setTestVideoError("")}
                      />
                      {testVideoError && (
                        <p className="mt-2 border border-amber-200 bg-amber-50 p-2 text-xs font-semibold text-amber-900">
                          {testVideoError}
                        </p>
                      )}
                    </>
                  )}

                  <label className="mt-3 flex cursor-pointer items-center justify-center border border-[#cbd5e1] bg-white px-3 py-2 text-xs font-black uppercase tracking-[0.14em] text-[#0f766e] transition hover:border-[#0f766e]">
                    {testVideoLabel || "Upload test video"}
                    <input
                      className="hidden"
                      type="file"
                      accept="video/mp4,video/quicktime,video/webm,.mp4,.mov,.m4v,.webm"
                      onChange={(event) => {
                        const file = event.target.files?.[0] ?? null;
                        onTestVideoChange(file);
                        event.currentTarget.value = "";
                      }}
                    />
                  </label>

                  <div className="mt-3 grid grid-cols-3 gap-2">
                    <PolicyStat label="Scans" value={String(autoScanCount)} />
                    <PolicyStat label="Interval" value="5s" />
                    <PolicyStat label="Last scan" value={lastAutoScanTime || "None"} />
                  </div>

                  <div className="mt-3 grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={startAutoScan}
                      disabled={!testVideoFile || autoScanning}
                      className="inline-flex items-center justify-center gap-2 bg-[#111827] px-3 py-2 text-xs font-black uppercase tracking-[0.14em] text-white transition hover:bg-[#263244] disabled:opacity-50"
                    >
                      <Play size={16} />
                      Start auto scan
                    </button>
                    <button
                      type="button"
                      onClick={() => stopAutoScan()}
                      disabled={!autoScanning}
                      className="inline-flex items-center justify-center gap-2 border border-[#111827] bg-white px-3 py-2 text-xs font-black uppercase tracking-[0.14em] text-[#111827] transition hover:bg-[#e5e7eb] disabled:opacity-50"
                    >
                      <Square size={14} />
                      Stop
                    </button>
                  </div>
                  {autoScanError && <p className="mt-3 border border-red-200 bg-red-50 p-2 text-xs font-semibold text-red-700">{autoScanError}</p>}
                </div>
                </div>
              </Panel>

              <Panel title="Illegal Parking Verdict" icon={<ShieldAlert size={20} />} action={detection ? `${detection.vehicle_count} vehicles scanned` : "Awaiting frame"} className="h-[860px]">
                <div className="flex h-full flex-col gap-4">
                <div className={`border px-5 py-6 ${primaryAlert ? "border-red-300 bg-red-50" : "border-emerald-200 bg-emerald-50"}`}>
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-xs font-black uppercase tracking-[0.24em] text-[#617186]">Detection Status</p>
                      <h2 className={`mt-3 text-4xl font-black ${primaryAlert ? "text-red-700" : "text-emerald-700"}`}>
                        {primaryAlert ? "Illegal Parking" : detection ? "No Alert" : "Ready"}
                      </h2>
                    </div>
                    {primaryAlert ? <AlertTriangle className="text-red-600" size={34} /> : <CheckCircle2 className="text-emerald-600" size={34} />}
                  </div>
                  <p className="mt-4 text-sm leading-6 text-[#334155]">
                    {detection?.recommendation ?? "Upload a captured camera frame and run analysis to generate an enforcement action."}
                  </p>
                </div>

                <div className="grid grid-cols-3 gap-3">
                  <CompactStat label="Vehicles" value={detection?.vehicle_count ?? 0} />
                  <CompactStat label="Alerts" value={detection?.alert_count ?? 0} />
                  <CompactStat label="Stationary" value={`${maxStationarySeconds}s`} />
                </div>

                {primaryAlert?.offence_code && (
                  <div className="border border-red-200 bg-red-50 p-4 text-red-800">
                    <p className="text-xs font-black uppercase tracking-[0.18em] opacity-75">Official BTP offence</p>
                    <p className="mt-2 text-2xl font-black">
                      {primaryAlert.offence_label}
                    </p>
                    {primaryAlert.offence_legal_section && (
                      <p className="mt-1 text-sm font-black uppercase tracking-[0.12em] opacity-85">{primaryAlert.offence_legal_section}</p>
                    )}
                    {!primaryAlert.offence_legal_section && <p className="mt-1 break-words text-sm font-bold">{primaryAlert.offence_code}</p>}
                    {primaryAlert.offence_subtype && (
                      <p className="mt-2 text-sm font-semibold">Detected subtype: {primaryAlert.offence_subtype}</p>
                    )}
                    {primaryAlert.offence_fine_amount && (
                      <p className="mt-1 text-sm font-bold">Spot fine: INR {primaryAlert.offence_fine_amount}</p>
                    )}
                    {primaryAlert.offence_source && (
                      <a className="mt-2 inline-block text-xs font-black uppercase tracking-[0.14em] underline" href={primaryAlert.offence_source} target="_blank" rel="noreferrer">
                        Source: Bengaluru Traffic Police
                      </a>
                    )}
                  </div>
                )}

                <CombinedDecisionCard decision={combinedDecision} loading={combinedLoading} error={combinedError} onExport={exportEvidence} />

                <div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1 command-scroll">
                  <p className="text-xs font-black uppercase tracking-[0.18em] text-[#617186]">Decision trail</p>
                  {decisionEvents.map((event, index) => (
                    <div key={event} className="flex items-center gap-3 border border-[#d5dde6] bg-white px-3 py-2">
                      <span className={`h-2.5 w-2.5 ${index <= (primaryAlert ? 4 : 1) ? "bg-[#0f766e]" : "bg-[#cbd5e1]"}`} />
                      <span className="text-sm font-semibold text-[#334155]">{event}</span>
                    </div>
                  ))}
                </div>
                </div>
              </Panel>
            </div>
          </div>

          <div ref={historicalPanelRef} className="min-h-[860px] self-stretch">
          <Panel title="Historical Risk Intelligence" icon={<Gauge size={20} />} action="Coordinate hotspot matching" className="h-full">
            <div className="h-full space-y-5 overflow-y-auto pr-1 command-scroll">
              <div className="grid gap-3 md:grid-cols-[1fr_1fr_0.5fr]">
                <label className="space-y-1">
                  <span className="text-xs font-black uppercase tracking-[0.18em] text-[#617186]">Latitude</span>
                  <input type="number" step="0.000001" value={latitude} onChange={(event) => setLatitude(Number(event.target.value))} className="input" />
                </label>
                <label className="space-y-1">
                  <span className="text-xs font-black uppercase tracking-[0.18em] text-[#617186]">Longitude</span>
                  <input type="number" step="0.000001" value={longitude} onChange={(event) => setLongitude(Number(event.target.value))} className="input" />
                </label>
                <label className="space-y-1">
                  <span className="text-xs font-black uppercase tracking-[0.18em] text-[#617186]">Hour</span>
                  <input type="number" min={0} max={23} value={hour} onChange={(event) => setHour(Number(event.target.value))} className="input" />
                </label>
              </div>
              <div className="border border-[#d5dde6] bg-[#f8fafc] p-3 text-sm font-bold text-[#334155]">
                Matched hotspot: <span className="text-[#0f766e]">{prediction ? `${prediction.junction}, ${prediction.zone}` : "Loading dataset hotspots..."}</span>
                {prediction?.distance_km !== null && prediction?.distance_km !== undefined ? ` (${prediction.distance_km} km)` : ""}
              </div>
              <LocationMap
                latitude={latitude}
                longitude={longitude}
                matchedLatitude={prediction?.matched_latitude ?? null}
                matchedLongitude={prediction?.matched_longitude ?? null}
                hotspots={hotspots}
                onSelect={chooseMapLocation}
                onHotspotSelect={chooseHotspot}
              />
              <button
                onClick={() => runPrediction()}
                disabled={predictionLoading}
                className="inline-flex w-full items-center justify-center gap-2 bg-[#111827] px-4 py-3 text-sm font-black uppercase tracking-[0.16em] text-white transition hover:bg-[#263244] disabled:cursor-wait disabled:opacity-60"
              >
                {predictionLoading ? <RefreshCw className="animate-spin" size={18} /> : <Zap size={18} />}
                Forecast Obstruction Risk
              </button>
              {predictionError && <p className="border border-red-200 bg-red-50 p-3 text-sm font-semibold text-red-700">{predictionError}</p>}

              <div className="grid items-start gap-4 md:grid-cols-[0.72fr_1fr]">
                <RiskGauge score={riskScore} level={prediction?.risk_level ?? "Idle"} />
                <div className="max-h-[300px] space-y-3 overflow-y-auto pr-1 command-scroll">
                  <p className="text-xs font-black uppercase tracking-[0.18em] text-[#617186]">Top hotspots</p>
                  {hotspots.map((item) => (
                    <button
                      key={`${item.junction}-${item.zone}-${item.hour}-${item.latitude}-${item.longitude}`}
                      onClick={() => chooseHotspot(item)}
                      className="flex w-full items-center justify-between border border-[#d5dde6] bg-white px-3 py-3 text-left transition hover:border-[#0f766e]"
                    >
                      <span>
                        <span className="block max-w-[260px] truncate text-sm font-black">{item.junction}</span>
                        <span className="block text-xs font-semibold text-[#617186]">
                          {item.police_station ?? item.zone} · {hotspotHourSummary(item)} · {item.latitude?.toFixed(4)}, {item.longitude?.toFixed(4)}
                        </span>
                      </span>
                      <ArrowUpRight size={18} className="text-[#0f766e]" />
                    </button>
                  ))}
                </div>
              </div>

              <div className="border border-[#d5dde6] bg-[#f8fafc] p-4">
                <p className="text-xs font-black uppercase tracking-[0.18em] text-[#617186]">Reason</p>
                <p className="mt-2 text-sm font-semibold leading-6 text-[#334155]">
                  {prediction?.reason ?? "Run a forecast to explain why a junction is risky."}
                </p>
                {prediction?.match_note && <p className="mt-2 text-sm font-semibold text-[#617186]">{prediction.match_note}</p>}
                {prediction?.dispatch_station && (
                  <p className="mt-3 border border-teal-200 bg-teal-50 p-3 text-sm font-black text-[#0f766e]">
                    Notify: {prediction.dispatch_station}
                  </p>
                )}
                <p className="mt-3 text-sm font-black text-[#0f766e]">{prediction?.recommendation ?? "No recommendation yet"}</p>
              </div>
            </div>
          </Panel>
          </div>
        </section>

        <section className="grid gap-5 xl:grid-cols-[0.92fr_1fr_1.08fr]">
          <Panel title="Congestion Impact" icon={<Activity size={20} />} action="Lane loss + queue proxy" className="min-h-[420px]">
            <CongestionImpactPanel impact={congestionImpact} />
          </Panel>
          <Panel title="Incident Workflow" icon={<ListChecks size={20} />} action="Dispatch lifecycle" className="min-h-[420px]">
            <IncidentWorkflowPanel
              current={combinedDecision}
              incidents={incidents}
              error={incidentError}
              updating={incidentUpdating}
              onStatus={updateIncidentStatus}
            />
          </Panel>
          <Panel title="Before / After Analytics" icon={<BarChart3 size={20} />} action="Enforcement benefit" className="min-h-[420px]">
            <BeforeAfterPanel analytics={analytics} error={analyticsError} />
          </Panel>
        </section>

      </div>
    </main>
  );
}

function formatApiError(payload: unknown, fallback: string) {
  if (!payload || typeof payload !== "object") return fallback;
  const detail = (payload as { detail?: unknown }).detail;

  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (!item || typeof item !== "object") return String(item);
        const issue = item as { loc?: unknown[]; msg?: string };
        const location = Array.isArray(issue.loc) ? issue.loc.join(" > ") : "request";
        return `${location}: ${issue.msg ?? JSON.stringify(item)}`;
      })
      .join("; ");
  }
  if (detail && typeof detail === "object") return JSON.stringify(detail);
  return fallback;
}

function Panel({
  title,
  icon,
  action,
  children,
  className = "",
}: {
  title: string;
  icon: React.ReactNode;
  action: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`flex flex-col border border-[#d5dde6] bg-white p-4 shadow-panel ${className}`}>
      <div className="mb-4 flex shrink-0 items-center justify-between gap-3 border-b border-[#e2e8f0] pb-3">
        <div className="flex items-center gap-2">
          <span className="grid h-9 w-9 place-items-center bg-[#e6fffb] text-[#0f766e]">{icon}</span>
          <h2 className="text-lg font-black">{title}</h2>
        </div>
        <span className="hidden text-xs font-black uppercase tracking-[0.18em] text-[#617186] md:inline">{action}</span>
      </div>
      <div className="min-h-0 flex-1">{children}</div>
    </section>
  );
}

function CityRiskHeatmapHeader({
  prediction,
  primaryAlert,
  activeHotspots,
}: {
  prediction: LocationPredictionResponse | null;
  primaryAlert: boolean;
  activeHotspots: number;
}) {
  return (
    <div className="city-risk-summary mb-4 grid gap-3 border border-[#d5dde6] bg-[#f8fafc] p-4 lg:grid-cols-[1fr_auto] lg:items-center">
      <div>
        <p className="text-xs font-black uppercase tracking-[0.2em] text-[#0f766e]">Bengaluru corridor watch</p>
        <h3 className="mt-1 text-2xl font-black text-[#111827]">Predictive enforcement radar</h3>
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm">
        <CompactStat label="Coverage" value={primaryAlert ? "Live alert" : `${activeHotspots} corridors`} />
        <CompactStat label="Mode" value={primaryAlert ? "Alert" : "Patrol"} />
        <CompactStat label="Risk score" value={prediction ? prediction.risk_score.toFixed(0) : 0} />
      </div>
    </div>
  );
}

function CityRiskHeatmap({
  hotspots,
  prediction,
  primaryAlert,
  onHotspotSelect,
}: {
  hotspots: Hotspot[];
  prediction: LocationPredictionResponse | null;
  primaryAlert: boolean;
  onHotspotSelect: (hotspot: Hotspot) => void;
}) {
  const rankedHotspots = [...hotspots]
    .filter((item) => item.latitude !== null && item.longitude !== null)
    .sort((left, right) => right.risk_score - left.risk_score);
  const [selectedCorridorKey, setSelectedCorridorKey] = useState("");
  const selectedCorridor = rankedHotspots.find((item) => hotspotKey(item) === selectedCorridorKey) ?? rankedHotspots[0] ?? null;

  useEffect(() => {
    if (!rankedHotspots.length) return;
    if (!selectedCorridorKey || !rankedHotspots.some((item) => hotspotKey(item) === selectedCorridorKey)) {
      setSelectedCorridorKey(hotspotKey(rankedHotspots[0]));
    }
  }, [rankedHotspots, selectedCorridorKey]);

  function selectCorridor(item: Hotspot) {
    setSelectedCorridorKey(hotspotKey(item));
    onHotspotSelect(item);
  }

  return (
    <div className="relative min-h-[720px] overflow-hidden border border-[#d5dde6] bg-[#10151f] text-white">
      <div className="absolute inset-0 z-0">
        <CityRiskDarkMap hotspots={rankedHotspots} selectedHotspot={selectedCorridor} onSelect={selectCorridor} />
      </div>
      <div className="pointer-events-none absolute inset-0 z-10 city-grid opacity-25" />
      <div className="pointer-events-none absolute inset-0 z-10 bg-[linear-gradient(180deg,rgba(16,21,31,0.62)_0%,rgba(16,21,31,0.12)_44%,rgba(16,21,31,0.82)_100%)]" />

      <div className="absolute bottom-5 right-5 top-5 z-20 hidden w-[340px] flex-col border border-white/10 bg-[#10151f]/88 p-4 shadow-panel backdrop-blur lg:flex">
        <div className="flex items-center justify-between gap-3 border-b border-white/10 pb-3">
          <div>
            <p className="text-xs font-black uppercase tracking-[0.18em] text-teal-200">Priority corridors</p>
            <p className="mt-1 text-sm font-semibold text-slate-300">{rankedHotspots.length} unique corridors</p>
          </div>
          <Crosshair size={20} className="text-teal-200" />
        </div>
        <div className="mt-4 min-h-0 flex-1 space-y-2 overflow-y-auto pr-1 command-scroll">
          {rankedHotspots.map((item, index) => (
            <button
              key={hotspotKey(item)}
              type="button"
              onClick={() => selectCorridor(item)}
              className={`grid w-full grid-cols-[34px_1fr_auto] items-center gap-3 border p-2 text-left transition hover:border-teal-200 hover:bg-teal-400/10 ${
                selectedCorridor && hotspotKey(item) === hotspotKey(selectedCorridor)
                  ? "border-teal-200 bg-teal-400/15"
                  : "border-white/10 bg-white/5"
              }`}
            >
              <span className={`grid h-8 w-8 place-items-center text-xs font-black ${item.risk_score >= 70 ? "bg-red-500" : item.risk_score >= 40 ? "bg-amber-400 text-[#111827]" : "bg-teal-400 text-[#111827]"}`}>
                {index + 1}
              </span>
              <span className="min-w-0">
                <span className="block truncate text-sm font-black">{item.junction}</span>
                <span className="block text-xs font-semibold text-slate-400">
                  {item.police_station ?? item.zone} · {hotspotHourSummary(item)}
                </span>
              </span>
              <span className="flex items-center gap-2 text-lg font-black">
                {item.risk_score.toFixed(0)}
                <ArrowUpRight size={16} className="text-teal-200" />
              </span>
            </button>
          ))}
        </div>
        <div className="mt-4 grid grid-cols-3 gap-2 text-[10px] font-black uppercase tracking-[0.12em] text-slate-200">
          <span className="border border-red-400/40 bg-red-500/20 px-2 py-2 text-center">High</span>
          <span className="border border-amber-300/40 bg-amber-400/20 px-2 py-2 text-center">Med</span>
          <span className="border border-teal-300/40 bg-teal-400/20 px-2 py-2 text-center">Watch</span>
        </div>
      </div>

      <div className="absolute bottom-5 left-5 right-5 z-20 grid grid-cols-3 gap-3 text-sm lg:right-[400px]">
        <CompactDark label="Peak hour" value={prediction?.features.peak_hour ? "Yes" : "No"} />
        <CompactDark label="Avg delay" value={`${prediction?.features.avg_duration ?? 0}m`} />
        <CompactDark label="Risk score" value={prediction ? prediction.risk_score.toFixed(0) : 0} />
      </div>

      <div className="absolute bottom-[132px] left-5 right-5 z-20 grid gap-2 lg:hidden">
        <p className="text-xs font-black uppercase tracking-[0.18em] text-teal-200">Top corridors</p>
        <div className="max-h-[180px] grid gap-2 overflow-y-auto pr-1 command-scroll sm:grid-cols-2">
          {rankedHotspots.map((item) => (
            <button
              key={hotspotKey(item)}
              type="button"
              onClick={() => selectCorridor(item)}
              className={`border px-3 py-2 text-left transition hover:border-teal-200 ${
                selectedCorridor && hotspotKey(item) === hotspotKey(selectedCorridor)
                  ? "border-teal-200 bg-teal-400/15"
                  : "border-white/10 bg-[#10151f]/88"
              }`}
            >
              <p className="truncate text-sm font-black">{item.junction}</p>
              <p className="text-xs font-semibold text-slate-400">{item.risk_score.toFixed(0)} risk · {hotspotHourSummary(item)}</p>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function CityRiskDarkMap({
  hotspots,
  selectedHotspot,
  onSelect,
}: {
  hotspots: Hotspot[];
  selectedHotspot: Hotspot | null;
  onSelect: (hotspot: Hotspot) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layerRef = useRef<L.LayerGroup | null>(null);
  const onSelectRef = useRef(onSelect);
  const selectedKey = selectedHotspot ? hotspotKey(selectedHotspot) : "";

  useEffect(() => {
    onSelectRef.current = onSelect;
  }, [onSelect]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      attributionControl: false,
      boxZoom: false,
      doubleClickZoom: false,
      dragging: false,
      keyboard: false,
      scrollWheelZoom: false,
      touchZoom: false,
      zoomControl: false,
    }).setView([12.9716, 77.5946], 11);
    mapRef.current = map;

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      maxZoom: 19,
    }).addTo(map);
    layerRef.current = L.layerGroup().addTo(map);
    window.setTimeout(() => map.invalidateSize(), 0);

    return () => {
      map.remove();
      mapRef.current = null;
      layerRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!mapRef.current || !layerRef.current) return;
    const rows = hotspots.filter((item): item is Hotspot & { latitude: number; longitude: number } => item.latitude !== null && item.longitude !== null);
    layerRef.current.clearLayers();

    rows.forEach((hotspot) => {
      const isSelected = hotspotKey(hotspot) === selectedKey;
      const marker = L.marker([hotspot.latitude, hotspot.longitude], {
        icon: cityRiskMapIcon(hotspot, isSelected),
        zIndexOffset: isSelected ? 1000 : Math.round(hotspot.risk_score),
      });
      marker.bindTooltip(`${hotspot.junction} · ${hotspot.risk_score.toFixed(0)} risk`, {
        className: "city-risk-tooltip",
        direction: "top",
        opacity: 0.95,
      });
      marker.on("click", () => onSelectRef.current(hotspot));
      marker.addTo(layerRef.current as L.LayerGroup);
    });

    if (rows.length && !selectedHotspot) {
      const bounds = L.latLngBounds(rows.map((item) => [item.latitude, item.longitude] as [number, number]));
      mapRef.current.fitBounds(bounds, { animate: false, padding: [36, 36] });
    }
  }, [hotspots, selectedKey, selectedHotspot]);

  useEffect(() => {
    if (!mapRef.current || !selectedHotspot || selectedHotspot.latitude === null || selectedHotspot.longitude === null) return;
    mapRef.current.flyTo([selectedHotspot.latitude, selectedHotspot.longitude], 13, {
      animate: true,
      duration: 0.7,
    });
  }, [selectedKey, selectedHotspot]);

  return (
    <div className="relative h-full w-full overflow-hidden border border-white/10 bg-[#060b14]">
      <div ref={containerRef} className="city-risk-map h-full w-full" />
    </div>
  );
}

function cityRiskMapIcon(hotspot: Hotspot, selected: boolean) {
  const tone = hotspot.risk_score >= 70 ? "high" : hotspot.risk_score >= 40 ? "medium" : "watch";
  return L.divIcon({
    className: "city-risk-marker",
    html: `<span class="city-risk-pin city-risk-pin-${tone} ${selected ? "city-risk-pin-selected" : ""}">${Math.round(hotspot.risk_score)}</span>`,
    iconSize: selected ? [54, 54] : [38, 38],
    iconAnchor: selected ? [27, 27] : [19, 19],
  });
}

function hotspotKey(item: Hotspot) {
  return `${item.junction}-${item.zone}-${item.hour}-${item.latitude ?? "na"}-${item.longitude ?? "na"}`;
}

function hotspotHourSummary(item: Hotspot) {
  const hours = [...new Set((item.active_hours ?? [item.hour]).filter((hour) => Number.isFinite(hour)))].sort((a, b) => a - b);
  const peak = `${String(item.hour).padStart(2, "0")}:00`;
  if (hours.length <= 1) return `peak ${peak}`;
  return `peak ${peak} · ${hours.length} active hours`;
}

function Metric({ icon, label, value, tone }: { icon: React.ReactNode; label: string; value: string; tone: "teal" | "red" | "amber" }) {
  const tones = {
    teal: "bg-teal-50 text-teal-800 border-teal-200",
    red: "bg-red-50 text-red-800 border-red-200",
    amber: "bg-amber-50 text-amber-900 border-amber-200",
  };
  return (
    <div className={`min-w-[120px] border px-3 py-2 ${tones[tone]}`}>
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-xs font-black uppercase tracking-[0.12em]">{label}</span>
      </div>
      <p className="mt-1 text-2xl font-black">{value}</p>
    </div>
  );
}

function CompactStat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="border border-[#d5dde6] bg-white p-3">
      <p className="text-xs font-black uppercase tracking-[0.14em] text-[#617186]">{label}</p>
      <p className="mt-1 text-2xl font-black">{value}</p>
    </div>
  );
}

function PolicyStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-[#d5dde6] bg-[#f8fafc] p-3">
      <p className="text-[10px] font-black uppercase tracking-[0.14em] text-[#617186]">{label}</p>
      <p className="mt-1 text-sm font-black leading-5 text-[#111827]">{value}</p>
    </div>
  );
}

function CombinedDecisionCard({
  decision,
  loading,
  error,
  onExport,
}: {
  decision: CombinedDecisionResponse | null;
  loading: boolean;
  error: string;
  onExport: () => void;
}) {
  const tone =
    decision?.priority === "Critical"
      ? "border-red-300 bg-red-50 text-red-800"
      : decision?.priority === "High"
        ? "border-orange-300 bg-orange-50 text-orange-900"
        : decision?.priority === "Watch"
          ? "border-amber-300 bg-amber-50 text-amber-900"
          : "border-teal-200 bg-teal-50 text-teal-800";

  if (!decision && !loading && !error) {
    return (
      <div className="border border-[#d5dde6] bg-[#f8fafc] p-4">
        <p className="text-xs font-black uppercase tracking-[0.18em] text-[#617186]">Unified enforcement decision</p>
        <p className="mt-2 text-sm font-semibold text-[#334155]">Run a frame after the risk forecast to create an auditable incident package.</p>
      </div>
    );
  }

  return (
    <div className={`border p-4 ${tone}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-black uppercase tracking-[0.18em] opacity-75">Unified enforcement decision</p>
          <p className="mt-2 text-2xl font-black">{loading ? "Evaluating..." : decision?.dispatch ?? "Review required"}</p>
          {decision && (
            <p className="mt-1 text-xs font-black uppercase tracking-[0.14em] opacity-80">
              {decision.priority} · fused score {decision.fused_score.toFixed(0)} · {decision.incident_id}
            </p>
          )}
        </div>
        {decision && (
          <button
            type="button"
            onClick={onExport}
            title="Download evidence package"
            className="grid h-10 w-10 shrink-0 place-items-center border border-current bg-white/40 transition hover:bg-white"
          >
            <Download size={18} />
          </button>
        )}
      </div>
      {error && <p className="mt-3 border border-red-200 bg-white/65 p-2 text-sm font-semibold text-red-700">{error}</p>}
      {decision && (
        <div className="mt-3 space-y-3">
          {decision.dispatch_plan && (
            <div className="border border-current/20 bg-white/45 p-3">
              <p className="text-xs font-black uppercase tracking-[0.16em] opacity-75">Tactical dispatch plan</p>
              <p className="mt-2 text-sm font-black leading-5">From: {decision.dispatch_plan.from_station}</p>
              <p className="mt-1 text-sm font-semibold leading-5">To: {decision.dispatch_plan.target_stop}</p>
              <div className="mt-3 grid grid-cols-3 gap-2">
                <MiniDispatchStat label="Personnel" value={decision.dispatch_plan.personnel_count} />
                <MiniDispatchStat label="Units" value={decision.dispatch_plan.unit_count} />
                <MiniDispatchStat label="ETA" value={`${decision.dispatch_plan.eta_minutes}m`} />
              </div>
            </div>
          )}
          <div className="max-h-[112px] space-y-2 overflow-y-auto pr-1 command-scroll">
            {decision.reasons.map((reason) => (
              <p key={reason} className="border border-current/20 bg-white/45 px-3 py-2 text-xs font-semibold leading-5">
                {reason}
              </p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function MiniDispatchStat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="border border-current/20 bg-white/50 px-2 py-2">
      <p className="text-[10px] font-black uppercase tracking-[0.12em] opacity-70">{label}</p>
      <p className="mt-1 text-lg font-black">{value}</p>
    </div>
  );
}

function CongestionImpactPanel({ impact }: { impact: CongestionImpactResponse | null }) {
  if (!impact) {
    return (
      <div className="flex h-full min-h-[320px] flex-col justify-center border border-dashed border-[#98a9ba] bg-[#f8fafc] p-5 text-center">
        <Gauge className="mx-auto text-[#0f766e]" size={34} />
        <p className="mt-3 text-lg font-black">Run a frame to quantify road impact</p>
        <p className="mt-2 text-sm font-semibold leading-6 text-[#617186]">
          The impact model will combine live vehicle alerts, stationary duration, camera policy, and historical risk.
        </p>
      </div>
    );
  }

  const tone =
    impact.severity === "Severe"
      ? "text-red-700"
      : impact.severity === "High"
        ? "text-orange-700"
        : impact.severity === "Moderate"
          ? "text-amber-700"
          : "text-teal-700";

  return (
    <div className="space-y-4">
      <div className="border border-[#d5dde6] bg-[#f8fafc] p-4">
        <p className="text-xs font-black uppercase tracking-[0.18em] text-[#617186]">Impact score</p>
        <div className="mt-2 flex items-end justify-between gap-3">
          <p className={`text-5xl font-black ${tone}`}>{impact.impact_score.toFixed(0)}</p>
          <p className="text-right text-sm font-black uppercase tracking-[0.16em] text-[#617186]">{impact.severity}</p>
        </div>
        <p className="mt-3 text-sm font-semibold text-[#334155]">{impact.flow_state}</p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <ImpactStat label="Lane loss" value={`${impact.lane_capacity_loss_pct.toFixed(0)}%`} />
        <ImpactStat label="Throughput loss" value={`${impact.throughput_loss_pct.toFixed(0)}%`} />
        <ImpactStat label="Delay proxy" value={`${impact.estimated_delay_minutes.toFixed(1)}m`} />
        <ImpactStat label="Queue risk" value={`${impact.queue_risk_meters}m`} />
      </div>

      <div className="border border-[#99f6e4] bg-[#f0fdfa] p-4">
        <p className="text-xs font-black uppercase tracking-[0.18em] text-[#0f766e]">Recommended traffic action</p>
        <p className="mt-2 text-sm font-black leading-6 text-[#115e59]">{impact.action}</p>
      </div>
    </div>
  );
}

function ImpactStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-[#d5dde6] bg-white p-3">
      <p className="text-[10px] font-black uppercase tracking-[0.14em] text-[#617186]">{label}</p>
      <p className="mt-1 text-2xl font-black text-[#111827]">{value}</p>
    </div>
  );
}

function IncidentWorkflowPanel({
  current,
  incidents,
  error,
  updating,
  onStatus,
}: {
  current: CombinedDecisionResponse | null;
  incidents: CombinedDecisionResponse[];
  error: string;
  updating: string;
  onStatus: (incidentId: string, status: string) => void;
}) {
  const rows = incidents.length ? incidents : current ? [current] : [];

  return (
    <div className="flex h-full min-h-[320px] flex-col gap-4">
      <div className="border border-[#d5dde6] bg-[#f8fafc] p-4">
        <p className="text-xs font-black uppercase tracking-[0.18em] text-[#617186]">Current package</p>
        <p className="mt-2 text-2xl font-black">{current?.incident_id ?? "No incident yet"}</p>
        <p className="mt-1 text-sm font-semibold text-[#617186]">
          {current ? `${current.priority} · ${current.status} · ${current.dispatch}` : "Analyze a camera frame to open an incident."}
        </p>
      </div>

      {error && <p className="border border-red-200 bg-red-50 p-3 text-sm font-semibold text-red-700">{error}</p>}

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1 command-scroll">
        {rows.map((incident) => (
          <div key={incident.incident_id} className="border border-[#d5dde6] bg-white p-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-sm font-black">{incident.incident_id}</p>
                <p className="mt-1 text-xs font-bold text-[#617186]">
                  {incident.priority} · score {incident.fused_score.toFixed(0)} · {incident.status}
                </p>
              </div>
              <span className={`px-2 py-1 text-[10px] font-black uppercase tracking-[0.12em] ${incident.status === "Resolved" ? "bg-emerald-100 text-emerald-800" : incident.status === "Dispatched" ? "bg-blue-100 text-blue-800" : "bg-amber-100 text-amber-900"}`}>
                {incident.status}
              </span>
            </div>
            <p className="mt-2 text-xs font-semibold leading-5 text-[#334155]">{incident.decision}</p>
            <div className="mt-3 grid grid-cols-3 gap-2">
              {["Dispatched", "Resolved", "Dismissed"].map((status) => (
                <button
                  key={status}
                  type="button"
                  onClick={() => onStatus(incident.incident_id, status)}
                  disabled={updating === `${incident.incident_id}:${status}` || incident.status === status}
                  className="border border-[#cbd5e1] bg-[#f8fafc] px-2 py-2 text-[10px] font-black uppercase tracking-[0.1em] text-[#334155] transition hover:border-[#0f766e] hover:text-[#0f766e] disabled:cursor-not-allowed disabled:opacity-45"
                >
                  {status}
                </button>
              ))}
            </div>
          </div>
        ))}
        {!rows.length && (
          <div className="border border-dashed border-[#98a9ba] bg-[#f8fafc] p-5 text-center text-sm font-semibold text-[#617186]">
            Incident queue is empty.
          </div>
        )}
      </div>
    </div>
  );
}

function BeforeAfterPanel({ analytics, error }: { analytics: BeforeAfterResponse | null; error: string }) {
  if (error) return <p className="border border-red-200 bg-red-50 p-3 text-sm font-semibold text-red-700">{error}</p>;
  if (!analytics) {
    return (
      <div className="flex h-full min-h-[320px] items-center justify-center border border-dashed border-[#98a9ba] bg-[#f8fafc] p-5 text-center text-sm font-semibold text-[#617186]">
        Loading enforcement benefit model...
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-[320px] flex-col gap-4">
      <div className="grid grid-cols-3 gap-3">
        <ImpactStat label="Delay saved" value={`${analytics.summary.delay_saved_minutes.toFixed(0)}m`} />
        <ImpactStat label="Risk reduction" value={`${analytics.summary.risk_reduction_pct.toFixed(0)}%`} />
        <ImpactStat label="Patrol hours" value={`${analytics.summary.patrol_hours_saved.toFixed(1)}h`} />
      </div>
      <div className="border border-[#d5dde6] bg-[#f8fafc] p-4">
        <div className="flex items-center justify-between gap-3">
          <span>
            <p className="text-xs font-black uppercase tracking-[0.18em] text-[#617186]">Average corridor risk</p>
            <p className="mt-1 text-sm font-semibold text-[#334155]">Before targeted patrols vs after response staging</p>
          </span>
          <span className="text-right text-lg font-black text-[#0f766e]">
            {analytics.summary.average_risk_before.toFixed(0)} → {analytics.summary.average_risk_after.toFixed(0)}
          </span>
        </div>
      </div>
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1 command-scroll">
        {analytics.corridors.map((corridor) => (
          <div key={`${corridor.junction}-${corridor.zone}-${corridor.hour}`} className="border border-[#d5dde6] bg-white p-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="truncate text-sm font-black">{corridor.junction}</p>
                <p className="mt-1 text-xs font-semibold text-[#617186]">
                  {corridor.zone} · {corridor.hour}:00 · {corridor.action}
                </p>
              </div>
              <p className="text-sm font-black text-[#0f766e]">{corridor.delay_saved_minutes.toFixed(0)}m saved</p>
            </div>
            <div className="mt-3 h-2 bg-[#dbe4ee]">
              <div
                className="h-full bg-[#0f766e]"
                style={{ width: `${Math.max(8, Math.min(100, corridor.expected_reduction_pct))}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function CompactDark({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="border border-white/20 bg-white/10 p-3 backdrop-blur">
      <p className="text-xs font-black uppercase tracking-[0.14em] text-teal-100">{label}</p>
      <p className="mt-1 text-2xl font-black text-white">{value}</p>
    </div>
  );
}

function RiskGauge({ score, level }: { score: number; level: string }) {
  const circumference = 2 * Math.PI * 42;
  const dash = (Math.min(100, Math.max(0, score)) / 100) * circumference;
  return (
    <div className="flex flex-col items-center justify-center border border-[#d5dde6] bg-[#f8fafc] p-4">
      <div className="relative h-40 w-40">
        <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90">
          <circle cx="50" cy="50" r="42" stroke="#dbe4ee" strokeWidth="10" fill="none" />
          <circle
            cx="50"
            cy="50"
            r="42"
            stroke={score >= 70 ? "#dc2626" : score >= 40 ? "#d97706" : "#0f766e"}
            strokeWidth="10"
            fill="none"
            strokeLinecap="square"
            strokeDasharray={`${dash} ${circumference - dash}`}
          />
        </svg>
        <div className="absolute inset-0 grid place-items-center text-center">
          <div>
            <p className="text-4xl font-black">{score.toFixed(0)}</p>
            <p className="text-xs font-black uppercase tracking-[0.18em] text-[#617186]">{level}</p>
          </div>
        </div>
      </div>
      <p className="mt-2 text-sm font-bold text-[#617186]">Risk score</p>
    </div>
  );
}

function LocationMap({
  latitude,
  longitude,
  matchedLatitude,
  matchedLongitude,
  hotspots,
  onSelect,
  onHotspotSelect,
}: {
  latitude: number;
  longitude: number;
  matchedLatitude: number | null;
  matchedLongitude: number | null;
  hotspots: Hotspot[];
  onSelect: (latitude: number, longitude: number) => void;
  onHotspotSelect: (hotspot: Hotspot) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const selectedMarkerRef = useRef<L.Marker | null>(null);
  const matchedMarkerRef = useRef<L.Marker | null>(null);
  const hotspotLayerRef = useRef<L.LayerGroup | null>(null);
  const onSelectRef = useRef(onSelect);
  const onHotspotSelectRef = useRef(onHotspotSelect);

  useEffect(() => {
    onSelectRef.current = onSelect;
    onHotspotSelectRef.current = onHotspotSelect;
  }, [onSelect, onHotspotSelect]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      zoomControl: false,
      attributionControl: false,
    }).setView([latitude, longitude], 13);
    mapRef.current = map;

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
    }).addTo(map);
    L.control.zoom({ position: "bottomright" }).addTo(map);

    hotspotLayerRef.current = L.layerGroup().addTo(map);
    selectedMarkerRef.current = L.marker([latitude, longitude], {
      icon: mapIcon("#0f766e", "Selected"),
    }).addTo(map);

    map.on("click", (event: L.LeafletMouseEvent) => {
      onSelectRef.current(event.latlng.lat, event.latlng.lng);
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!mapRef.current || !selectedMarkerRef.current) return;
    const point: [number, number] = [latitude, longitude];
    selectedMarkerRef.current.setLatLng(point);
    mapRef.current.setView(point, Math.max(mapRef.current.getZoom(), 13), { animate: true });
  }, [latitude, longitude]);

  useEffect(() => {
    if (!mapRef.current) return;
    if (matchedLatitude === null || matchedLongitude === null) {
      matchedMarkerRef.current?.remove();
      matchedMarkerRef.current = null;
      return;
    }
    const point: [number, number] = [matchedLatitude, matchedLongitude];
    if (!matchedMarkerRef.current) {
      matchedMarkerRef.current = L.marker(point, {
        icon: mapIcon("#dc2626", "Match"),
      }).addTo(mapRef.current);
    } else {
      matchedMarkerRef.current.setLatLng(point);
    }
  }, [matchedLatitude, matchedLongitude]);

  useEffect(() => {
    if (!hotspotLayerRef.current) return;
    hotspotLayerRef.current.clearLayers();
    hotspots
      .filter((hotspot) => hotspot.latitude !== null && hotspot.longitude !== null)
      .forEach((hotspot) => {
        const marker = L.marker([hotspot.latitude as number, hotspot.longitude as number], {
          icon: mapIcon(hotspot.risk_score >= 70 ? "#dc2626" : hotspot.risk_score >= 40 ? "#d97706" : "#0f766e", String(Math.round(hotspot.risk_score))),
        });
        marker.bindTooltip(`${hotspot.junction} · ${hotspot.risk_score}`, {
          direction: "top",
          opacity: 0.95,
        });
        marker.on("click", () => onHotspotSelectRef.current(hotspot));
        marker.addTo(hotspotLayerRef.current as L.LayerGroup);
      });
  }, [hotspots]);

  return (
    <div className="space-y-2">
      <div ref={containerRef} className="h-[280px] w-full border border-[#cbd5e1]" />
      <p className="text-xs font-bold uppercase tracking-[0.14em] text-[#617186]">
        Click the map to select exact coordinates. Hotspot markers use real dataset latitude/longitude.
      </p>
    </div>
  );
}

function mapIcon(color: string, label: string) {
  return L.divIcon({
    className: "parking-map-marker",
    html: `<span style="background:${color}">${label}</span>`,
    iconSize: [34, 34],
    iconAnchor: [17, 17],
  });
}

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("Dashboard root element is missing.");

const rootWindow = window as Window & { __parkingDashboardRoot?: ReturnType<typeof createRoot> };
rootWindow.__parkingDashboardRoot ??= createRoot(rootElement);
rootWindow.__parkingDashboardRoot.render(<App />);
