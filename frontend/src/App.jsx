import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  Bell,
  ChartNoAxesCombined,
  ExternalLink,
  LayoutDashboard,
  Loader2,
  Map,
  MapPin,
  Menu,
  Play,
  RefreshCw,
  Route,
  Search,
  Settings,
  Zap,
} from "lucide-react";

import RouteMap from "./components/RouteMap";
import {
  API_BASE_URL,
  fetchHealth,
  fetchModels,
  fetchTrips,
  getTripMapUrl,
  predictTrip,
} from "./services/api";
import "./App.css";

function formatPercent(val) {
  if (val === null || val === undefined || Number.isNaN(val)) return "—";
  return `${(val * 100).toFixed(2)}%`;
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "—";
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
}

function App() {
  const [backendStatus, setBackendStatus] = useState("checking"); // checking | online | offline
  const [backendError, setBackendError] = useState("");
  const [activePartition, setActivePartition] = useState("");
  const [trips, setTrips] = useState([]);
  const [tripsLoading, setTripsLoading] = useState(false);

  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [useHmm, setUseHmm] = useState(true);

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTripId, setSelectedTripId] = useState("");

  const [predicting, setPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState("");
  const [predictionResult, setPredictionResult] = useState(null);
  const [processedCount, setProcessedCount] = useState(0);

  const loadBackendData = useCallback(async () => {
    setBackendStatus("checking");
    setBackendError("");
    setTripsLoading(true);
    try {
      const [healthData, modelsData, tripsData] = await Promise.all([
        fetchHealth(),
        fetchModels(),
        fetchTrips(),
      ]);

      setBackendStatus(healthData.status === "healthy" ? "online" : "offline");

      const availableModels = (modelsData.models || []).filter((m) => m.available);
      setModels(modelsData.models || []);
      setSelectedModel((prev) => {
        if (prev && availableModels.some((m) => m.name === prev)) return prev;
        return modelsData.default_model || availableModels[0]?.name || "";
      });

      const fetchedTrips = tripsData.trips || [];
      setTrips(fetchedTrips);
      setActivePartition(tripsData.partition || "");
      setSelectedTripId((prev) => {
        if (prev && fetchedTrips.some((t) => t.trip_id === prev)) return prev;
        return fetchedTrips[0]?.trip_id || "";
      });
    } catch (err) {
      setBackendStatus("offline");
      setBackendError(err.message || "Unable to connect to PathSnap backend");
    } finally {
      setTripsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadBackendData();
  }, [loadBackendData]);

  const filteredTrips = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return trips;
    return trips.filter((t) => t.trip_id.toLowerCase().includes(q));
  }, [trips, searchQuery]);

  const selectedTrip = useMemo(
    () => trips.find((t) => t.trip_id === selectedTripId) || null,
    [trips, selectedTripId]
  );

  const handleRunReconstruction = async () => {
    if (!selectedTripId || predicting) return;
    setPredicting(true);
    setPredictionError("");
    try {
      const result = await predictTrip(selectedTripId, {
        modelName: selectedModel || null,
        useHmm,
        partition: activePartition || null,
      });
      setPredictionResult(result);
      setProcessedCount((c) => c + 1);
    } catch (err) {
      setPredictionError(err.message || "Route reconstruction failed");
    } finally {
      setPredicting(false);
    }
  };

  const handleOpenFoliumMap = () => {
    if (!selectedTripId) return;
    const url = getTripMapUrl(selectedTripId, {
      modelName: selectedModel || null,
      useHmm,
      partition: activePartition || null,
    });
    window.open(url, "_blank", "noopener,noreferrer");
  };

  const serverEval = predictionResult?.server_evaluation || null;
  const activeEvalBlock = predictionResult?.use_hmm
    ? serverEval?.hmm_viterbi
    : serverEval?.classifier_only;

  const top1Accuracy = activeEvalBlock?.top1_accuracy ?? null;
  const macroF1 = serverEval?.candidate_macro_f1 ?? null;
  const routeContinuity = predictionResult?.route_continuity ?? null;
  const totalTimeS = predictionResult?.timings_seconds?.total ?? null;

  return (
    <div className="app-container">
      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Route size={23} />
          </div>
          <div>
            <h2>PathSnap</h2>
            <span>MAP MATCHING PLATFORM</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <nav className="navigation">
          <a className="nav-item active" href="#dashboard">
            <LayoutDashboard size={19} />
            Dashboard
          </a>

          <a className="nav-item" href="#trip-explorer">
            <Map size={19} />
            Trip Explorer
          </a>

          <a className="nav-item" href="#map-section">
            <Route size={19} />
            Route Reconstruction
          </a>

          <a className="nav-item" href="#performance-section">
            <ChartNoAxesCombined size={19} />
            Model Performance
          </a>
        </nav>

        <div className="sidebar-bottom">
          <div className="api-endpoint-pill" title={API_BASE_URL}>
            <Settings size={14} />
            <span>{API_BASE_URL}</span>
          </div>

          <div className="profile">
            <div className="avatar">P</div>
            <div>
              <strong>PathSnap Team</strong>
              <span>Evaluator Interface</span>
            </div>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main-content" id="dashboard">
        {/* TOPBAR */}
        <header className="topbar">
          <div className="breadcrumb">
            <Menu className="mobile-menu" size={20} />
            <span>Workspace</span>
            <span>/</span>
            <strong>Dashboard</strong>
            {activePartition && (
              <span className="partition-pill">{activePartition}</span>
            )}
          </div>

          <div className="topbar-actions">
            <div className="search-box">
              <Search size={17} />
              <input
                placeholder="Filter trip ID..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            <button
              className="icon-button"
              onClick={loadBackendData}
              title="Refresh backend connection & trips"
            >
              <RefreshCw size={17} className={tripsLoading ? "spin" : ""} />
            </button>

            <button className="icon-button" aria-label="Notifications">
              <Bell size={19} />
            </button>

            <div className="avatar small">P</div>
          </div>
        </header>

        {/* DASHBOARD */}
        <section className="dashboard">
          <div className="welcome-section">
            <div>
              <div className="eyebrow">
                <Activity size={15} />
                GEOSPATIAL ANALYTICS
              </div>

              <h1>Route Intelligence Dashboard</h1>

              <p>
                Monitor GPS trajectories, reconstruct routes with ML + HMM/Viterbi,
                and inspect live server-side evaluation metrics.
              </p>
            </div>

            <div
              className={`system-status ${
                backendStatus === "offline"
                  ? "status-offline"
                  : backendStatus === "checking"
                  ? "status-checking"
                  : ""
              }`}
              title={`API: ${API_BASE_URL}`}
            >
              <span className="status-dot"></span>
              {backendStatus === "online" && "Backend connected"}
              {backendStatus === "checking" && "Connecting to backend..."}
              {backendStatus === "offline" && "Backend offline"}
            </div>
          </div>

          {(backendError || predictionError) && (
            <div className="error-banner" role="alert">
              <AlertCircle size={17} />
              <span>{predictionError || backendError}</span>
              <button className="retry-button" onClick={loadBackendData}>
                Retry
              </button>
            </div>
          )}

          {/* STATISTICS */}
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-top">
                <span>Total Trips</span>
                <MapPin size={19} />
              </div>
              <h2>{backendStatus === "online" ? trips.length : "—"}</h2>
              <p>
                {activePartition
                  ? `Loaded from ${activePartition}`
                  : "Available evaluation trips"}
              </p>
            </div>

            <div className="stat-card">
              <div className="stat-top">
                <span>Top-1 Accuracy</span>
                <ChartNoAxesCombined size={19} />
              </div>
              <h2>{formatPercent(top1Accuracy)}</h2>
              <p>
                {predictionResult
                  ? `Trip ${predictionResult.trip_id} (${
                      predictionResult.use_hmm ? "HMM/Viterbi" : "Classifier-only"
                    })`
                  : "Awaiting backend prediction"}
              </p>
            </div>

            <div className="stat-card">
              <div className="stat-top">
                <span>Routes Processed</span>
                <Route size={19} />
              </div>
              <h2>{processedCount}</h2>
              <p>Live predictions completed</p>
            </div>

            <div className="stat-card">
              <div className="stat-top">
                <span>Inference Time</span>
                <Zap size={19} />
              </div>
              <h2>{totalTimeS !== null ? `${totalTimeS.toFixed(2)}s` : "—"}</h2>
              <p>
                {predictionResult
                  ? `Model: ${predictionResult.model_name}`
                  : "Awaiting model execution"}
              </p>
            </div>
          </div>

          {/* MAP AND TRIP PANEL */}
          <div className="content-grid">
            <div className="panel map-panel" id="map-section">
              <div className="panel-header">
                <div>
                  <h3>Geospatial Overview</h3>
                  <p>
                    {predictionResult
                      ? `Trip ${predictionResult.trip_id} · ${predictionResult.gps_points_count} GPS points · ${predictionResult.predicted_arc_count} reconstructed arcs`
                      : "GPS trajectory and road network visualization"}
                  </p>
                </div>

                <button
                  className="secondary-button"
                  onClick={handleOpenFoliumMap}
                  disabled={!selectedTripId || backendStatus !== "online"}
                >
                  <ExternalLink size={15} />
                  Open Folium Map
                </button>
              </div>

              <div className="map-placeholder">
                {predictionResult ? (
                  <RouteMap
                    gpsCoordinates={predictionResult.gps_coordinates}
                    routeCoordinates={predictionResult.predicted_route_coordinates}
                    tripId={predictionResult.trip_id}
                  />
                ) : (
                  <>
                    <div className="map-grid"></div>
                    <div className="map-center">
                      <div className="map-icon">
                        {predicting ? (
                          <Loader2 size={28} className="spin" />
                        ) : (
                          <MapPin size={28} />
                        )}
                      </div>
                      <h3>
                        {predicting
                          ? `Reconstructing Trip ${selectedTripId}...`
                          : "Interactive Map"}
                      </h3>
                      <p>
                        {predicting
                          ? "Extracting road candidates, scoring with ML classifier, and decoding route with HMM/Viterbi."
                          : "Select an evaluation trip and click Run Reconstruction to render GPS fixes and the matched road path."}
                      </p>
                    </div>
                  </>
                )}

                <div className="map-legend">
                  <span>
                    <i className="gps-dot"></i> GPS Points
                  </span>
                  <span>
                    <i className="route-line"></i> Predicted Route
                  </span>
                </div>
              </div>
            </div>

            {/* TRIP SELECTOR */}
            <div className="panel trip-panel" id="trip-explorer">
              <div className="panel-header">
                <div>
                  <h3>Trip Explorer</h3>
                  <p>Select a trip and model to run inference</p>
                </div>
                <MapPin size={20} />
              </div>

              <label className="field-label" htmlFor="trip-select">
                SELECT TRIP
              </label>

              <select
                id="trip-select"
                className="select-control"
                value={selectedTripId}
                onChange={(e) => setSelectedTripId(e.target.value)}
                disabled={backendStatus !== "online" || !filteredTrips.length}
              >
                {!filteredTrips.length && (
                  <option value="">
                    {tripsLoading ? "Loading trips..." : "No trips available"}
                  </option>
                )}
                {filteredTrips.map((trip) => (
                  <option key={trip.trip_id} value={trip.trip_id}>
                    Trip {trip.trip_id} ({trip.gps_points} pts · {trip.approx_distance_km} km)
                  </option>
                ))}
              </select>

              <label className="field-label field-label-spaced" htmlFor="model-select">
                CLASSIFIER MODEL
              </label>

              <select
                id="model-select"
                className="select-control"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={backendStatus !== "online" || !models.length}
              >
                {models.map((m) => (
                  <option key={m.name} value={m.name} disabled={!m.available}>
                    {m.display_name || m.name}
                    {m.algorithm_family ? ` — ${m.algorithm_family}` : ""}
                    {!m.available ? " (unavailable)" : ""}
                  </option>
                ))}
              </select>

              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={useHmm}
                  onChange={(e) => setUseHmm(e.target.checked)}
                />
                <span>Enable HMM + Viterbi Route Decoding</span>
              </label>

              <div className="trip-info">
                <div>
                  <span>GPS observations</span>
                  <strong>{selectedTrip ? selectedTrip.gps_points.toLocaleString() : "—"}</strong>
                </div>
                <div>
                  <span>Approx. distance</span>
                  <strong>
                    {selectedTrip ? `${selectedTrip.approx_distance_km} km` : "—"}
                  </strong>
                </div>
                <div>
                  <span>Duration</span>
                  <strong>
                    {selectedTrip ? formatDuration(selectedTrip.duration_seconds) : "—"}
                  </strong>
                </div>
                <div>
                  <span>Road network arcs</span>
                  <strong>{selectedTrip ? selectedTrip.arcs.toLocaleString() : "—"}</strong>
                </div>
              </div>

              <button
                className="primary-button"
                onClick={handleRunReconstruction}
                disabled={backendStatus !== "online" || !selectedTripId || predicting}
              >
                {predicting ? (
                  <>
                    <Loader2 size={17} className="spin" />
                    Running Pipeline...
                  </>
                ) : (
                  <>
                    <Play size={17} />
                    Run Reconstruction
                  </>
                )}
              </button>

              <p className="helper-text">
                {backendStatus === "online"
                  ? `Connected to ${API_BASE_URL}`
                  : `Waiting for backend at ${API_BASE_URL}`}
              </p>
            </div>
          </div>

          {/* PERFORMANCE */}
          <div className="panel performance-panel" id="performance-section">
            <div className="panel-header">
              <div>
                <h3>Model Performance</h3>
                <p>
                  {predictionResult
                    ? `Live server-side evaluation for Trip ${predictionResult.trip_id} using ${predictionResult.model_name}`
                    : "Overview of route reconstruction metrics"}
                </p>
              </div>

              <span
                className={`pending-badge ${predictionResult ? "ready-badge" : ""}`}
              >
                {predictionResult
                  ? predictionResult.model_type === "regression_scorer"
                    ? predictionResult.use_hmm
                      ? "Scorer + HMM/Viterbi"
                      : "Candidate Scorer Only"
                    : predictionResult.use_hmm
                    ? "ML + HMM/Viterbi"
                    : "Classifier Only"
                  : "Awaiting evaluation"}
              </span>
            </div>

            <div className="performance-grid">
              <div>
                <span>Top-1 Accuracy</span>
                <h2>{formatPercent(top1Accuracy)}</h2>
                {serverEval && (
                  <small className="metric-sub">
                    Clf-only: {formatPercent(serverEval.classifier_only?.top1_accuracy)}
                  </small>
                )}
              </div>
              <div>
                <span>Candidate Macro F1</span>
                <h2>{macroF1 !== null ? macroF1.toFixed(4) : "—"}</h2>
                {serverEval && (
                  <small className="metric-sub">
                    Log Loss:{" "}
                    {serverEval.candidate_log_loss !== null &&
                    serverEval.candidate_log_loss !== undefined
                      ? serverEval.candidate_log_loss.toFixed(4)
                      : "N/A (regression scorer)"}
                  </small>
                )}
              </div>
              <div>
                <span>Route Continuity</span>
                <h2>{formatPercent(routeContinuity)}</h2>
                {predictionResult && (
                  <small className="metric-sub">
                    Clf-only: {formatPercent(predictionResult.classifier_only_route_continuity)}
                  </small>
                )}
              </div>
              <div>
                <span>Total Latency</span>
                <h2>{totalTimeS !== null ? `${totalTimeS.toFixed(2)}s` : "—"}</h2>
                {predictionResult?.timings_seconds && (
                  <small className="metric-sub">
                    ML: {predictionResult.timings_seconds.classifier_inference}s · HMM:{" "}
                    {predictionResult.timings_seconds.hmm_viterbi_decoding}s
                  </small>
                )}
              </div>
            </div>

            {predictionResult && (
              <div className="prediction-meta-strip">
                <span>
                  Predicted Arcs: <strong>{predictionResult.predicted_arc_count}</strong>
                </span>
                <span>
                  Candidates Scored:{" "}
                  <strong>{predictionResult.candidate_rows_count.toLocaleString()}</strong>
                </span>
                {activeEvalBlock && (
                  <>
                    <span>
                      Route F1: <strong>{activeEvalBlock.route_f1?.toFixed(4)}</strong>
                    </span>
                    <span>
                      Route Jaccard: <strong>{activeEvalBlock.route_jaccard?.toFixed(4)}</strong>
                    </span>
                  </>
                )}
                {serverEval && (
                  <span>
                    Candidate Miss Rate:{" "}
                    <strong>{formatPercent(serverEval.candidate_miss_rate)}</strong>
                  </span>
                )}
              </div>
            )}
          </div>

          <footer>
            PathSnap © 2026 · Distributed Map-Matching System
            <span>ML Hackathon Project</span>
          </footer>
        </section>
      </main>
    </div>
  );
}

export default App;
