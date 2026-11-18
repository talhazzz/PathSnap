import axios from "axios";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL
).replace(/\/+$/, "");

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
  headers: {
    "Content-Type": "application/json",
  },
});

function extractErrorMessage(err, fallback = "Backend request failed") {
  if (err?.response?.data?.detail) {
    const detail = err.response.data.detail;
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  }
  if (err?.message) {
    return err.message;
  }
  return fallback;
}

export async function fetchHealth() {
  try {
    const { data } = await apiClient.get("/health", { timeout: 8000 });
    return data;
  } catch (err) {
    throw new Error(extractErrorMessage(err, "Unable to reach backend /health"));
  }
}

export async function fetchModels() {
  try {
    const { data } = await apiClient.get("/models", { timeout: 8000 });
    return data;
  } catch (err) {
    throw new Error(extractErrorMessage(err, "Unable to fetch available models"));
  }
}

export async function fetchTrips(partition = null) {
  try {
    const params = partition ? { partition } : {};
    const { data } = await apiClient.get("/trips", { params, timeout: 30000 });
    return data;
  } catch (err) {
    throw new Error(extractErrorMessage(err, "Unable to fetch evaluation trips"));
  }
}

export async function predictTrip(
  tripId,
  { modelName = null, useHmm = true, partition = null } = {}
) {
  try {
    const { data } = await apiClient.post(`/predict/trip/${encodeURIComponent(tripId)}`, {
      model_name: modelName,
      use_hmm: useHmm,
      partition,
    });
    return data;
  } catch (err) {
    throw new Error(extractErrorMessage(err, `Prediction failed for trip ${tripId}`));
  }
}

export function getTripMapUrl(
  tripId,
  { modelName = null, useHmm = true, partition = null } = {}
) {
  const params = new URLSearchParams();
  if (modelName) params.set("model_name", modelName);
  params.set("use_hmm", String(useHmm));
  if (partition) params.set("partition", partition);
  return `${API_BASE_URL}/trips/${encodeURIComponent(tripId)}/map?${params.toString()}`;
}

export async function explainTripWithAI(predictionResult) {
  try {
    const { data } = await apiClient.post(
      "/ai/explain-trip",
      {
        trip_id: predictionResult.trip_id,
        model_name: predictionResult.model_name,
        use_hmm: predictionResult.use_hmm,
        summary: {
          trip_id: predictionResult.trip_id,
          model_name: predictionResult.model_name,
          model_type: predictionResult.model_type,
          use_hmm: predictionResult.use_hmm,
          gps_points_count: predictionResult.gps_points_count,
          candidate_rows_count: predictionResult.candidate_rows_count,
          predicted_arc_count: predictionResult.predicted_arc_count,
          approx_distance_km: predictionResult.approx_distance_km,
          duration_seconds: predictionResult.duration_seconds,
          route_continuity: predictionResult.route_continuity,
          classifier_only_route_continuity: predictionResult.classifier_only_route_continuity,
          timings_seconds: predictionResult.timings_seconds,
        },
      },
      { timeout: 35000 }
    );
    return data;
  } catch (err) {
    throw new Error(extractErrorMessage(err, "Unable to generate AI route explanation"));
  }
}
