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
