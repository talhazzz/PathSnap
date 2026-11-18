import { useEffect, useMemo } from "react";
import {
  CircleMarker,
  MapContainer,
  Polyline,
  TileLayer,
  Tooltip,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";

function FitBoundsController({ bounds }) {
  const map = useMap();
  useEffect(() => {
    if (bounds && bounds.length === 2) {
      map.fitBounds(bounds, { padding: [28, 28] });
    }
  }, [map, bounds]);
  return null;
}

export default function RouteMap({ gpsCoordinates = [], routeCoordinates = [], tripId }) {
  const sampledGps = useMemo(() => {
    if (!gpsCoordinates.length) return [];
    const step = Math.max(1, Math.floor(gpsCoordinates.length / 350));
    return gpsCoordinates.filter((_, idx) => idx % step === 0);
  }, [gpsCoordinates]);

  const bounds = useMemo(() => {
    const allPts = routeCoordinates.length ? routeCoordinates : gpsCoordinates;
    if (!allPts.length) return null;
    let minLat = Infinity;
    let maxLat = -Infinity;
    let minLon = Infinity;
    let maxLon = -Infinity;
    for (const [lat, lon] of allPts) {
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
      if (lon < minLon) minLon = lon;
      if (lon > maxLon) maxLon = lon;
    }
    return [
      [minLat, minLon],
      [maxLat, maxLon],
    ];
  }, [gpsCoordinates, routeCoordinates]);

  const center = useMemo(() => {
    if (!bounds) return [20.5937, 78.9629];
    return [
      (bounds[0][0] + bounds[1][0]) / 2,
      (bounds[0][1] + bounds[1][1]) / 2,
    ];
  }, [bounds]);

  const startPt = routeCoordinates[0] || gpsCoordinates[0] || null;
  const endPt =
    routeCoordinates[routeCoordinates.length - 1] ||
    gpsCoordinates[gpsCoordinates.length - 1] ||
    null;

  return (
    <div className="leaflet-wrapper">
      <MapContainer
        center={center}
        zoom={14}
        style={{ height: "100%", width: "100%", borderRadius: "9px" }}
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution="&copy; OpenStreetMap contributors &copy; CARTO"
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />

        {bounds && <FitBoundsController bounds={bounds} />}

        {sampledGps.map((pt, idx) => (
          <CircleMarker
            key={`gps-${idx}`}
            center={pt}
            radius={2.5}
            pathOptions={{
              color: "#38bdf8",
              fillColor: "#42a5ff",
              fillOpacity: 0.75,
              weight: 1,
            }}
          />
        ))}

        {routeCoordinates.length > 1 && (
          <Polyline
            positions={routeCoordinates}
            pathOptions={{
              color: "#a855f7",
              weight: 4.5,
              opacity: 0.92,
            }}
          >
            <Tooltip sticky>Predicted Route (Trip {tripId})</Tooltip>
          </Polyline>
        )}

        {startPt && (
          <CircleMarker
            center={startPt}
            radius={6}
            pathOptions={{
              color: "#ffffff",
              fillColor: "#22c55e",
              fillOpacity: 1,
              weight: 2,
            }}
          >
            <Tooltip>Start</Tooltip>
          </CircleMarker>
        )}

        {endPt && (
          <CircleMarker
            center={endPt}
            radius={6}
            pathOptions={{
              color: "#ffffff",
              fillColor: "#ef4444",
              fillOpacity: 1,
              weight: 2,
            }}
          >
            <Tooltip>End</Tooltip>
          </CircleMarker>
        )}
      </MapContainer>
    </div>
  );
}
