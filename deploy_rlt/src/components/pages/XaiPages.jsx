import { useEffect, useMemo, useState } from "react";
import axios from "axios";

export default function XaiPage() {
  const [status, setStatus] = useState(null);
  const [instanceIdx, setInstanceIdx] = useState(0);
  const [plotType, setPlotType] = useState("bar");
  const [refreshKey, setRefreshKey] = useState(Date.now());
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    setError(null);

    axios
      .get("/xai/status")
      .then((res) => {
        if (mounted) setStatus(res.data);
      })
      .catch((e) => {
        if (mounted) setError(e?.response?.data?.detail ?? String(e));
      });

    return () => {
      mounted = false;
    };
  }, []);

  const heatmapSrc = useMemo(
    () => `/xai/rlt-heatmap.png?t=${refreshKey}`,
    [refreshKey]
  );

  const shapSrc = useMemo(
    () =>
      `/xai/shap-explanation.png?instance_idx=${encodeURIComponent(
        instanceIdx
      )}&plot_type=${encodeURIComponent(plotType)}&t=${refreshKey}`,
    [instanceIdx, plotType, refreshKey]
  );

  const comparisonSrc = useMemo(
    () => `/xai/comparison.png?t=${refreshKey}`,
    [refreshKey]
  );

  const onRefresh = () => setRefreshKey(Date.now());

  return (
    <div className="page">
      <h1>XAI</h1>

      {error && (
        <div className="error">
          {error}
        </div>
      )}

      {status && (
        <div className="card">
          <div>Pipeline loaded: {String(status.pipeline_loaded)}</div>
          <div>XAI ready: {String(status.xai_available)}</div>
          <div>SHAP installed: {String(status.shap_installed)}</div>
        </div>
      )}

      <div className="card">
        <h2>Contrôles SHAP</h2>
        <h2>       </h2>
        

        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          <label>
            instance_idx:
            <input
              type="number"
              min={0}
              value={instanceIdx}
              onChange={(e) => setInstanceIdx(Number(e.target.value))}
              style={{ marginLeft: 8, width: 120 }}
            />
          </label>

          <label>
            plot_type:
            <select
              value={plotType}
              onChange={(e) => setPlotType(e.target.value)}
              style={{ marginLeft: 8 }}
            >
              <option value="bar">bar</option>
              <option value="waterfall">waterfall</option>
              <option value="force">force</option>
            </select>
          </label>

          <button type="button" onClick={onRefresh}>
            Refresh images
          </button>
        </div>
      </div>

      <div className="grid">
        <div className="card">
          <h2>RLT Heatmap</h2>
          <img
            src={heatmapSrc}
            alt="RLT heatmap"
            style={{ width: "100%", borderRadius: 10 }}
            onError={() => console.log("Heatmap load error:", heatmapSrc)}
          />
        </div>

        <div className="card">
          <h2>SHAP explanation</h2>
          <img
            src={shapSrc}
            alt="SHAP explanation"
            style={{ width: "100%", borderRadius: 10 }}
            onError={() => console.log("SHAP load error:", shapSrc)}
          />
        </div>

        <div className="card">
          <h2>Comparison</h2>
          <img
            src={comparisonSrc}
            alt="Model comparison"
            style={{ width: "100%", borderRadius: 10 }}
            onError={() => console.log("Comparison load error:", comparisonSrc)}
          />
        </div>
      </div>
    </div>
  );
}
