import { useState } from "react";

export default function ScoreFeatures() {
  const [error, setError] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  const imageUrl = `http://127.0.0.1:8000/eda/top.png?n=20&reload=${reloadKey}`;

  return (
    <div style={{ padding: "20px" }}>
      <h1>Top 20 Feature Importance</h1>

      <button
        onClick={() => {
          setError(false);
          setReloadKey(prev => prev + 1);
        }}
        style={{
          marginBottom: "15px",
          padding: "8px 16px",
          cursor: "pointer"
        }}
      >
       
      </button>

      {error && (
        <div style={{ color: "red" }}>
          VI non disponible. Entraînez le modèle d'abord.
        </div>
      )}

      <img
        src={imageUrl}
        alt="Top Features"
        style={{ maxWidth: "100%", border: "1px solid #ccc" }}
        onError={() => setError(true)}
      />
    </div>
  );
}
