// src/App.jsx
import { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import "./App.css";
import Layout from "./layout/layout";

import UploadTrain from "./components/UploadTrain";
import EdaUnderstanding from "./components/pages/EdaUnderstanding";
import EdaCorrelation from "./components/pages/EdaCorrelation";
import EdaBoxplots from "./components/pages/EdaBoxplots";
import ScoreFeatures from "./components/pages/ScoreFeatures";
import XaiPage from "./components/pages/XaiPages"; // ✅ NEW

export default function App() {
  const [trainingDone, setTrainingDone] = useState(false);

  const handleTrainSuccess = () => {
    console.log("Training finished!");
    setTrainingDone(true);
  };

  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Navigate to="/train" replace />} />

          <Route
            path="/train"
            element={<UploadTrain onTrainSuccess={handleTrainSuccess} />}
          />

          <Route path="/eda/understanding" element={<EdaUnderstanding />} />
          <Route path="/eda/correlation" element={<EdaCorrelation />} />
          <Route path="/eda/boxplots" element={<EdaBoxplots />} />
          <Route path="/eda/top" element={<ScoreFeatures />} />

          {/* ✅ XAI (one page) */}
          <Route
            path="/xai"
            element={
              trainingDone ? <XaiPage /> : <Navigate to="/train" replace />
            }
          />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
