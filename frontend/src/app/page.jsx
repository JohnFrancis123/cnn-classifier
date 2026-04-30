"use client"; // This page uses React state, so it must run in the browser.

import { useState } from "react";
import ImageUploader from "../components/ImageUploader";
import PredictionResult from "../components/PredictionResult";
import styles from "./page.module.css";

export default function Home() {
  // selectedFile  — the File object the user chose
  // preview       — a temporary browser URL used to display the image
  // result        — the JSON response from the FastAPI /predict endpoint
  // loading       — true while the API request is in flight
  // error         — holds an error message if something goes wrong
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Called by ImageUploader whenever the user picks a new file
  const handleFileChange = (file) => {
    setSelectedFile(file);
    setResult(null);
    setError(null);
    // Revoke the old preview URL to free memory, then create a new one
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(file));
  };

  // Sends the image to the FastAPI backend and saves the prediction
  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    // FormData is how browsers send file uploads (multipart/form-data)
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      // /api/predict is proxied by Next.js to http://localhost:8000/predict
      const res = await fetch("/api/predict", { method: "POST", body: formData });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed");
      }

      setResult(await res.json()); // { prediction: "frog", confidence: 0.9834 }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className={styles.main}>
      <h1 className={styles.title}>CNN Image Classifier</h1>
      <p className={styles.subtitle}>
        Upload any image. The model will classify it into one of 10 CIFAR-10 categories
      </p>

      <div className={styles.card}>
        {/* File picker + preview */}
        <ImageUploader preview={preview} onFileChange={handleFileChange} />

        {/* Classify button — disabled until a file is chosen or while loading */}
        <button
          className={styles.button}
          onClick={handlePredict}
          disabled={!selectedFile || loading}
        >
          {loading ? "Classifying…" : "Classify Image"}
        </button>

        {/* Show an error if the API call failed */}
        {error && <p className={styles.error}>{error}</p>}

        {/* Show the prediction once the API responds */}
        {result && <PredictionResult result={result} />}
      </div>
    </main>
  );
}
