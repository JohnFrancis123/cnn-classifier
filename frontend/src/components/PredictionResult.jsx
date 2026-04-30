// PredictionResult displays the response from the /predict API.
// It receives a result object shaped like: { prediction: "frog", confidence: 0.9834 }

import styles from "./PredictionResult.module.css";

export default function PredictionResult({ result }) {
  // The API returns confidence as a 0–1 float; multiply by 100 for a percentage
  const confidencePct = Math.round(result.confidence * 100);

  return (
    <div className={styles.result}>
      {/* Predicted class name */}
      <p className={styles.label}>Prediction</p>
      <p className={styles.prediction}>{result.prediction}</p>

      {/* Confidence score with a visual progress bar */}
      <p className={styles.label}>Confidence</p>
      <div className={styles.barTrack}>
        {/*
          The filled portion width is set inline based on the confidence value.
          A CSS transition animates it in when the result first appears.
        */}
        <div className={styles.barFill} style={{ width: `${confidencePct}%` }} />
      </div>
      <p className={styles.confidence}>{confidencePct}%</p>
    </div>
  );
}
