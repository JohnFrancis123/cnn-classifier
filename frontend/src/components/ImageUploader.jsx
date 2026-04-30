// ImageUploader renders a clickable drop zone.
// - When a file is chosen, it calls onFileChange(file) to tell the parent page.
// - If a preview URL already exists, it shows the image inside the zone.

import styles from "./ImageUploader.module.css";

export default function ImageUploader({ preview, onFileChange }) {
  // Extract the File object from the native input change event
  const handleChange = (e) => {
    const file = e.target.files?.[0];
    if (file) onFileChange(file);
  };

  return (
    <div className={styles.uploader}>
      {/*
        The <label> is tied to the hidden <input> via the htmlFor / id pair.
        Clicking anywhere on the label opens the file picker.
      */}
      <label className={styles.label} htmlFor="file-input">
        {preview ? (
          // Show the chosen image as a preview inside the drop zone
          <img src={preview} alt="Selected image preview" className={styles.preview} />
        ) : (
          // Placeholder text shown before any file is selected
          <span className={styles.placeholder}>Click to select an image</span>
        )}
      </label>

      {/* The actual file input — hidden so the styled label acts as the UI */}
      <input
        id="file-input"
        type="file"
        accept="image/*"          // Restrict the picker to image files only
        className={styles.hiddenInput}
        onChange={handleChange}
      />
    </div>
  );
}
