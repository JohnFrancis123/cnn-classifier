// The root layout wraps every page in the app.
// Next.js requires this file — it injects the <html> and <body> tags.
import "./globals.css";

// Metadata shown in the browser tab and search engines
export const metadata = {
  title: "CNN Classifier",
  description: "Image classification powered by a PyTorch CNN trained on CIFAR-10",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
