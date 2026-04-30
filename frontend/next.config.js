/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy any /api/* request from the browser to the FastAPI backend.
  // This avoids CORS errors because the browser talks to Next.js (same origin)
  // and Next.js forwards the request to FastAPI internally.
  async rewrites() {
    return [
      {
        source: "/api/:path*",           // what the frontend calls
        destination: "http://localhost:8000/:path*", // where FastAPI is running
      },
    ];
  },
};

module.exports = nextConfig;
