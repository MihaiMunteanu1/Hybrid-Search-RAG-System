import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Development: `npm run dev` serves the UI on :5173 and forwards /api to the Python
// server on :8000, so both run side by side with hot reload.
// Production: `npm run build` writes dist/, which FastAPI serves itself at "/".
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
