# Interface

The React + TypeScript interface of the app, built with Vite. See the main
[README](../README.md) for setup.

```bash
npm install
npm run build    # writes dist/, which the Python server serves at http://127.0.0.1:8000
npm run dev      # hot reload on http://localhost:5173, forwarding /api to the server on :8000
npm run lint
```

- `src/api.ts`: typed client for the server's endpoints, including the streamed answers
- `src/App.tsx`: the page layout
- `src/components/`: folder list, document panel (upload, delete), question panel,
  answer text with clickable citations, source list
