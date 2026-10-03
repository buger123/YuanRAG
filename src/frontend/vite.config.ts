import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
    target: "es2020",
  },
  server: {
    port: 5173,
    proxy: {
      // v2.0.18 — same rationale as the preview proxy below: client
      // calls ``${BASE}/<path>`` with ``BASE = ""``, so the proxy
      // forwards every non-asset path to the backend.
      //
      // v2.0.28.19 — added ``bypass`` callback so the SPA entry ``/``
      // is served by vite directly (source ``index.html`` + HMR) and
      // NOT proxied to the backend. Without this:
      //   (a) bare ``/`` matched the negative-lookahead proxy pattern
      //       above (the lookahead only excluded ``assets/``, asset
      //       extensions, ``favicon.svg``, and the literal
      //       ``index.html`` — not the bare root);
      //   (b) the proxy forwarded ``/`` to FastAPI, which mounted
      //       ``dist/`` as static files and returned
      //       ``dist/index.html`` — a built bundle reference
      //       (``<script src="/assets/index-*.js">``);
      //   (c) vite dev had no ``/assets/`` source files (those are in
      //       ``dist/`` only), so its SPA fallback returned
      //       ``index.html`` HTML for the JS path;
      //   (d) the browser tried to parse HTML as JS, the bundle
      //       failed to load, React never mounted,
      //       ``useModelsReady`` never ran, and ``.chat-header-loading``
      //       showed "正在加载模型中" forever — even though the backend
      //       had models ready and was returning ``{"ready": true}``
      //       on every poll. A manual browser refresh recovered only
      //       because vite caches ``index.html`` and on retry served
      //       the dev variant instead of the proxied dist variant.
      "^/(?!assets/|favicon\\.svg|index\\.html|.*\\.(?:js|css|png|jpg|svg|ico|webp|woff2?|ttf))": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
        bypass: (req) => {
          // SPA entry: let vite serve source ``index.html`` so the
          // dev bundle (``/src/main.tsx``) gets the HMR client and
          // Vite middleware. Returning the URL means "vite should
          // handle this"; returning ``undefined`` means "proxy it".
          if (req.url === "/" || req.url === "") return req.url;
          // Source files: vite dev serves ``/src/...`` and ``/@...``
          // (HMR + React Refresh shims) via its own middleware. The
          // negative-lookahead above only excluded a fixed extension
          // list that didn't include ``.ts``/``.tsx``, so without
          // these bypasses those requests were proxied to FastAPI and
          // returned ``{"detail":"Not Found"}`` with ``server: uvicorn``
          // — same broken-bundle symptom as the bare ``/`` case, with
          // the React mount failing on a different first import.
          if (req.url.startsWith("/src/") || req.url.startsWith("/@")) {
            return req.url;
          }
          // ``/node_modules/.vite/deps/...`` (vite pre-bundled deps)
          // + ``/node_modules/vite/dist/client/env.mjs`` (referenced
          // by ``/@vite/client``) — vite dev's optimized cache.
          // Without this bypass those requests are proxied to
          // FastAPI and 404, which silently breaks the @vite/client
          // boot sequence and prevents React from ever mounting.
          if (req.url.startsWith("/node_modules/")) {
            return req.url;
          }
        },
      },
      "/ws": {
        target: "ws://127.0.0.1:8765",
        ws: true,
        changeOrigin: true,
      },
    },
  },
  preview: {
    // Layer 5 (Playwright e2e) needs the same backend proxy as `vite dev`
    // because the built static app calls `${window.location.host}/ws/chat`
    // and would otherwise try to hit the preview server itself.
    port: 4173,
    proxy: {
      // v2.0.18 — ``api/client.ts`` calls every backend endpoint via
      // ``fetch(`${BASE}/<path>``)`` with ``BASE = ""`` (same-origin),
      // NOT via an ``/api`` prefix. The previous proxy only matched
      // ``/api/*`` paths; the unprefixed paths fell through to vite
      // preview's HTML fallback, the JSON parser failed, and the
      // useModelsReady hook silently never flipped ready=true —
      // ``.chat-header-loading`` stayed forever and the smoke +
      // web_search e2e specs hung out at the 60 s prewarm wait.
      //
      // v2.0.28.19 — same ``bypass`` as the dev server: bare ``/``
      // is served by vite preview directly from ``dist/`` (which is
      // what we want — dist already has the built bundle). Without
      // bypass, the proxy intercepted ``/`` and forwarded to the
      // backend, which also returned ``dist/index.html``, BUT
      // ``/assets/...`` then went through vite preview's static
      // resolver and returned 200 OK + JS content (correct in preview,
      // broken in dev). See the dev-server comment for the bug story.
      //
      // The ``/src/`` and ``/@`` bypasses are dev-mode only — in
      // preview the dist bundle has no source imports — but harmless
      // to keep for symmetry: vite preview ignores them anyway.
      "^/(?!assets/|favicon\\.svg|index\\.html|.*\\.(?:js|css|png|jpg|svg|ico|webp|woff2?|ttf))": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
        bypass: (req) => {
          if (req.url === "/" || req.url === "") return req.url;
          if (req.url.startsWith("/src/") || req.url.startsWith("/@")) {
            return req.url;
          }
          if (req.url.startsWith("/node_modules/")) {
            return req.url;
          }
        },
      },
      "/ws": {
        target: "ws://127.0.0.1:8765",
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
