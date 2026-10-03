import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { LocaleProvider } from "./i18n";
import { ThemeProvider } from "./i18n/ThemeProvider";
import { ToastProvider, ToastViewport } from "./components/Toast";
import "./styles.css";

const rootEl = document.getElementById("root");
if (!rootEl) throw new Error("Root element missing");

createRoot(rootEl).render(
  <StrictMode>
    {/* v2.0.26.2 (PR-5) — ThemeProvider wraps the app so any
        descendant can call `useTheme()` and pick light / dark /
        system. LocaleProvider stays outermost so the catalog
        loads before the theme side-effect mutates the document.
        v2.0.26.3 (PR-6) — ToastProvider sits INSIDE so descendants
        can call `useToast()`, and ToastViewport lives at the same
        level so the fixed-position container is mounted once at
        the root (siblings of the App tree, not inside it). */}
    <LocaleProvider>
      <ThemeProvider>
        <ToastProvider>
          <App />
          <ToastViewport />
        </ToastProvider>
      </ThemeProvider>
    </LocaleProvider>
  </StrictMode>,
);