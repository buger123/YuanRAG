import { useEffect } from "react";
import type { ModelState } from "../hooks/useModelsReady";
import { Logo } from "./Logo";
import { statusLabel, type ModelStatus } from "../i18n/status";
import { useLocale } from "../i18n";

interface Props {
  // v2.0.25 P0-F1 — App owns the shared ``useModelsReady`` hook
  // (single poll source) and passes the slice each modal needs
  // down. ``modelsReady`` flips when both embedding + reranker
  // are ready; the effect below fires ``onReady`` so App can
  // dismiss the modal.
  modelsReady: boolean;
  embedding: ModelState | null;
  reranker: ModelState | null;
  onReady: () => void;
}

export function ModelDownloadProgress({
  modelsReady,
  embedding,
  reranker,
  onReady,
}: Props) {
  // v2.0.7 i18n-1 — translate the first-run modal strings.
  const { t } = useLocale();

  useEffect(() => {
    if (modelsReady) onReady();
  }, [modelsReady, onReady]);

  return (
    <div className="modal-backdrop">
      <div className="modal modal-download">
        <div className="modal-download-logo">
          <Logo size={48} />
        </div>
        <h2>{t("modelDownload.title")}</h2>
        <p>{t("modelDownload.body")}</p>
        <ModelRow
          name={t("modelDownload.embeddingLabel")}
          status={(embedding?.status as ModelStatus) ?? "pending"}
        />
        <ModelRow
          name={t("modelDownload.rerankerLabel")}
          status={(reranker?.status as ModelStatus) ?? "pending"}
        />
        <p className="modal-footnote">
          {t("modelDownload.footnote")}
        </p>
      </div>
    </div>
  );
}

function ModelRow({ name, status }: { name: string; status: ModelStatus }) {
  // PR-4 (v2.0.26.1): use ``statusLabel(t, ...)`` so the model-
  // download modal flips when SettingsDialog toggles the language.
  // ``useLocale`` is imported below alongside the helper.
  const { t } = useLocale();
  const label = statusLabel(t, status);
  return (
    <div className="settings-model-row">
      <span>{name}</span>
      <span
        style={{
          color: status === "ready" ? "var(--accent)" : "var(--text-dim)",
          fontSize: 12,
        }}
      >
        {label}
      </span>
    </div>
  );
}