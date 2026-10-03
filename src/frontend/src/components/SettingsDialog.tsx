import { useEffect, useRef, useState } from "react";
import {
  getSettings,
  triggerModelDownload,
  updateSettings,
  type Settings,
} from "../api/client";
import type { ModelState } from "../hooks/useModelsReady";
import { statusLabel, type ModelStatus } from "../i18n/status";
import { useLocale, type Locale } from "../i18n";
import { useTheme, type Theme } from "../i18n/ThemeProvider";

interface Props {
  current: Settings | null;
  // v2.0.25 P0-F1 — App owns the shared ``useModelsReady`` hook
  // (single poll source) and passes the slice each modal needs
  // down. SettingsDialog renders per-model rows so it needs
  // ``embedding`` + ``reranker`` for the labels; ``modelsReady``
  // drives the ``onModelsReady`` callback below.
  modelsReady: boolean;
  embedding: ModelState | null;
  reranker: ModelState | null;
  onClose: () => void;
  onSaved: (s: Settings) => void;
  onModelsReady?: () => void;
}

const OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"];
const ANTHROPIC_MODELS = [
  "MiniMax-M3",
  "claude-opus-4",
  "claude-sonnet-5",
  "claude-haiku-4-5",
];

const LANG_OPTIONS: Array<{ value: Locale; key:
  | "settings.langZh"
  | "settings.langEn";
}> = [
  { value: "zh", key: "settings.langZh" },
  { value: "en", key: "settings.langEn" },
];

// v2.0.26.2 (PR-5) — 3-state theme toggle (light / dark / system).
// v2.0.28.8 — default changed to "light" (see ThemeProvider for why);
// a fresh user now lands in light mode and can opt into dark or
// system-auto from this segmented control. The Token list drives
// both the radios rendered below and the i18n key lookup.
const THEME_OPTIONS: Array<{
  value: Theme;
  key: "settings.themeLight" | "settings.themeDark" | "settings.themeSystem";
}> = [
  { value: "light", key: "settings.themeLight" },
  { value: "dark", key: "settings.themeDark" },
  { value: "system", key: "settings.themeSystem" },
];

export function SettingsDialog({
  current,
  modelsReady,
  embedding,
  reranker,
  onClose,
  onSaved,
  onModelsReady,
}: Props) {
  const { t, locale, setLocale } = useLocale();
  // v2.0.26.2 (PR-5) — pull the theme so the segmented control can
  // render the active state and so a click on any radio flips it
  // via useTheme() (which persists to ``rag.theme`` localStorage).
  const { theme, setTheme } = useTheme();
  const [provider, setProvider] = useState<string>(
    current?.llm_provider ?? "anthropic",
  );
  const [model, setModel] = useState<string>(
    current?.llm_model ?? "MiniMax-M3",
  );
  const [saving, setSaving] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiKeySource, setApiKeySource] = useState<string>("none");

  // v2.0.25 P0-F1 — close the dialog ONLY when modelsReady flips
  // false → true WHILE the dialog is open. A naive ``if
  // (modelsReady) onModelsReady()`` fires on every mount when
  // models are already ready (the normal case), which would close
  // the dialog immediately after the user opens it — see PR-5's
  // Layer 5 mobile.spec.ts Assertion 5 catching this. The ref
  // captures the previous value so we only act on a transition.
  const prevModelsReadyRef = useRef<boolean>(modelsReady);
  useEffect(() => {
    if (modelsReady && !prevModelsReadyRef.current) {
      onModelsReady?.();
    }
    prevModelsReadyRef.current = modelsReady;
  }, [modelsReady, onModelsReady]);
  // now come from App (which owns the shared ``useModelsReady``
  // hook). The hook is the single polling source; opening the
  // settings dialog no longer spawns a second 5 s poll loop.
  // ``apiKeySource`` is still fetched locally because the hook only
  // exposes a boolean (``apiKeyConfigured``) — we need the literal
  // string to drive the colored banner.

  // Re-fetch to learn the latest provider/model + api key source
  useEffect(() => {
    getSettings()
      .then((s) => {
        setProvider(s.llm_provider);
        setModel(s.llm_model);
        setApiKeySource(s.api_key_source);
      })
      .catch((e) => setError((e as Error).message));
  }, []);

  const modelOptions = provider === "openai" ? OPENAI_MODELS : ANTHROPIC_MODELS;
  const embStatus = (embedding?.status as ModelStatus) ?? "pending";
  const rerStatus = (reranker?.status as ModelStatus) ?? "pending";
  const bothDownloading =
    (embStatus === "downloading" || embStatus === "pending") &&
    (rerStatus === "downloading" || rerStatus === "pending");

  async function handleSave() {
    setSaving(true);
    setError(null);
    try {
      const s = await updateSettings({ llm_provider: provider, llm_model: model });
      onSaved(s);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function handleDownload() {
    setDownloading(true);
    setError(null);
    // v2.0.25 P0-F1 — the local ``modelStatus`` state was removed;
    // the shared hook now owns it. We don't optimistically flip the
    // UI to "downloading" because the hook will catch up within ~2 s
    // of its next poll tick; flipping locally would create a
    // brief inconsistency if the user already had ``ready`` showing.
    try {
      await triggerModelDownload();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>{t("settings.title")}</h2>

        <div
          className={`settings-key-banner settings-key-banner-${apiKeySource}`}
        >
          {apiKeySource === "env" ? (
            <>
              <span aria-hidden="true">✓</span>{" "}
              {t("settings.apiKeyEnv", { name: "MINIMAX_API_KEY" })}
            </>
          ) : apiKeySource === "keyring" ? (
            <>
              <span aria-hidden="true">✓</span>{" "}
              {t("settings.apiKeyKeyring")}
            </>
          ) : (
            <>
              <span aria-hidden="true">⚠</span>{" "}
              {t("settings.apiKeyMissing", { name: "MINIMAX_API_KEY" })}
            </>
          )}
        </div>

        <label className="form-row-label">{t("settings.provider")}</label>
        <select
          value={provider}
          onChange={(e) => {
            setProvider(e.target.value);
            setModel(
              e.target.value === "anthropic" ? "MiniMax-M3" : "gpt-4o-mini",
            );
          }}
        >
          <option value="anthropic">{t("settings.providerAnthropic")}</option>
          <option value="openai">OpenAI</option>
        </select>

        <label className="form-row-label">{t("settings.model")}</label>
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          {modelOptions.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>

        {/* v2.0.6 P2-4 — language picker lives in its own section so
            it doesn't get lost among the provider/model rows. The
            segmented control reflects the active choice on every
            render so the user always sees what's currently in
            effect. */}
        <div className="settings-section">
          <div className="settings-section-header">
            <div className="settings-section-title-sm">
              {t("settings.theme")}
            </div>
          </div>
          <div className="segmented" role="radiogroup" aria-label={t("settings.theme")}>
            {THEME_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                type="button"
                role="radio"
                aria-checked={theme === opt.value}
                data-active={theme === opt.value}
                className="segmented-option"
                onClick={() => setTheme(opt.value)}
              >
                {t(opt.key)}
              </button>
            ))}
          </div>
        </div>
        <div className="settings-section">
          <div className="settings-section-header">
            <div className="settings-section-title-sm">
              {t("settings.language")}
            </div>
          </div>
          <div className="segmented" role="radiogroup" aria-label={t("settings.language")}>
            {LANG_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                type="button"
                role="radio"
                aria-checked={locale === opt.value}
                data-active={locale === opt.value}
                className="segmented-option"
                onClick={() => setLocale(opt.value)}
              >
                {t(opt.key)}
              </button>
            ))}
          </div>
        </div>

        {/* Model download panel */}
        <div className="settings-models-panel">
          <div className="settings-section-title">{t("settings.models")}</div>
          <ModelRow name={t("settings.embeddingModel")} status={embStatus} />
          <ModelRow name={t("settings.rerankerModel")} status={rerStatus} />
          <button
            className="btn-secondary"
            onClick={handleDownload}
            disabled={downloading}
          >
            {downloading || bothDownloading
              ? t("settings.downloading")
              : t("settings.downloadModels")}
          </button>
        </div>

        {error && <div className="error">{error}</div>}

        <div className="modal-actions">
          <button className="btn-secondary" onClick={onClose}>
            {t("settings.close")}
          </button>
          <button className="btn-primary" onClick={handleSave} disabled={saving}>
            {saving ? t("settings.saving") : t("settings.save")}
          </button>
        </div>

        <p className="modal-footnote">{t("settings.footnote")}</p>
      </div>
    </div>
  );
}

function ModelRow({ name, status }: { name: string; status: ModelStatus }) {
  // PR-4 (v2.0.26.1): use ``statusLabel(t, ...)`` so the label flips
  // when SettingsDialog toggles the language. The legacy
  // ``STATUS_LABEL`` record is kept for backward compatibility
  // (tests / unknown consumers) but new code threads ``t``.
  const { t } = useLocale();
  const label = statusLabel(t, status);
  const color =
    status === "ready"
      ? "var(--success)"
      : status === "downloading"
        ? "var(--warning)"
        : "var(--text-dim)";
  return (
    <div className="settings-model-row">
      <span>{name}</span>
      <span style={{ color, fontSize: 12 }}>{label}</span>
    </div>
  );
}
