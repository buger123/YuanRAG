/**
 * Custom in-app confirmation dialog.
 *
 * Replaces the native ``window.confirm()`` calls used for deleting
 * documents and sessions. Native confirm dialogs:
 *
 *   - are ugly (browser-chrome styled, OS-dependent)
 *   - can't be themed to match the dark UI
 *   - are blocking and freeze the page
 *   - don't allow Chinese-aware copy or multi-option UX
 *
 * This component renders inside the existing modal styling, supports
 * a primary destructive action + optional secondary action, and is
 * fully localizable. Used by Sidebar for delete confirmations and by
 * the upload-error fallback.
 */
import { useEffect } from "react";
import { useLocale } from "../i18n";

export interface ConfirmDialogProps {
  title: string;
  message: React.ReactNode;
  /** Primary (destructive) button label. */
  confirmLabel?: string;
  /** Secondary button label. If omitted, only the confirm button shows. */
  cancelLabel?: string;
  /** Tone of the primary button. */
  variant?: "danger" | "warning" | "primary";
  /** When provided, renders a second action next to Cancel.
   *  Used for "delete session + keep documents" vs "delete both". */
  secondaryAction?: {
    label: string;
    onClick: () => void | Promise<void>;
    variant?: "danger" | "warning" | "primary";
  };
  onConfirm: () => void | Promise<void>;
  onCancel: () => void;
}

export function ConfirmDialog({
  title,
  message,
  confirmLabel,
  cancelLabel,
  variant = "danger",
  secondaryAction,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  // v2.0.7 i18n-1 — translate the default button labels. Callers
  // that pass an explicit label still win (those are usually the
  // domain-specific labels like "Delete document").
  const { t } = useLocale();
  const resolvedConfirmLabel = confirmLabel ?? t("confirm.confirmDefault");
  const resolvedCancelLabel = cancelLabel ?? t("confirm.cancelDefault");
  // Allow Escape to dismiss without taking the destructive action.
  // Without this, a misclick can only be escaped by clicking Cancel —
  // not great when the whole dialog is dark and the Cancel button is
  // visually similar to the destructive one.
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onCancel();
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onCancel]);

  const primaryClass =
    variant === "danger" ? "btn-danger" : "btn-primary";

  return (
    <div className="modal-backdrop" onClick={onCancel}>
      <div
        className="modal modal-confirm"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <div className="modal-confirm-icon" aria-hidden="true">
          {variant === "danger" ? "⚠" : "ℹ"}
        </div>
        <h2 className="modal-confirm-title">{title}</h2>
        <div className="modal-confirm-message">{message}</div>
        <div className="modal-actions modal-confirm-actions">
          <button className="btn-secondary" onClick={onCancel}>
            {resolvedCancelLabel}
          </button>
          {secondaryAction && (
            <button
              className={
                secondaryAction.variant === "danger"
                  ? "btn-danger"
                  : "btn-primary"
              }
              onClick={secondaryAction.onClick}
            >
              {secondaryAction.label}
            </button>
          )}
          <button className={primaryClass} onClick={onConfirm}>
            {resolvedConfirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}