/**
 * Yuan RAG logo. Inline SVG so we can theme it via CSS without an
 * external asset pipeline, and reuse at every size — favicon (16×16),
 * sidebar header (28×28), settings/about (larger).
 *
 * Design: rounded-square with a clean blue→sky gradient and a small
 * light highlight for subtle depth. Kept geometric so it stays legible
 * at the favicon size where the "源" Chinese character would blur on
 * systems missing CJK fallbacks.
 */
interface Props {
  size?: number;
  className?: string;
}

export function Logo({ size = 28, className }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="Yuan RAG"
    >
      <defs>
        <linearGradient id="yuanRagGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#2563eb" />
          <stop offset="100%" stopColor="#3b82f6" />
        </linearGradient>
        <radialGradient id="yuanRagGlow" cx="50%" cy="50%" r="60%">
          <stop offset="0%" stopColor="rgba(255,255,255,0.22)" />
          <stop offset="100%" stopColor="rgba(255,255,255,0)" />
        </radialGradient>
      </defs>
      <rect width="64" height="64" rx="14" fill="url(#yuanRagGradient)" />
      <rect width="64" height="64" rx="14" fill="url(#yuanRagGlow)" />
      {/* Stylized "Y" — two diagonal strokes meeting at the center,
          plus a vertical stem reaching the bottom. */}
      <path d="M 18 14 L 30 32 L 24 32 L 16 18 Z" fill="white" />
      <path d="M 46 14 L 34 32 L 40 32 L 48 18 Z" fill="white" />
      <rect x="29" y="28" width="6" height="22" fill="white" />
      {/* Small white dot — replaces the warm coral accent from the
          previous palette; the new theme uses blue + white only. */}
      <circle cx="50" cy="48" r="3.2" fill="white" opacity="0.85" />
    </svg>
  );
}