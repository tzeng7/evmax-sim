/**
 * VenueLogo — small inline company logo identifying which prediction-market
 * venue a play was quoted on (Kalshi vs Polymarket US).
 *
 * Logos are simplified brand marks drawn as inline SVG so the dashboard
 * stays fully self-contained (no external image fetches). Every logo
 * carries a title tooltip + aria-label; unknown venues fall back to a
 * neutral text badge. Rows predating the venue column render as Kalshi
 * (every pre-multi-venue row was a Kalshi row).
 */

const SIZE = 14

function KalshiMark() {
  // Kalshi: green rounded-square "K" mark.
  return (
    <svg
      width={SIZE}
      height={SIZE}
      viewBox="0 0 24 24"
      role="img"
      aria-label="Kalshi"
      style={{ display: 'block' }}
    >
      <rect x="0" y="0" width="24" height="24" rx="5" fill="#00d991" />
      <path
        d="M7 4.5h3.4v6.2l5.2-6.2h4l-5.9 7 6.2 8h-4.2l-4.3-5.8-1 1.2v4.6H7z"
        fill="#0b1220"
      />
    </svg>
  )
}

function PolymarketUSMark() {
  // Polymarket: blue rounded-square with the angular "P" monogram.
  return (
    <svg
      width={SIZE}
      height={SIZE}
      viewBox="0 0 24 24"
      role="img"
      aria-label="Polymarket US"
      style={{ display: 'block' }}
    >
      <rect x="0" y="0" width="24" height="24" rx="5" fill="#2d5bff" />
      <path
        d="M7 4.5h7.2c2.9 0 4.8 1.9 4.8 4.6s-1.9 4.6-4.8 4.6h-3.8v5.8H7zm3.4 6.4h3.3c1.1 0 1.8-.7 1.8-1.8s-.7-1.8-1.8-1.8h-3.3z"
        fill="#ffffff"
      />
    </svg>
  )
}

const VENUE_NAMES: Record<string, string> = {
  kalshi: 'Kalshi',
  polymarket_us: 'Polymarket US',
}

export function VenueLogo({ venue }: { venue?: string | null }) {
  const v = venue || 'kalshi'
  const name = VENUE_NAMES[v] || v
  return (
    <span
      className="venue-logo"
      title={name}
      style={{ display: 'inline-flex', alignItems: 'center', verticalAlign: 'middle' }}
    >
      {v === 'kalshi' ? (
        <KalshiMark />
      ) : v === 'polymarket_us' ? (
        <PolymarketUSMark />
      ) : (
        <span className="badge">{name}</span>
      )}
    </span>
  )
}
