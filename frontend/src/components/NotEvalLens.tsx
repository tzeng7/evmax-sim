/** Caption: these cards are a dashboard lens, not `evmax cleanup eval`. */
export function NotEvalLens({ onOpenEval }: { onOpenEval?: () => void }) {
  return (
    <p className="muted" style={{ fontSize: 12, margin: '6px 0 12px' }}>
      Dashboard lens — not the evaluation population.{' '}
      {onOpenEval ? (
        <a href="#eval" onClick={e => { e.preventDefault(); onOpenEval() }}>
          Open the Eval tab
        </a>
      ) : 'See the Eval tab'}{' '}
      for the funnel-defined ROI / CLV / Brier numbers to quote.
    </p>
  )
}
