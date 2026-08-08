import { useEffect, useRef, useState } from 'react'

interface Props {
  /** The target value to display. */
  value: number
  /** Render the (possibly mid-animation) numeric value as a string. */
  format?: (n: number) => string
  /** Count-up duration in ms. A spring has no fixed duration, but for a
   *  monotonic count-up an easeOut over a short window reads the same. */
  duration?: number
  /** Extra classes merged onto the wrapping span (e.g. green / red). */
  className?: string
  /** Flash the highlight on change. Direction (up/down) picks the colour. */
  flash?: boolean
}

/**
 * A number that counts up to its new value instead of snapping, and pulses a
 * short highlight when it changes. Apple's "behavior over animation" applied to
 * a finance dashboard: money should move, not teleport.
 *
 * Interruptibility (§3): a change that lands mid-count starts the next
 * animation from the value currently ON SCREEN (`displayRef`), never from the
 * previous logical target — so a burst of updates never produces a visible
 * jump back to an intermediate frame.
 *
 * Reduced motion (§14): when the user asks for less motion, the value is set
 * instantly and no highlight plays.
 */
export function AnimatedNumber({
  value,
  format = (n) => String(n),
  duration = 500,
  className = '',
  flash = true,
}: Props) {
  const [display, setDisplay] = useState(value)
  const [flashCls, setFlashCls] = useState('')

  // Live on-screen value — the presentation value we re-target from on interrupt.
  const displayRef = useRef(value)
  // Last committed target — the reference for change direction, so a burst of
  // frames doesn't flip the flash colour on every intermediate step.
  const prevTargetRef = useRef(value)
  const rafRef = useRef<number | undefined>(undefined)
  const flashTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  useEffect(() => {
    const to = value
    const from = displayRef.current
    const prevTarget = prevTargetRef.current
    prevTargetRef.current = to
    if (to === from && to === prevTarget) return

    const changed = to !== prevTarget
    if (flash && changed) {
      const cls = to > prevTarget ? 'flash-up' : 'flash-down'
      setFlashCls(cls)
      clearTimeout(flashTimerRef.current)
      flashTimerRef.current = setTimeout(() => setFlashCls(''), 700)
    }

    const reduce =
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches

    if (reduce || from === to) {
      displayRef.current = to
      setDisplay(to)
      return
    }

    const start = performance.now()
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration)
      const eased = 1 - Math.pow(1 - t, 3) // easeOut — critically-damped feel, no overshoot
      const v = from + (to - from) * eased
      displayRef.current = v
      setDisplay(v)
      if (t < 1) rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [value, duration, flash])

  useEffect(() => () => clearTimeout(flashTimerRef.current), [])

  return <span className={`animated-num ${flashCls} ${className}`.trim()}>{format(display)}</span>
}
