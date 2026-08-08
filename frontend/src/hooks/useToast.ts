import { useState, useCallback, useRef } from 'react'

export type ToastType = 'info' | 'ok' | 'err'

/** An optional action button rendered inside the toast (e.g. Undo). */
export interface ToastAction {
  label: string
  onClick: () => void
}

export interface ToastOptions {
  /** Auto-dismiss delay in ms. Defaults to 3000, or 6000 when an action is
   *  present so the user has time to reach for it. */
  ms?: number
  action?: ToastAction
}

export function useToast() {
  const [msg, setMsg] = useState('')
  const [type, setType] = useState<ToastType>('info')
  const [visible, setVisible] = useState(false)
  const [action, setAction] = useState<ToastAction | undefined>(undefined)
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  const toast = useCallback((message: string, toastType: ToastType = 'info', opts?: ToastOptions) => {
    setMsg(message)
    setType(toastType)
    setAction(opts?.action)
    setVisible(true)
    clearTimeout(timer.current)
    const ms = opts?.ms ?? (opts?.action ? 6000 : 3000)
    timer.current = setTimeout(() => setVisible(false), ms)
  }, [])

  const dismiss = useCallback(() => {
    clearTimeout(timer.current)
    setVisible(false)
  }, [])

  return { msg, type, visible, action, toast, dismiss }
}
