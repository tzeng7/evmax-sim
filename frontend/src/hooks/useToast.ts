import { useState, useCallback, useRef } from 'react'

export type ToastType = 'info' | 'ok' | 'err'

export function useToast() {
  const [msg, setMsg] = useState('')
  const [type, setType] = useState<ToastType>('info')
  const [visible, setVisible] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  const toast = useCallback((message: string, toastType: ToastType = 'info', ms = 3000) => {
    setMsg(message)
    setType(toastType)
    setVisible(true)
    clearTimeout(timer.current)
    timer.current = setTimeout(() => setVisible(false), ms)
  }, [])

  return { msg, type, visible, toast }
}
