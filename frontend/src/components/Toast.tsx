import type { ToastType } from '../hooks/useToast'

interface Props {
  msg: string
  type: ToastType
  visible: boolean
}

export function Toast({ msg, type, visible }: Props) {
  return <div className={`toast ${visible ? 'show' : ''} ${type}`}>{msg}</div>
}
