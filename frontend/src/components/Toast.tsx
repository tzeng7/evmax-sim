import type { ToastAction, ToastType } from '../hooks/useToast'

interface Props {
  msg: string
  type: ToastType
  visible: boolean
  action?: ToastAction
  dismiss?: () => void
}

export function Toast({ msg, type, visible, action, dismiss }: Props) {
  return (
    <div className={`toast ${visible ? 'show' : ''} ${type}`} role={type === 'err' ? 'alert' : 'status'} aria-live="polite">
      <span>{msg}</span>
      {action && (
        <button
          type="button"
          className="toast-action"
          onClick={() => { action.onClick(); dismiss?.() }}
        >
          {action.label}
        </button>
      )}
      {dismiss && (
        <button type="button" className="toast-close" aria-label="Dismiss" onClick={dismiss}>&times;</button>
      )}
    </div>
  )
}
