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
    <div className={`toast ${visible ? 'show' : ''} ${type}`}>
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
    </div>
  )
}
