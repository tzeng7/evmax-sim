import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
  info: ErrorInfo | null
}

// Catches render exceptions so a single bad row in any child (e.g. a bet with
// a missing kalshi_yes_price triggering NaN math after Pick) doesn't blank the
// entire dashboard. Before this, an unhandled throw bubbled to the root and
// React unmounted everything — same URL, black screen, no console hint
// unless the user already had devtools open.
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, info: null }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Dashboard render crashed:', error, info)
    this.setState({ error, info })
  }

  handleReset = () => {
    this.setState({ error: null, info: null })
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div style={{ padding: 24, color: '#e0e0e0', maxWidth: 920, margin: '40px auto' }}>
        <h2 style={{ color: '#f87171', marginTop: 0 }}>Dashboard render failed</h2>
        <p style={{ color: '#9ca3af' }}>
          A component threw while rendering. The error message is below; check the
          browser console for the full stack. The most recent action (e.g. Pick
          Selected) probably succeeded server-side — refresh to reload state.
        </p>
        <pre
          style={{
            background: '#111827',
            padding: 12,
            borderRadius: 6,
            border: '1px solid #1f2937',
            color: '#fca5a5',
            overflow: 'auto',
            fontSize: 12,
          }}
        >
{this.state.error.name}: {this.state.error.message}
{'\n\n'}
{this.state.error.stack || '(no stack)'}
        </pre>
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <button className="btn primary" onClick={this.handleReset}>Try Again</button>
          <button className="btn" onClick={() => window.location.reload()}>Hard Reload</button>
        </div>
      </div>
    )
  }
}
