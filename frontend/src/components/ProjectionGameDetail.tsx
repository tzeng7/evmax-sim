import { useEffect, useState } from 'react'
import type { ProjectionGameDetail as Detail, ProjectionOption, ProjectionOptionValues, ProjectionSection } from '../lib/types'
import { ProjectionOptionFields } from './ProjectionOptionFields'
import { ProjectionPlayerTable } from './ProjectionTables'

/** Seconds since `since`, ticking while mounted — feedback for runs that take a while. */
export function RunProgress({ since, label }: { since: number; label: string }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 100)
    return () => window.clearInterval(id)
  }, [])
  return (
    <div className="proj-progress" role="status" aria-live="polite">
      <span className="proj-spinner" aria-hidden="true" />
      {label} <span className="num muted">{((now - since) / 1000).toFixed(1)} s</span>
    </div>
  )
}

export interface GameRunState {
  loading: boolean
  startedAt?: number
  data?: Detail
  error?: string
}

interface Props {
  specs: ProjectionOption[]
  values: ProjectionOptionValues
  onValues: (v: ProjectionOptionValues) => void
  state: GameRunState | undefined
  onRun: () => void
  runLabel: string
}

function Section({ s }: { s: ProjectionSection }) {
  if (s.kind === 'kv') {
    return (
      <dl className="proj-kv">
        {s.items.map(it => (
          <div key={it.label}>
            <dt>{it.label}</dt>
            <dd>{it.value}</dd>
          </div>
        ))}
      </dl>
    )
  }
  if (s.kind === 'players') {
    return <ProjectionPlayerTable rows={s.rows} columns={s.columns} showEvent={false} />
  }
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>{s.columns.map(c => <th key={c.key} className={c.align === 'right' ? 'num' : undefined}>{c.label}</th>)}</tr>
        </thead>
        <tbody>
          {s.rows.map((r, i) => (
            <tr key={i}>
              {s.columns.map(c => <td key={c.key} className={c.align === 'right' ? 'num' : undefined}>{r[c.key]}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** One game's per-game model run: its options, a re-run button, and the engine's result sections. */
export function ProjectionGameDetail({ specs, values, onValues, state, onRun, runLabel }: Props) {
  const loading = !!state?.loading
  return (
    <div className="proj-detail">
      <div className="proj-controls">
        {state?.data && <strong className="proj-detail-title">{state.data.title}</strong>}
        <div style={{ flex: 1 }} />
        <ProjectionOptionFields specs={specs} values={values} onChange={onValues} disabled={loading} />
        <button className="btn btn-sm primary" onClick={onRun} disabled={loading}>
          {loading ? 'Running…' : `Re-${runLabel.toLowerCase()}`}
        </button>
      </div>

      {loading && state?.startedAt != null && <RunProgress since={state.startedAt} label="Running the per-game model…" />}
      {!loading && state?.error && <p className="red" style={{ margin: '12px 0 0' }}>{state.error}</p>}

      {!loading && state?.data && (
        <>
          {state.data.sections.map((s, i) => (
            <div key={i} className="proj-section">
              <h3>{s.title}</h3>
              <Section s={s} />
            </div>
          ))}
          {state.data.notes.length > 0 && (
            <ul className="proj-lines" style={{ marginTop: 12, marginBottom: 0 }}>
              {state.data.notes.map(n => <li key={n}>{n}</li>)}
            </ul>
          )}
        </>
      )}
    </div>
  )
}
