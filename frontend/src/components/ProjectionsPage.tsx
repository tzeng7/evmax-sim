import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchProjectionSectors, fetchStoredProjections, runProjectionGame, runProjectionSlate } from '../lib/api'
import type { ProjectionGame, ProjectionPeriod, ProjectionSector, ProjectionSlate } from '../lib/types'
import { optionDefaults } from '../lib/projections'
import { ProjectionOptionFields } from './ProjectionOptionFields'
import { ProjectionGamesTable, ProjectionPlayerTable } from './ProjectionTables'
import { ProjectionGameDetail, RunProgress, type GameRunState } from './ProjectionGameDetail'

type Toast = (msg: string, type?: 'info' | 'ok' | 'err') => void

const SAVED_SECTOR = 'evmax.projections.sector'

function readSavedSector(): string | null {
  try { return localStorage.getItem(SAVED_SECTOR) } catch { return null }
}

function saveSector(key: string) {
  try { localStorage.setItem(SAVED_SECTOR, key) } catch { /* storage blocked: the default sector is fine */ }
}

/**
 * Projections tab — one entrypoint for every sector's projection model.
 *
 * The sector list, each sector's run options and its capabilities come from
 * GET /api/projections/sectors (data/projections.yaml + the engine's declared
 * options), so adding a sector is a backend change only. Projections are a
 * standalone product: the models read no market price, and the market line is
 * shown for comparison.
 */
export function ProjectionsPage({ toast }: { toast: Toast }) {
  const [sectors, setSectors] = useState<ProjectionSector[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [active, setActive] = useState<string | null>(null)
  // Panels mount on first visit and stay mounted, so switching sectors keeps results.
  const [visited, setVisited] = useState<string[]>([])

  useEffect(() => {
    let alive = true
    fetchProjectionSectors()
      .then(list => {
        if (!alive) return
        setSectors(list)
        const saved = readSavedSector()
        const first = list.find(s => s.key === saved) ?? list.find(s => s.status === 'available') ?? list[0]
        if (first) {
          setActive(first.key)
          setVisited([first.key])
        }
      })
      .catch(e => { if (alive) setError((e as Error).message) })
    return () => { alive = false }
  }, [])

  const select = (key: string) => {
    setActive(key)
    setVisited(v => (v.includes(key) ? v : [...v, key]))
    saveSector(key)
  }

  return (
    <>
      <div className="proj-head">
        <h2 style={{ margin: 0 }}>Projections</h2>
        <span className="muted">Run each sector's projection model, then any game's deeper model</span>
      </div>

      {error && (
        <div className="panel">
          <p className="red" style={{ margin: 0 }}>Could not load the projection sectors: {error}</p>
        </div>
      )}
      {!sectors && !error && <div className="skeleton" style={{ height: 140 }} />}

      {sectors && (
        <nav className="segmented proj-sectors" aria-label="Projection sector">
          {sectors.map(s => (
            <button
              key={s.key}
              className={`seg ${active === s.key ? 'active' : ''} ${s.status === 'planned' ? 'planned' : ''}`}
              aria-pressed={active === s.key}
              title={s.status === 'planned' ? s.note : s.description}
              onClick={() => select(s.key)}
            >
              {s.label}
              {s.status === 'planned' && <span className="seg-tag">soon</span>}
            </button>
          ))}
        </nav>
      )}

      {sectors?.filter(s => visited.includes(s.key)).map(s => (
        <div key={s.key} hidden={s.key !== active}>
          {s.status === 'available'
            ? <SectorPanel sector={s} toast={toast} />
            : (
              <div className="panel">
                <div className="empty-state">
                  <div className="empty-title">{s.label} projections are planned</div>
                  <div style={{ maxWidth: 520 }}>{s.note}</div>
                </div>
              </div>
            )}
        </div>
      ))}
    </>
  )
}

function SectorPanel({ sector, toast }: { sector: ProjectionSector; toast: Toast }) {
  const slateSpecs = sector.slate_options ?? []
  const gameSpecs = sector.game_options ?? []
  const canStore = !!sector.capabilities?.stored
  const canRunGame = !!sector.capabilities?.game_run
  const runLabel = sector.game_run_label ?? 'Run'

  const [options, setOptions] = useState(() => optionDefaults(slateSpecs))
  const [gameOptions, setGameOptions] = useState(() => optionDefaults(gameSpecs))
  const [slate, setSlate] = useState<ProjectionSlate | null>(null)
  // The latest run stays selectable after the user switches to a stored week.
  const [lastRun, setLastRun] = useState<ProjectionSlate | null>(null)
  const [periods, setPeriods] = useState<ProjectionPeriod[]>([])
  const [busy, setBusy] = useState<{ kind: 'run' | 'stored'; since: number } | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [gameRuns, setGameRuns] = useState<Record<string, GameRunState>>({})
  // Bumped per slate request: a slower earlier request must not overwrite a newer one.
  const slateReq = useRef(0)
  // Bumped per slate SHOWN: a per-game result belongs to the slate it was run against. Keyed
  // on what is displayed, not on requests, so a failed newer request cannot strand a game run.
  const shownSlate = useRef(0)

  const showSlate = useCallback((res: ProjectionSlate) => {
    shownSlate.current++
    setSlate(res)
    setExpanded(null)
    setGameRuns({})
  }, [])

  const loadStored = useCallback(async (params: Record<string, string | number> = {}) => {
    const id = ++slateReq.current
    setBusy({ kind: 'stored', since: Date.now() })
    try {
      const res = await fetchStoredProjections(sector.key, params)
      if (id !== slateReq.current) return
      showSlate(res)
      setPeriods(res.periods)
    } catch (e) {
      if (id === slateReq.current) toast(`${sector.label} stored projections failed: ${(e as Error).message}`, 'err')
    } finally {
      if (id === slateReq.current) setBusy(null)
    }
  }, [sector.key, sector.label, showSlate, toast])

  useEffect(() => { if (canStore) loadStored() }, [canStore, loadStored])

  const run = async () => {
    const id = ++slateReq.current
    setBusy({ kind: 'run', since: Date.now() })
    try {
      const res = await runProjectionSlate(sector.key, options)
      if (id !== slateReq.current) return
      showSlate(res)
      setLastRun(res)
      toast(`${res.title}: ${res.games.length} game${res.games.length === 1 ? '' : 's'} projected`
        + (res.elapsed_s != null ? ` in ${res.elapsed_s.toFixed(1)} s` : ''), 'ok')
      // A stored run adds a week to the picker; refresh the list without leaving this run's view.
      if (canStore && res.options?.store) {
        fetchStoredProjections(sector.key).then(r => setPeriods(r.periods)).catch(() => {})
      }
    } catch (e) {
      if (id === slateReq.current) toast(`${sector.label} run failed: ${(e as Error).message}`, 'err')
    } finally {
      if (id === slateReq.current) setBusy(null)
    }
  }

  const runGame = async (g: ProjectionGame) => {
    const gen = shownSlate.current
    setGameRuns(m => ({ ...m, [g.game_id]: { ...m[g.game_id], loading: true, startedAt: Date.now(), error: undefined } }))
    try {
      const data = await runProjectionGame(sector.key, g, gameOptions)
      if (gen === shownSlate.current) setGameRuns(m => ({ ...m, [g.game_id]: { loading: false, data } }))
    } catch (e) {
      if (gen === shownSlate.current) setGameRuns(m => ({ ...m, [g.game_id]: { loading: false, error: (e as Error).message } }))
    }
  }

  const toggleGame = (g: ProjectionGame) => {
    if (expanded === g.game_id) {
      setExpanded(null)
      return
    }
    setExpanded(g.game_id)
    const st = gameRuns[g.game_id]
    if (!st?.data && !st?.loading) runGame(g)
  }

  const periodValue = slate?.source === 'stored' ? (slate.period ?? '') : 'run'
  const hasGames = !!slate && slate.games.length > 0

  return (
    <>
      <div className="panel proj-run">
        <div className="proj-run-head">
          <h2 style={{ margin: 0 }}>{sector.label}</h2>
          {sector.description && <p className="muted" style={{ margin: 0 }}>{sector.description}</p>}
        </div>
        <div className="proj-controls">
          <ProjectionOptionFields specs={slateSpecs} values={options} onChange={setOptions} disabled={!!busy} />
          <div className="proj-actions">
            {canStore && periods.length > 0 && (
              <select
                value={periodValue}
                aria-label="Projections to show"
                disabled={!!busy}
                onChange={e => {
                  if (e.target.value === 'run') {
                    if (lastRun) showSlate(lastRun)
                    return
                  }
                  const p = periods.find(x => x.key === e.target.value)
                  if (p) loadStored(p.params)
                }}
              >
                {lastRun && <option value="run">This run · {lastRun.title}</option>}
                {periods.map(p => <option key={p.key} value={p.key}>Stored · {p.label}</option>)}
              </select>
            )}
            <button className="btn primary" onClick={run} disabled={!!busy}>
              {busy?.kind === 'run' ? 'Running…' : 'Run model'}
            </button>
          </div>
        </div>
      </div>

      {busy && (
        <RunProgress
          since={busy.since}
          label={busy.kind === 'run' ? `Running the ${sector.label} model…` : 'Loading stored projections…'}
        />
      )}
      {busy && !slate && <div className="skeleton" style={{ height: 260 }} />}

      {slate && (
        <div className={busy ? 'proj-stale' : undefined} aria-busy={!!busy}>
          {(slate.summary.length > 0 || (hasGames && slate.notes.length > 0)) && (
            <ul className="proj-lines">
              {slate.summary.map(s => <li key={s}>{s}</li>)}
              {hasGames && slate.notes.map(n => <li key={n} className="proj-note">{n}</li>)}
            </ul>
          )}

          <div className="panel">
            <div className="panel-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <h2>{slate.title}</h2>
                <span className={`badge ${slate.source === 'run' ? 'new' : ''}`}>{slate.source === 'stored' ? 'Stored' : 'This run'}</span>
              </div>
              <span className="muted" style={{ fontSize: 12 }}>
                {slate.games.length} game{slate.games.length === 1 ? '' : 's'}
                {slate.source === 'run' && slate.elapsed_s != null && ` · ${slate.elapsed_s.toFixed(1)} s`}
              </span>
            </div>
            {!hasGames
              ? (
                <div className="empty-state">
                  <div className="empty-title">No games</div>
                  {(slate.notes.length > 0 ? slate.notes : ['Nothing to project for this slate.']).map(n => (
                    <div key={n} style={{ maxWidth: 560 }}>{n}</div>
                  ))}
                </div>
              )
              : (
                <ProjectionGamesTable
                  games={slate.games}
                  runLabel={runLabel}
                  expanded={expanded}
                  onToggle={canRunGame ? toggleGame : undefined}
                  renderDetail={g => (
                    <ProjectionGameDetail
                      specs={gameSpecs}
                      values={gameOptions}
                      onValues={setGameOptions}
                      state={gameRuns[g.game_id]}
                      onRun={() => runGame(g)}
                      runLabel={runLabel}
                    />
                  )}
                />
              )}
          </div>

          {slate.players && slate.players.length > 0 && (
            <div className="panel">
              <div className="panel-header"><h2>Players</h2></div>
              <ProjectionPlayerTable
                key={`${slate.source}-${slate.title}`}
                rows={slate.players}
                columns={slate.player_columns}
                controls
                defaultSort={slate.player_sort}
              />
            </div>
          )}

          {slate.footnote && <p className="muted proj-footnote">{slate.footnote}</p>}
        </div>
      )}

      {!slate && !busy && (
        <div className="panel">
          <div className="empty-state">
            <div className="empty-title">No projections yet</div>
            <div style={{ maxWidth: 440 }}>Run the model to project {sector.label}'s next slate.</div>
          </div>
        </div>
      )}
    </>
  )
}
