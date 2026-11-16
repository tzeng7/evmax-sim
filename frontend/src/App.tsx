import { useState, useEffect } from 'react'
import { useHashRoute, type Page } from './hooks/useHashRoute'
import { usePersistedState } from './hooks/usePersistedState'
import { useDashboard } from './hooks/useDashboard'
import { useToast } from './hooks/useToast'
import { KpiCards } from './components/KpiCards'
import { ProfitChart } from './components/ProfitChart'
import { ActionBar } from './components/ActionBar'
import { ScanResults } from './components/ScanResults'
import { PlacedBets } from './components/PlacedBets'
import { SectorPerformance } from './components/SectorPerformance'
import { OpenPositions } from './components/OpenPositions'
import { RestingOrders } from './components/RestingOrders'
import { RecentSettled } from './components/RecentSettled'
import { MetricsPage } from './components/MetricsPage'
import { PromotionBoard } from './components/PromotionBoard'
import { ProjectionsPage } from './components/ProjectionsPage'
import { EvalPage } from './components/EvalPage'
import { ArbPage } from './components/ArbPage'
import { PortfolioGrid } from './components/PortfolioGrid'
import { PortfolioDetail } from './components/PortfolioDetail'
import { ShortcutHelp } from './components/ShortcutHelp'
import { HealthChip } from './components/HealthChip'
import { Toast } from './components/Toast'
import { ActivityLog } from './components/ActivityLog'
import { ErrorBoundary } from './components/ErrorBoundary'
import './App.css'

function RefreshChip({ updatedAt, paused, onTogglePause, onRefresh }: {
  updatedAt: number | null; paused: boolean; onTogglePause: () => void; onRefresh: () => void
}) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [])
  const age = updatedAt === null ? null : Math.max(0, Math.round((Math.max(now, updatedAt) - updatedAt) / 1000))
  return (
    <span className="refresh-chip" data-testid="refresh-chip" style={{ display: 'inline-flex', gap: 6, alignItems: 'center', fontSize: 12, opacity: 0.8 }}>
      <span>{age === null ? 'Loading…' : `Updated ${age}s ago`}{paused ? ' (paused)' : ''}</span>
      <button className="seg" onClick={onRefresh} aria-label="Refresh now" title="Refresh now">↻</button>
      <button className="seg" onClick={onTogglePause} aria-pressed={paused} aria-label={paused ? 'Resume auto-refresh' : 'Pause auto-refresh'} title={paused ? 'Resume auto-refresh' : 'Pause auto-refresh'}>{paused ? '▶' : '⏸'}</button>
    </span>
  )
}

export default function App() {
  const [route, updateRoute] = useHashRoute()
  const { page, view } = route
  const setPage = (p: Page) => updateRoute({ page: p })
  const setView = (v: 'all' | 'placed') => updateRoute({ view: v })
  const dash = useDashboard(page.kind === 'dashboard')
  const { toast, activity, ...toastProps } = useToast()
  // Top-bar slot that ActionBar portals its Scan/Resolve verbs into, so the
  // primary actions sit with navigation rather than trailing the toolbar.
  const [actionsSlot, setActionsSlot] = useState<HTMLDivElement | null>(null)
  // Lifted to App so ScanResults can re-size stakes live when these change
  // without forcing a re-scan. ActionBar still owns the inputs.
  const [bankrollStr, setBankrollStr] = usePersistedState('bankroll', '250')
  const [kelly, setKelly] = usePersistedState('kelly', 0.5)
  const bankroll = Number(bankrollStr) || 0

  const summary = view === 'placed' ? dash.summaryPlaced : dash.summaryAll
  const loadingDash = dash.loading && page.kind === 'dashboard'

  return (
    <div className="app">
      <ShortcutHelp />
      <header className="topbar">
        <div className="topbar-bar">
          <div className="brand">
            <h1
              className="home-link"
              role="button"
              tabIndex={0}
              onClick={() => setPage({ kind: 'dashboard' })}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setPage({ kind: 'dashboard' }) } }}
            >evmax</h1>
            <span className="subtitle">+EV prediction market dashboard</span>
          </div>
          <div className="topbar-right">
            <HealthChip />
            <nav className="segmented" aria-label="Main">
              <button
                className={`seg ${page.kind === 'dashboard' ? 'active' : ''}`}
                aria-current={page.kind === 'dashboard' ? 'page' : undefined}
                onClick={() => setPage({ kind: 'dashboard' })}
              >
                Dashboard
              </button>
              <button
                className={`seg ${page.kind === 'metrics' ? 'active' : ''}`}
                aria-current={page.kind === 'metrics' ? 'page' : undefined}
                onClick={() => setPage({ kind: 'metrics' })}
              >
                Metrics
              </button>
              <button
                className={`seg ${page.kind === 'board' ? 'active' : ''}`}
                aria-current={page.kind === 'board' ? 'page' : undefined}
                onClick={() => setPage({ kind: 'board' })}
              >
                Board
              </button>
              <button
                className={`seg ${page.kind === 'eval' ? 'active' : ''}`}
                onClick={() => setPage({ kind: 'eval' })}
              >
                Eval
              </button>
              <button
                className={`seg ${page.kind === 'arb' ? 'active' : ''}`}
                aria-current={page.kind === 'arb' ? 'page' : undefined}
                onClick={() => setPage({ kind: 'arb' })}
              >
                Arb
              </button>
              <button
                className={`seg ${page.kind === 'projections' ? 'active' : ''}`}
                aria-current={page.kind === 'projections' ? 'page' : undefined}
                onClick={() => setPage({ kind: 'projections' })}
              >
                Projections
              </button>
              <button
                className={`seg ${page.kind === 'portfolios' || page.kind === 'portfolio' ? 'active' : ''}`}
                aria-current={page.kind === 'portfolios' || page.kind === 'portfolio' ? 'page' : undefined}
                onClick={() => setPage({ kind: 'portfolios' })}
              >
                Portfolios
              </button>
            </nav>
            <ActivityLog entries={activity} />
            {page.kind === 'dashboard' && (
              <>
                <span className="topbar-div" aria-hidden="true" />
                <div className="topbar-actions" ref={setActionsSlot} />
                <RefreshChip updatedAt={dash.updatedAt} paused={dash.paused} onTogglePause={() => dash.setPaused(!dash.paused)} onRefresh={() => { void dash.refresh() }} />
              </>
            )}
          </div>
        </div>
      </header>

      <div className="content">
      <ErrorBoundary>
        {/* The scan controls stay mounted while the dashboard data loads so the
            Scan / Resolve verbs in the top bar never flicker out. */}
        {page.kind === 'dashboard' && (
          <div className="scanbar">
            <ActionBar
              bankrollStr={bankrollStr}
              setBankrollStr={setBankrollStr}
              kelly={kelly}
              setKelly={setKelly}
              onScanComplete={dash.setScanResults}
              onResolve={dash.refresh}
              toast={toast}
              actionsSlot={actionsSlot}
            />
          </div>
        )}
        {loadingDash && <DashboardSkeleton />}
        {!loadingDash && page.kind === 'dashboard' && (
          <>
            {/* Workflow order: what to bet now -> what is on -> how it has gone. */}
            <ScanResults
              gaps={dash.scanGaps}
              meta={dash.scanMeta}
              scanAt={dash.scanAt}
              bankroll={bankroll}
              kelly={kelly}
              scanKelly={dash.scanKelly}
              toast={toast}
              onPicked={dash.refresh}
            />

            <RestingOrders toast={toast} onChanged={dash.refresh} />

            <PlacedBets bets={dash.placedBets} toast={toast} onChanged={dash.refresh} />

            <OpenPositions bets={dash.openBets} scanGaps={dash.scanGaps} bankroll={bankroll} kelly={kelly} scanKelly={dash.scanKelly} toast={toast} onPicked={dash.refresh} />

            <div className="section-head">
              <h2>Performance</h2>
              <div className="segmented" role="group" aria-label="Performance scope">
                <button aria-pressed={view === 'all'} className={`seg ${view === 'all' ? 'active' : ''}`} onClick={() => setView('all')}
                  title="Every logged live play, sized at its scan-time Kelly stake">All Scanned</button>
                <button aria-pressed={view === 'placed'} className={`seg ${view === 'placed' ? 'active' : ''}`} onClick={() => setView('placed')}
                  title="Only plays you confirmed with Pick, at your recorded fill">Placed Only</button>
              </div>
            </div>
            <KpiCards summary={summary} onOpenEval={() => setPage({ kind: 'eval' })} />
            <ProfitChart seriesAll={dash.seriesAll} seriesPlaced={dash.seriesPlaced} view={view} />

            <div className="two-col">
              <SectorPerformance sectors={dash.sectors} />
              <RecentSettled bets={dash.recent} />
            </div>
          </>
        )}

        {page.kind === 'metrics' && <MetricsPage toast={toast} onOpenEval={() => setPage({ kind: 'eval' })} />}

        {page.kind === 'board' && <PromotionBoard toast={toast} />}

        {page.kind === 'eval' && <EvalPage toast={toast} />}

        {page.kind === 'arb' && <ArbPage toast={toast} />}

        {page.kind === 'projections' && <ProjectionsPage toast={toast} />}

        {page.kind === 'portfolios' && (
          <PortfolioGrid
            onSelect={(id) => setPage({ kind: 'portfolio', id })}
            toast={toast}
          />
        )}

        {page.kind === 'portfolio' && (
          <PortfolioDetail
            portfolioId={page.id}
            onBack={() => setPage({ kind: 'portfolios' })}
            toast={toast}
          />
        )}
      </ErrorBoundary>
      </div>

      <Toast {...toastProps} />
    </div>
  )
}

function DashboardSkeleton() {
  return (
    <>
      <div className="skeleton" style={{ height: 150, marginBottom: 18 }} />
      <div className="skeleton" style={{ height: 200, marginBottom: 18 }} />
      <div className="grid-kpi">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="skeleton" style={{ height: 74 }} />
        ))}
      </div>
      <div className="skeleton" style={{ height: 320 }} />
    </>
  )
}
