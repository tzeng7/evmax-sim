import { useState } from 'react'
import { useDashboard } from './hooks/useDashboard'
import { useToast } from './hooks/useToast'
import { KpiCards } from './components/KpiCards'
import { ProfitChart } from './components/ProfitChart'
import { ActionBar } from './components/ActionBar'
import { ScanResults } from './components/ScanResults'
import { PlacedBets } from './components/PlacedBets'
import { SectorPerformance } from './components/SectorPerformance'
import { OpenPositions } from './components/OpenPositions'
import { RecentSettled } from './components/RecentSettled'
import { MetricsPage } from './components/MetricsPage'
import { PortfolioGrid } from './components/PortfolioGrid'
import { PortfolioDetail } from './components/PortfolioDetail'
import { Toast } from './components/Toast'
import { ErrorBoundary } from './components/ErrorBoundary'
import './App.css'

type Page = { kind: 'dashboard' } | { kind: 'metrics' } | { kind: 'portfolios' } | { kind: 'portfolio'; id: string }

export default function App() {
  const dash = useDashboard()
  const { toast, ...toastProps } = useToast()
  const [view, setView] = useState<'all' | 'placed'>('all')
  const [page, setPage] = useState<Page>({ kind: 'dashboard' })
  // Lifted to App so ScanResults can re-size stakes live when these change
  // without forcing a re-scan. ActionBar still owns the inputs.
  const [bankrollStr, setBankrollStr] = useState('250')
  const [kelly, setKelly] = useState(0.5)
  const bankroll = Number(bankrollStr) || 0

  const summary = view === 'placed' ? dash.summaryPlaced : dash.summaryAll
  const loadingDash = dash.loading && page.kind === 'dashboard'

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <h1 style={{ cursor: 'pointer' }} onClick={() => setPage({ kind: 'dashboard' })}>evmax</h1>
          <span className="subtitle">+EV prediction market dashboard</span>
        </div>
        <nav className="segmented">
          <button
            className={`seg ${page.kind === 'dashboard' ? 'active' : ''}`}
            onClick={() => setPage({ kind: 'dashboard' })}
          >
            Dashboard
          </button>
          <button
            className={`seg ${page.kind === 'metrics' ? 'active' : ''}`}
            onClick={() => setPage({ kind: 'metrics' })}
          >
            Metrics
          </button>
          <button
            className={`seg ${page.kind === 'portfolios' || page.kind === 'portfolio' ? 'active' : ''}`}
            onClick={() => setPage({ kind: 'portfolios' })}
          >
            Portfolios
          </button>
        </nav>
        <div style={{ flex: 1 }} />
        {page.kind === 'dashboard' && (
          <ActionBar
            bankrollStr={bankrollStr}
            setBankrollStr={setBankrollStr}
            kelly={kelly}
            setKelly={setKelly}
            onScanComplete={dash.setScanResults}
            onResolve={dash.refresh}
            toast={toast}
          />
        )}
      </header>

      <div className="content">
      <ErrorBoundary>
        {loadingDash && <DashboardSkeleton />}
        {!loadingDash && page.kind === 'dashboard' && (
          <>
            <div className="segmented" style={{ marginBottom: 16 }}>
              <button className={`seg ${view === 'all' ? 'active' : ''}`} onClick={() => setView('all')}>All Scanned</button>
              <button className={`seg ${view === 'placed' ? 'active' : ''}`} onClick={() => setView('placed')}>Placed Only</button>
            </div>

            <KpiCards summary={summary} />
            <ProfitChart seriesAll={dash.seriesAll} seriesPlaced={dash.seriesPlaced} view={view} />

            <ScanResults
              gaps={dash.scanGaps}
              meta={dash.scanMeta}
              bankroll={bankroll}
              kelly={kelly}
              scanKelly={dash.scanKelly}
              toast={toast}
              onPicked={dash.refresh}
            />

            <PlacedBets bets={dash.placedBets} toast={toast} onChanged={dash.refresh} />

            <div className="two-col">
              <SectorPerformance sectors={dash.sectors} />
              <OpenPositions bets={dash.openBets} scanGaps={dash.scanGaps} bankroll={bankroll} kelly={kelly} toast={toast} onPicked={dash.refresh} />
            </div>

            <RecentSettled bets={dash.recent} />
          </>
        )}

        {page.kind === 'metrics' && <MetricsPage toast={toast} />}

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
      <div className="grid-kpi">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="skeleton" style={{ height: 74 }} />
        ))}
      </div>
      <div className="skeleton" style={{ height: 320, marginBottom: 18 }} />
      <div className="skeleton" style={{ height: 200 }} />
    </>
  )
}
