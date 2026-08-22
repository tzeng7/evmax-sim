import { useCallback, useEffect, useState, type CSSProperties } from 'react'
import { runScan, resolve, fetchBalances } from '../lib/api'
import { useCategories } from '../lib/useCategories'
import { SectorMultiSelect } from './SectorMultiSelect'
import { fmtDate, fmtTomorrow, yesterday } from '../lib/odds'
import type { ScanGap } from '../lib/types'

// Venues that can seed the bankroll from their live account balance. Keys match
// the backend `venue` column / balances.SUPPORTED_VENUES; '' means "Manual".
const BANKROLL_VENUES: { value: string; label: string }[] = [
  { value: '', label: 'Manual' },
  { value: 'kalshi', label: 'Kalshi' },
  { value: 'polymarket_us', label: 'Polymarket US' },
]

const venueLabel = (v: string) =>
  BANKROLL_VENUES.find(o => o.value === v)?.label ?? v

const fmtUsd = (n: number) =>
  n.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 })

interface Props {
  bankrollStr: string
  setBankrollStr: (s: string) => void
  kelly: number
  setKelly: (k: number) => void
  onScanComplete: (
    gaps: ScanGap[],
    meta: { markets_fetched: number; markets_matched: number },
    snapshot: { bankroll: number; kelly: number },
  ) => void
  onResolve: () => void
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
}

export function ActionBar({ bankrollStr, setBankrollStr, kelly, setKelly, onScanComplete, onResolve, toast }: Props) {
  const cats = useCategories()
  const allSectors = cats.map(c => c.key).join(',')
  // Sector list comes from the registry via /api/categories. Default the
  // selection to "all sectors" once the list loads, but never clobber a choice
  // the user has already made.
  const [sectors, setSectors] = useState('')
  useEffect(() => {
    setSectors(prev => prev || allSectors)
  }, [allSectors])
  // Bankroll stored as a string so leading zeros the user types get stripped
  // explicitly. The previous Number-typed state had a known React bug where
  // typing "0" before an existing number left the visual "0…" in place
  // because the parsed value didn't change, so React never re-rendered the
  // input. String state with onChange-time sanitization fixes that.
  const bankroll = Number(bankrollStr) || 0
  const [dateFrom, setDateFrom] = useState(fmtDate)
  const [dateTo, setDateTo] = useState(fmtTomorrow)
  const [scanning, setScanning] = useState(false)
  const [resolving, setResolving] = useState(false)
  // Bankroll source: '' = manual (type it), or a venue whose LIVE total-wealth
  // balance seeds the bankroll. When a venue is selected the server re-resolves
  // the balance at scan time (bankroll_venue), so it is authoritative; the
  // client fetch here is just to show the number and keep live stake sizing in
  // sync before the scan runs.
  const [bankrollVenue, setBankrollVenue] = useState('')
  const [liveBalance, setLiveBalance] = useState<number | null>(null)
  const [balanceLoading, setBalanceLoading] = useState(false)

  const loadBalance = useCallback(async (venue: string) => {
    if (!venue) {
      setLiveBalance(null)
      return
    }
    setBalanceLoading(true)
    try {
      const balances = await fetchBalances(venue)
      const bal = balances[venue] ?? null
      setLiveBalance(bal)
      if (bal != null) {
        setBankrollStr(bal.toFixed(2))
      } else {
        toast(`No live balance for ${venueLabel(venue)} (no credentials?) — using manual bankroll`, 'info')
      }
    } catch (e) {
      setLiveBalance(null)
      toast('Balance fetch failed: ' + (e as Error).message, 'err')
    } finally {
      setBalanceLoading(false)
    }
  }, [setBankrollStr, toast])

  useEffect(() => { loadBalance(bankrollVenue) }, [bankrollVenue, loadBalance])

  const handleBankrollChange = (raw: string) => {
    // Strip leading zeros unless the user is typing "0" or "0.xxx" — keep
    // empty string allowed so the field can be cleared during editing.
    if (raw === '' || raw === '0' || raw.startsWith('0.')) {
      setBankrollStr(raw)
      return
    }
    setBankrollStr(raw.replace(/^0+/, ''))
  }

  const handleScan = async () => {
    setScanning(true)
    toast('Scanning markets...', 'info')
    try {
      const data = await runScan({
        sectors, bankroll, kelly, date_from: dateFrom, date_to: dateTo,
        bankroll_venue: bankrollVenue || undefined,
      })
      // The server may have re-sized the bankroll from a live venue balance.
      // Adopt the value it actually used so ScanResults sizes stakes against the
      // same number, and reflect it back into the input.
      const usedBankroll = typeof data.bankroll === 'number' ? data.bankroll : bankroll
      if (typeof data.bankroll === 'number' && data.bankroll !== bankroll) {
        setBankrollStr(data.bankroll.toFixed(2))
        setLiveBalance(data.bankroll)
      }
      onScanComplete(
        data.gaps,
        { markets_fetched: data.markets_fetched, markets_matched: data.markets_matched },
        { bankroll: usedBankroll, kelly },
      )
      const portfolioBets = (data.portfolio_results || []).reduce((s, r) => s + r.gaps_logged, 0)
      const portfolioSuffix = data.portfolio_results?.length
        ? ` · logged ${portfolioBets} to ${data.portfolio_results.length} portfolio(s)`
        : ''
      // Tell the user when the live-balance sizing did or did not apply.
      let bankrollSuffix = ''
      if (data.bankroll_source?.startsWith('live:')) {
        bankrollSuffix = ` · sized against ${venueLabel(bankrollVenue)} live ${fmtUsd(usedBankroll)}`
      } else if (data.bankroll_source === 'manual_fallback') {
        bankrollSuffix = ` · ${venueLabel(bankrollVenue)} balance unavailable, used manual ${fmtUsd(usedBankroll)}`
      }
      toast(`Found ${data.gaps.length} +EV plays across ${data.markets_fetched} markets${portfolioSuffix}${bankrollSuffix}`, 'ok')
    } catch (e) {
      toast('Scan failed: ' + (e as Error).message, 'err')
    } finally {
      setScanning(false)
    }
  }

  const handleResolve = async () => {
    setResolving(true)
    toast('Resolving outcomes...', 'info')
    try {
      // Resolve yesterday + today, then sync portfolios once on the second
      // call (after both dates' ev_outcomes are populated). One click now
      // covers both dashboard and portfolio outcomes.
      const r1 = await resolve(yesterday(), false)
      const r2 = await resolve(fmtDate(), true)
      const total = (r1.resolved || 0) + (r2.resolved || 0)
      const portfolio = r2.portfolio_resolved || 0
      const portfolioSuffix = portfolio ? ` · synced ${portfolio} portfolio bet(s)` : ''
      toast(`Resolved ${total} outcome(s)${portfolioSuffix}`, 'ok')
      onResolve()
    } catch (e) {
      toast('Resolve failed: ' + (e as Error).message, 'err')
    } finally {
      setResolving(false)
    }
  }

  return (
    <div className="actions">
      <SectorMultiSelect value={sectors} onChange={setSectors} />
      <select
        value={bankrollVenue}
        onChange={e => setBankrollVenue(e.target.value)}
        style={{ fontSize: 11 }}
        title="Bankroll source — Manual, or size Kelly against a venue's live total-wealth balance"
      >
        {BANKROLL_VENUES.map(o => (
          <option key={o.value || 'manual'} value={o.value}>
            {o.value ? `${o.label} (live)` : o.label}
          </option>
        ))}
      </select>
      <input
        type="number"
        value={bankrollStr}
        onChange={e => handleBankrollChange(e.target.value)}
        style={{ width: 80 }}
        readOnly={!!bankrollVenue}
        title={bankrollVenue
          ? `Live ${venueLabel(bankrollVenue)} balance — Kelly is sized against this at scan time`
          : 'Bankroll ($)'}
      />
      {bankrollVenue && (
        <span
          className="muted"
          style={{ fontSize: 11, display: 'inline-flex', alignItems: 'center', gap: 6 }}
        >
          <span className={`badge${liveBalance != null ? '' : ' warn'}`}>
            {balanceLoading ? 'live …' : liveBalance != null ? 'LIVE' : 'no creds'}
          </span>
          <button
            className="btn"
            onClick={() => loadBalance(bankrollVenue)}
            disabled={balanceLoading}
            style={{ padding: '2px 8px', fontSize: 11 }}
            title="Refresh live balance"
          >
            ⟳
          </button>
        </span>
      )}
      <div className="kelly-slider" title="Kelly fraction — drag to resize every stake live">
        <label htmlFor="kelly-range">Kelly</label>
        <input
          id="kelly-range"
          type="range"
          min={0.1}
          max={1}
          step={0.05}
          value={kelly}
          onChange={e => setKelly(+e.target.value)}
          style={{ '--pct': `${((kelly - 0.1) / 0.9) * 100}%` } as CSSProperties}
        />
        <span className="kelly-slider-val">{kelly.toFixed(2)}×</span>
      </div>
      <input type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)}
        style={{ width: 138, fontSize: 11 }} title="From date" />
      <span className="muted">–</span>
      <input type="date" value={dateTo} onChange={e => setDateTo(e.target.value)}
        style={{ width: 138, fontSize: 11 }} title="To date" />
      <button className="btn primary" onClick={handleScan} disabled={scanning}>
        {scanning ? 'Scanning...' : 'Scan'}
      </button>
      <button className="btn" onClick={handleResolve} disabled={resolving}>
        {resolving ? 'Resolving...' : 'Resolve'}
      </button>
    </div>
  )
}
