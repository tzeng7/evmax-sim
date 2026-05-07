const SECTORS = ['nba', 'wnba', 'ncaab', 'soccer', 'tennis', 'baseball', 'lol', 'cs2']

interface Props {
  value: string
  onChange: (v: string) => void
}

export function SectorFilter({ value, onChange }: Props) {
  return (
    <select value={value} onChange={e => onChange(e.target.value)} style={{ fontSize: 11, padding: '4px 8px' }}>
      <option value="">All Sectors</option>
      {SECTORS.map(s => (
        <option key={s} value={s}>{s.toUpperCase()}</option>
      ))}
    </select>
  )
}
