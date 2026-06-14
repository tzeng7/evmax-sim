import { useCategories } from '../lib/useCategories'

interface Props {
  value: string
  onChange: (v: string) => void
}

export function SectorFilter({ value, onChange }: Props) {
  const cats = useCategories()
  return (
    <select value={value} onChange={e => onChange(e.target.value)} style={{ fontSize: 11, padding: '4px 8px' }}>
      <option value="">All Sectors</option>
      {cats.map(c => (
        <option key={c.key} value={c.key} title={c.display_name}>{c.key.toUpperCase()}</option>
      ))}
    </select>
  )
}
