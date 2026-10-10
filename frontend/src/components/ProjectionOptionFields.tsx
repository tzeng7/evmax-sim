import type { ProjectionOption, ProjectionOptionValues } from '../lib/types'

interface Props {
  specs: ProjectionOption[]
  values: ProjectionOptionValues
  onChange: (values: ProjectionOptionValues) => void
  disabled?: boolean
}

/**
 * One input per option an engine declares (GET /api/projections/sectors), so
 * a new sector's settings render with no frontend change. Integers render as
 * number inputs (blank = auto when nullable); booleans as checkboxes. The
 * server re-validates everything on submit.
 */
export function ProjectionOptionFields({ specs, values, onChange, disabled }: Props) {
  const set = (key: string, v: number | boolean | null) => onChange({ ...values, [key]: v })

  return (
    <>
      {specs.map(s => {
        const v = values[s.key]
        if (s.type === 'bool') {
          return (
            <label key={s.key} className="proj-field proj-check" title={s.help}>
              <input
                type="checkbox"
                checked={Boolean(v)}
                disabled={disabled}
                onChange={e => set(s.key, e.target.checked)}
              />
              {s.label}
            </label>
          )
        }
        return (
          <label key={s.key} className="proj-field" title={s.help}>
            <span className="proj-field-label">{s.label}</span>
            <input
              type="number"
              inputMode="numeric"
              value={v == null ? '' : String(v)}
              min={s.min}
              max={s.max}
              step={s.step ?? 1}
              placeholder={s.placeholder || (s.default != null ? String(s.default) : '')}
              disabled={disabled}
              onChange={e => {
                const raw = e.target.value
                if (raw === '') set(s.key, s.nullable ? null : (s.default as number | null))
                else if (Number.isFinite(Number(raw))) set(s.key, Number(raw))
              }}
              style={{ width: s.max != null && s.max >= 10000 ? 96 : 76 }}
            />
          </label>
        )
      })}
    </>
  )
}
