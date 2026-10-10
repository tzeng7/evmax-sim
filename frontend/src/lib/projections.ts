import type { ProjectionOption, ProjectionOptionValues } from './types'

/** The declared defaults of an engine's options (null = auto). */
export function optionDefaults(specs: ProjectionOption[] | undefined): ProjectionOptionValues {
  return Object.fromEntries((specs ?? []).map(s => [s.key, s.default]))
}
