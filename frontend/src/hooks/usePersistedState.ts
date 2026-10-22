import { useCallback, useState } from 'react'

const PREFIX = 'evmax:'

function read<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(PREFIX + key)
    return raw === null ? fallback : (JSON.parse(raw) as T)
  } catch {
    return fallback
  }
}

/** useState backed by localStorage. Every storage access is guarded, so the
 *  hook degrades to plain in-memory state when storage is unavailable. */
export function usePersistedState<T>(key: string, initial: T): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() => read(key, initial))
  const set = useCallback((v: T) => {
    setValue(v)
    try { window.localStorage.setItem(PREFIX + key, JSON.stringify(v)) } catch { /* storage unavailable */ }
  }, [key])
  return [value, set]
}
