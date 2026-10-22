import { useSyncExternalStore } from 'react'

export type ActivityType = 'info' | 'ok' | 'err'

export interface ActivityEntry {
  id: number
  at: number
  type: ActivityType
  message: string
  /** Extra context, e.g. the action that failed and the API error body. */
  detail?: string
  /** False until the drawer has been opened since the entry arrived. */
  seen: boolean
}

const MAX_ENTRIES = 100
const STORAGE_KEY = 'evmax.activityLog'

function load(): ActivityEntry[] {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed.slice(-MAX_ENTRIES) : []
  } catch {
    return []
  }
}

let entries: ActivityEntry[] = load()
let nextId = entries.reduce((m, e) => Math.max(m, e.id), 0) + 1
const listeners = new Set<() => void>()

function commit(next: ActivityEntry[]) {
  entries = next
  try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(entries)) } catch { /* storage unavailable */ }
  listeners.forEach((l) => l())
}

export function logActivity(type: ActivityType, message: string, detail?: string) {
  commit([...entries, { id: nextId++, at: Date.now(), type, message, detail, seen: false }].slice(-MAX_ENTRIES))
}

export function dismissActivity(id: number) {
  commit(entries.filter((e) => e.id !== id))
}

export function clearActivity() {
  commit([])
}

export function markActivitySeen() {
  if (entries.some((e) => !e.seen)) commit(entries.map((e) => (e.seen ? e : { ...e, seen: true })))
}

export function activityAsText(list: ActivityEntry[]): string {
  return list
    .map((e) => `[${new Date(e.at).toISOString()}] ${e.type.toUpperCase()} ${e.message}${e.detail ? `\n    ${e.detail.replace(/\n/g, '\n    ')}` : ''}`)
    .join('\n')
}

function subscribe(l: () => void) {
  listeners.add(l)
  return () => { listeners.delete(l) }
}

export function useActivityLog(): ActivityEntry[] {
  return useSyncExternalStore(subscribe, () => entries)
}
