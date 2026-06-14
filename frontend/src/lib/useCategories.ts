import { useEffect, useState } from 'react'
import { fetchCategories } from './api'
import type { Category } from './types'

// Module-level cache so the sector list is fetched once and shared by every
// dropdown (ActionBar + the three SectorFilter mounts) regardless of how many
// components call the hook. Sourced from /api/categories, which the backend
// derives from data/categories.yaml — no hardcoded sector arrays.
let cache: Category[] | null = null
let inflight: Promise<Category[]> | null = null

export function useCategories(): Category[] {
  const [cats, setCats] = useState<Category[]>(cache ?? [])
  useEffect(() => {
    if (cache) return
    if (!inflight) inflight = fetchCategories().then(c => { cache = c; return c })
    let active = true
    inflight.then(c => { if (active) setCats(c) }).catch(() => {})
    return () => { active = false }
  }, [])
  return cats
}
