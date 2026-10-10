import { useEffect, useState } from 'react'
import { api, type DocumentText } from './api'

// A document's extracted text, loaded once per source.
export function useDocumentText(source: string) {
  const [doc, setDoc] = useState<DocumentText | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let cancelled = false
    api.documentText(source).then(
      (d) => { if (!cancelled) setDoc(d) },
      (err: unknown) => { if (!cancelled) setError(err instanceof Error ? err.message : String(err)) },
    )
    return () => { cancelled = true }
  }, [source])
  return { doc, error }
}

// The extracted text of a document is markdown-like: "#" headings, "- " items,
// "| a | b |" tables, paragraphs between blank lines. These helpers split it into blocks
// that keep their character range, so a cited span can be found in them.

export interface Block {
  kind: 'h' | 'p' | 'li' | 'table'
  level?: number
  text: string
  rows?: string[][]
  start: number
  end: number
}

interface OpenBlocks {
  para: { start: number; end: number; lines: string[] } | null
  table: { start: number; end: number; rows: string[][] } | null
}

export function parseBlocks(text: string): Block[] {
  const blocks: Block[] = []
  const open: OpenBlocks = { para: null, table: null }
  const flush = () => {
    if (open.para) blocks.push({ kind: 'p', text: open.para.lines.join(' '), start: open.para.start, end: open.para.end })
    if (open.table) blocks.push({ kind: 'table', text: '', rows: open.table.rows, start: open.table.start, end: open.table.end })
    open.para = null
    open.table = null
  }
  let pos = 0
  for (const line of text.split('\n')) {
    const start = pos
    const end = pos + line.length
    pos = end + 1
    const trimmed = line.trim()
    if (!trimmed) {
      flush()
      continue
    }
    const heading = /^(#{1,6})\s+(.*)$/.exec(trimmed)
    if (heading) {
      flush()
      blocks.push({ kind: 'h', level: heading[1].length, text: heading[2], start, end })
      continue
    }
    if (trimmed.startsWith('|')) {
      if (open.para) flush()
      if (/^\|[\s|:-]+\|$/.test(trimmed) && trimmed.includes('-')) {   // header separator
        if (open.table) open.table.end = end
        continue
      }
      const cells = trimmed.replace(/^\||\|$/g, '').split('|').map((c) => c.trim())
      if (open.table) {
        open.table.rows.push(cells)
        open.table.end = end
      } else {
        open.table = { start, end, rows: [cells] }
      }
      continue
    }
    if (open.table) flush()
    const item = /^([-*+]|\d+[.)])\s+(.*)$/.exec(trimmed)
    if (item) {
      flush()
      blocks.push({ kind: 'li', text: item[2], start, end })
      continue
    }
    if (open.para) {
      open.para.lines.push(trimmed)
      open.para.end = end
    } else {
      open.para = { start, end, lines: [trimmed] }
    }
  }
  flush()
  return blocks
}

// 1-based page of a character position, from the offsets where PDF pages start.
export function pageAt(pageStarts: number[] | null, position: number): number | null {
  if (!pageStarts || pageStarts.length === 0) return null
  let page = 0
  while (page < pageStarts.length && pageStarts[page] <= position) page++
  return page
}
