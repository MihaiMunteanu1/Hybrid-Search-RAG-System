import { useEffect, useMemo, useState } from 'react'
import { api, type Sheet } from '../api'

const columnName = (index: number) => {
  let name = ''
  for (let n = index + 1; n > 0; n = Math.floor((n - 1) / 26)) name = String.fromCharCode(65 + ((n - 1) % 26)) + name
  return name
}

// Excel widths are in characters of the default font; about 7 px each plus padding.
const columnWidth = (width: number | undefined) => Math.round(Math.max(width || 9, 4) * 7.2 + 10)

// A spreadsheet or CSV as a grid: column letters, row numbers, merged cells, one tab per sheet.
export function SheetView({ source }: { source: string }) {
  const [sheets, setSheets] = useState<Sheet[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [active, setActive] = useState(0)

  useEffect(() => {
    let cancelled = false
    api.sheets(source).then(
      (r) => { if (!cancelled) setSheets(r.sheets) },
      (err: unknown) => { if (!cancelled) setError(err instanceof Error ? err.message : String(err)) },
    )
    return () => { cancelled = true }
  }, [source])

  const sheet = sheets?.[active]

  // Cells covered by a merge are not drawn; the merge's first cell spans over them.
  const layout = useMemo(() => {
    const spans = new Map<string, { rows: number; cols: number }>()
    const hidden = new Set<string>()
    for (const m of sheet?.merges ?? []) {
      spans.set(`${m.r},${m.c}`, { rows: m.rows, cols: m.cols })
      for (let r = m.r; r < m.r + m.rows; r++)
        for (let c = m.c; c < m.c + m.cols; c++) if (r !== m.r || c !== m.c) hidden.add(`${r},${c}`)
    }
    return { spans, hidden }
  }, [sheet])

  if (error) return <p className="note error viewer-note">{error}</p>
  if (!sheets || !sheet) return <p className="note viewer-note">Loading…</p>

  const columns = sheet.rows[0]?.length ?? 0
  return (
    <div className="sheet-view">
      <div className="sheet-scroll">
        {sheet.rows.length === 0 ? (
          <p className="note viewer-note">This sheet is empty.</p>
        ) : (
          <table className="sheet-grid">
            <colgroup>
              <col style={{ width: 44 }} />
              {Array.from({ length: columns }, (_, c) => (
                <col key={c} style={{ width: columnWidth(sheet.widths[c]) }} />
              ))}
            </colgroup>
            <thead>
              <tr>
                <th className="corner" />
                {Array.from({ length: columns }, (_, c) => <th key={c}>{columnName(c)}</th>)}
              </tr>
            </thead>
            <tbody>
              {sheet.rows.map((row, r) => (
                <tr key={r}>
                  <th className="row-n">{r + 1}</th>
                  {row.map((cell, c) => {
                    const key = `${r},${c}`
                    if (layout.hidden.has(key)) return null
                    const span = layout.spans.get(key)
                    const numeric = cell !== '' && !Number.isNaN(Number(cell))
                    return (
                      <td key={c} rowSpan={span?.rows} colSpan={span?.cols} className={numeric ? 'num' : undefined}>
                        {cell}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {sheet.truncated && <p className="note viewer-note">Only the first 2000 rows are shown.</p>}
      </div>
      {sheets.length > 1 && (
        <div className="sheet-tabs" role="tablist">
          {sheets.map((s, i) => (
            <button key={s.name} type="button" role="tab" aria-selected={i === active} className={i === active ? 'on' : ''} onClick={() => setActive(i)}>
              {s.name}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
