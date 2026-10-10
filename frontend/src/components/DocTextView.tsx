import { useEffect, useMemo, useRef } from 'react'
import { pageAt, parseBlocks, useDocumentText, type Block } from '../docText'

// The text the app extracted from a document, laid out for reading, with page dividers
// for PDFs and the cited passage (a character span) highlighted and scrolled into view.
export function DocTextView({ source, span }: { source: string; span: [number, number] | null }) {
  const { doc, error } = useDocumentText(source)
  const markRef = useRef<HTMLDivElement>(null)

  const blocks = useMemo(() => (doc ? parseBlocks(doc.text) : []), [doc])
  const pages = useMemo(() => blocks.map((b) => pageAt(doc?.page_starts ?? null, b.start)), [blocks, doc])
  const isMarked = (b: Block) => span !== null && b.start < span[1] && span[0] < b.end
  const firstMarked = blocks.findIndex(isMarked)

  useEffect(() => {
    markRef.current?.scrollIntoView({ block: 'center' })
  }, [blocks, firstMarked])

  if (error) return <p className="note error viewer-note">{error}</p>
  if (!doc) return <p className="note viewer-note">Loading…</p>

  return (
    <article className="doc-text">
      {blocks.map((block, i) => (
        <div key={i} ref={i === firstMarked ? markRef : undefined}>
          {pages[i] !== null && pages[i] !== pages[i - 1] && <div className="page-mark">Page {pages[i]}</div>}
          <BlockView block={block} marked={isMarked(block)} />
        </div>
      ))}
    </article>
  )
}

export function BlockView({ block, marked = false }: { block: Block; marked?: boolean }) {
  const mark = marked ? ' mark' : ''
  if (block.kind === 'h') {
    const level = Math.min((block.level ?? 1) + 1, 6)
    return <div className={`doc-h doc-h${level}${mark}`}>{block.text}</div>
  }
  if (block.kind === 'li') return <p className={`doc-li${mark}`}>{block.text}</p>
  if (block.kind === 'table') {
    return (
      <div className={`doc-table${mark}`}>
        <table>
          <tbody>
            {block.rows?.map((row, r) => (
              <tr key={r}>{row.map((cell, c) => (r === 0 ? <th key={c}>{cell}</th> : <td key={c}>{cell}</td>))}</tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }
  return <p className={marked ? 'mark' : undefined}>{block.text}</p>
}
