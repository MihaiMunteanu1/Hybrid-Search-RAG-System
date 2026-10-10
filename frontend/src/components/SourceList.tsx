import { useEffect, useRef } from 'react'
import type { Source } from '../api'
import { ChevronIcon, ExternalIcon, FileIcon } from '../icons'
import type { ViewTarget } from './Viewer'

interface Props {
  sources: Source[]
  cited: number[]
  // The source to open and scroll to. `nonce` changes on every click, so clicking the
  // same citation twice still scrolls back to it.
  focus: { n: number; nonce: number } | null
  defaultOpen?: boolean
  onOpen: (target: ViewTarget) => void
}

export function SourceList({ sources, cited, focus, defaultOpen = false, onOpen }: Props) {
  const list = useRef<HTMLDetailsElement>(null)
  const refs = useRef(new Map<number, HTMLDetailsElement>())

  useEffect(() => {
    if (!focus) return
    const element = refs.current.get(focus.n)
    if (!element) return
    if (list.current) list.current.open = true
    element.open = true
    element.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    element.classList.add('flash')
    const timer = setTimeout(() => element.classList.remove('flash'), 1200)
    return () => clearTimeout(timer)
  }, [focus])

  if (sources.length === 0) return null
  const citedCount = sources.filter((s) => cited.includes(s.n)).length

  return (
    <details className="sources" ref={list} open={defaultOpen}>
      <summary>
        <ChevronIcon size={14} className="chevron" />
        {sources.length} {sources.length === 1 ? 'source' : 'sources'}
        {citedCount > 0 && <span className="cited-count">{citedCount} cited</span>}
      </summary>
      <div className="source-list">
        {sources.map((s) => (
          <details
            key={s.n}
            ref={(el) => {
              if (el) refs.current.set(s.n, el)
              else refs.current.delete(s.n)
            }}
            className={cited.includes(s.n) ? 'source cited' : 'source'}
          >
            <summary>
              <span className="source-n">{s.n}</span>
              <span className="source-title">
                <span className="source-file">
                  <FileIcon size={13} />
                  {s.source.split('/').pop()}
                  {s.page ? <span className="source-page">p. {s.page}</span> : null}
                </span>
                {s.heading_path && <span className="source-path">{s.heading_path}</span>}
              </span>
            </summary>
            <pre>{s.text}</pre>
            <div className="source-actions">
              <button
                type="button"
                className="link-btn"
                onClick={() => onOpen({ source: s.source, page: s.page, span: [s.start_char, s.end_char] })}
              >
                <ExternalIcon size={13} />
                Open in document{s.page ? `, page ${s.page}` : ''}
              </button>
            </div>
          </details>
        ))}
      </div>
    </details>
  )
}
