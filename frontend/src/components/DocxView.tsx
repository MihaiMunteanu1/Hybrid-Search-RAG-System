import { useEffect, useRef, useState } from 'react'
import { fileUrl } from '../api'

// A Word document laid out as Word would: pages, fonts, tables, images, headers and
// footers, rendered in the browser by docx-preview. The library is loaded only when a
// .docx is opened, so it does not weigh on the rest of the interface.
export function DocxView({ source }: { source: string }) {
  const body = useRef<HTMLDivElement>(null)
  const styles = useRef<HTMLDivElement>(null)
  const [state, setState] = useState<'loading' | 'ready' | string>('loading')

  useEffect(() => {
    let cancelled = false
    const target = body.current
    const styleTarget = styles.current
    if (!target || !styleTarget) return
    ;(async () => {
      try {
        const [{ renderAsync }, response] = await Promise.all([import('docx-preview'), fetch(fileUrl(source))])
        if (!response.ok) throw new Error(`could not load the file (${response.status})`)
        const data = await response.arrayBuffer()
        if (cancelled) return
        await renderAsync(data, target, styleTarget, {
          className: 'docx',
          inWrapper: true,
          breakPages: true,
          renderHeaders: true,
          renderFooters: true,
          renderFootnotes: true,
          useBase64URL: true,
        })
        // A document's own links may point anywhere, including "javascript:". Keep only
        // ordinary web and mail links, and open them outside the app.
        target.querySelectorAll('a[href]').forEach((a) => {
          const href = a.getAttribute('href') ?? ''
          if (/^(https?:|mailto:)/i.test(href)) {
            a.setAttribute('target', '_blank')
            a.setAttribute('rel', 'noreferrer')
          } else if (!href.startsWith('#')) {
            a.removeAttribute('href')
          }
        })
        if (!cancelled) setState('ready')
      } catch (err) {
        if (!cancelled) setState(err instanceof Error ? err.message : String(err))
      }
    })()
    return () => {
      cancelled = true
      target.replaceChildren()
      styleTarget.replaceChildren()
    }
  }, [source])

  return (
    <div className="docx-view">
      {state === 'loading' && <p className="note viewer-note">Laying out the document…</p>}
      {state !== 'loading' && state !== 'ready' && <p className="note error viewer-note">{state}</p>}
      <div ref={styles} />
      <div ref={body} />
    </div>
  )
}
