import { lazy, Suspense, useEffect, useState } from 'react'
import { fileUrl } from '../api'
import { CloseIcon, ExternalIcon, FileIcon } from '../icons'
import { DocTextView } from './DocTextView'
import { DocxView } from './DocxView'
import { SheetView } from './SheetView'
import { SlidesView } from './SlidesView'

const MarkdownView = lazy(() => import('./MarkdownView'))

// What to open: a document, optionally at a page and with a passage to highlight
// (the character span of a cited chunk in the extracted text).
export interface ViewTarget {
  source: string
  page?: number | null
  span?: [number, number] | null
}

interface Props {
  target: ViewTarget
  onClose: () => void
}

type Kind = 'pdf' | 'docx' | 'sheet' | 'markdown' | 'slides' | 'html' | 'text'

const KINDS: Record<string, Kind> = {
  pdf: 'pdf', docx: 'docx', xlsx: 'sheet', csv: 'sheet', md: 'markdown', markdown: 'markdown',
  pptx: 'slides', html: 'html', txt: 'text',
}

// The name of the faithful view for each kind; "Text" is always the extracted text.
const ORIGINAL_LABEL: Record<Kind, string> = {
  pdf: 'Original', docx: 'Original', sheet: 'Sheet', markdown: 'Formatted', slides: 'Slides',
  html: 'Page', text: 'Text',
}

export function Viewer({ target, onClose }: Props) {
  const extension = target.source.split('.').pop()?.toLowerCase() ?? ''
  const kind: Kind = KINDS[extension] ?? 'text'
  // A cited passage is highlighted in the text view, so a citation opens there; a PDF
  // opens as itself at the cited page, with the text one click away.
  const [mode, setMode] = useState<'original' | 'text'>(
    kind === 'text' || (target.span && kind !== 'pdf') ? 'text' : 'original',
  )
  const name = target.source.split('/').pop() ?? target.source

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div className="viewer-backdrop" onClick={onClose}>
      <div className="viewer" role="dialog" aria-modal="true" aria-label={name} onClick={(e) => e.stopPropagation()}>
        <header className="viewer-head">
          <span className={`viewer-icon kind-${kind}`}><FileIcon size={18} /></span>
          <div className="viewer-title">
            <strong title={target.source}>{name}</strong>
            <span>
              {target.source.split('/')[0]}
              {target.page ? ` · page ${target.page}` : ''}
              {target.span ? ' · cited passage highlighted in Text' : ''}
            </span>
          </div>
          {kind !== 'text' && (
            <div className="segmented" role="tablist">
              <button type="button" role="tab" aria-selected={mode === 'original'} className={mode === 'original' ? 'on' : ''} onClick={() => setMode('original')}>
                {ORIGINAL_LABEL[kind]}
              </button>
              <button type="button" role="tab" aria-selected={mode === 'text'} className={mode === 'text' ? 'on' : ''} onClick={() => setMode('text')}>
                Text
              </button>
            </div>
          )}
          <a className="btn ghost small viewer-open" href={fileUrl(target.source)} target="_blank" rel="noreferrer">
            <ExternalIcon size={14} />
            <span>Open original</span>
          </a>
          <button type="button" className="icon-btn" aria-label="Close" onClick={onClose}>
            <CloseIcon />
          </button>
        </header>

        <div className={`viewer-body view-${mode === 'text' ? 'text' : kind}`}>
          {mode === 'text' ? (
            <DocTextView source={target.source} span={target.span ?? null} />
          ) : (
            <Original kind={kind} target={target} name={name} />
          )}
        </div>
      </div>
    </div>
  )
}

function Original({ kind, target, name }: { kind: Kind; target: ViewTarget; name: string }) {
  switch (kind) {
    case 'pdf':
      return <iframe className="viewer-frame" src={`${fileUrl(target.source)}#page=${target.page ?? 1}`} title={name} />
    case 'docx':
      return <DocxView source={target.source} />
    case 'sheet':
      return <SheetView source={target.source} />
    case 'slides':
      return <SlidesView source={target.source} />
    case 'markdown':
      return (
        <Suspense fallback={<p className="note viewer-note">Loading…</p>}>
          <MarkdownView source={target.source} />
        </Suspense>
      )
    case 'html':
      // sandbox="" gives the page no scripts, forms or same-origin access; the server
      // sends a sandbox policy for it too.
      return <iframe className="viewer-frame" sandbox="" src={fileUrl(target.source)} title={name} />
    default:
      return <DocTextView source={target.source} span={target.span ?? null} />
  }
}
