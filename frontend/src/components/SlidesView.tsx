import { useMemo } from 'react'
import { parseBlocks, useDocumentText, type Block } from '../docText'
import { BlockView } from './DocTextView'

interface Slide {
  title: string
  body: Block[]
  notes: string[]
}

// A presentation as slides. The loader writes each slide as a "#" section with its
// bullets and speaker notes; each section becomes one 16:9 card.
export function SlidesView({ source }: { source: string }) {
  const { doc, error } = useDocumentText(source)

  const slides = useMemo(() => {
    const out: Slide[] = []
    for (const block of doc ? parseBlocks(doc.text) : []) {
      if (block.kind === 'h' && block.level === 1) {
        out.push({ title: block.text, body: [], notes: [] })
      } else if (out.length > 0) {
        const slide = out[out.length - 1]
        if (block.kind === 'p' && block.text.startsWith('Speaker notes:')) {
          slide.notes.push(block.text.slice('Speaker notes:'.length).trim())
        } else {
          slide.body.push(block)
        }
      }
    }
    return out
  }, [doc])

  if (error) return <p className="note error viewer-note">{error}</p>
  if (!doc) return <p className="note viewer-note">Loading…</p>

  return (
    <div className="slides">
      {slides.map((slide, i) => (
        <section key={i} className="slide-wrap">
          <div className="slide">
            <h2>{slide.title}</h2>
            <div className="slide-body">
              {slide.body.map((block, b) => <BlockView key={b} block={block} />)}
            </div>
            <span className="slide-n">{i + 1}</span>
          </div>
          {slide.notes.length > 0 && <p className="slide-notes">{slide.notes.join(' ')}</p>}
        </section>
      ))}
    </div>
  )
}
