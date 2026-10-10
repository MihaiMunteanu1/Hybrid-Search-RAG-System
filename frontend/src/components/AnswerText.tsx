import type { ReactNode } from 'react'

interface Props {
  text: string
  sourceCount: number
  onCite: (n: number) => void
}

// [1]   [1, 2]   [1][3]   [ 2 ]  -- the same forms rag/generation/prompt.py accepts.
const CITATION = /\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]/g
const BOLD = /\*\*(.+?)\*\*/g

// The model writes a little markdown: paragraphs, "* " bullets, **bold**, and [n]
// citations. It is turned into React elements, never injected as HTML, so whatever the
// model or a document contains is shown as text and cannot run in the page.
export function AnswerText({ text, sourceCount, onCite }: Props) {
  return (
    <div className="answer-text">
      {text.split(/\n{2,}/).map((paragraph, i) => (
        <p key={i}>
          {paragraph.split('\n').map((line, j) => (
            <span key={j}>
              {j > 0 && <br />}
              {renderLine(line.replace(/^\s*[*-]\s+/, '• '), sourceCount, onCite)}
            </span>
          ))}
        </p>
      ))}
    </div>
  )
}

function renderLine(line: string, sourceCount: number, onCite: (n: number) => void): ReactNode[] {
  const out: ReactNode[] = []
  let last = 0
  for (const match of line.matchAll(CITATION)) {
    out.push(...renderBold(line.slice(last, match.index), out.length))
    for (const n of match[1].split(/\s*[,;]\s*/).map(Number)) {
      const valid = n >= 1 && n <= sourceCount
      out.push(
        <button
          key={`c${out.length}`}
          type="button"
          className={valid ? 'cite' : 'cite invalid'}
          title={valid ? `Source ${n}` : 'No such source'}
          disabled={!valid}
          onClick={() => onCite(n)}
        >
          {n}
        </button>,
      )
    }
    last = match.index + match[0].length
  }
  out.push(...renderBold(line.slice(last), out.length))
  return out
}

function renderBold(text: string, offset: number): ReactNode[] {
  return text.split(BOLD).map((part, i) =>
    i % 2 === 1 ? <strong key={`b${offset + i}`}>{part}</strong> : part,
  )
}
