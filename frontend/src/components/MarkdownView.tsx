import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useDocumentText } from '../docText'

// A Markdown file rendered as a document: headings, lists, tables, code, links.
// react-markdown builds React elements and ignores raw HTML, so a file cannot inject
// markup or scripts. Loaded lazily, only when a Markdown file is opened.
export default function MarkdownView({ source }: { source: string }) {
  const { doc, error } = useDocumentText(source)
  if (error) return <p className="note error viewer-note">{error}</p>
  if (!doc) return <p className="note viewer-note">Loading…</p>
  return (
    <article className="markdown-view">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{ a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a> }}
      >
        {doc.text}
      </Markdown>
    </article>
  )
}
