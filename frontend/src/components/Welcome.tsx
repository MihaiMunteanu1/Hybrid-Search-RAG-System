import { useEffect } from 'react'
import type { Status } from '../api'
import { ArrowRightIcon, LogoMark } from '../icons'

interface Props {
  status: Status | null
  onEnter: () => void
}

export function Welcome({ status, onEnter }: Props) {
  // Enter anywhere on the page opens the app, like the button.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Enter') onEnter()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onEnter])

  return (
    <div className="welcome">
      <div className="welcome-inner">
        <div className="welcome-mark">
          <LogoMark size={64} />
        </div>
        <p className="eyebrow">RAG Library</p>
        <h1>
          Ask your
          <br />
          <span className="gradient-text">documents.</span>
        </h1>
        <p className="lead">
          Regulations, guides and theses, in English and Romanian. Every answer cites its
          sources, and when the documents don't say, it tells you.
        </p>
        <button type="button" className="enter" onClick={onEnter}>
          Enter
          <ArrowRightIcon size={18} />
        </button>
        <ul className="pills">
          <li>Hybrid search</li>
          <li>Verified citations</li>
          <li>Runs locally</li>
        </ul>
        <p className="welcome-meta">
          {status && `${status.documents} documents in ${status.folders} ${status.folders === 1 ? 'folder' : 'folders'}`}
        </p>
      </div>
    </div>
  )
}
