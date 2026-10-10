import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent, type ReactNode } from 'react'
import { api, ask, type AnswerResult, type Source, type Topic } from '../api'
import { ArrowUpIcon, ComposeIcon, FileIcon, FolderIcon, LogoMark, RefreshIcon, StackIcon, StopIcon } from '../icons'
import { AnswerText } from './AnswerText'
import { SourceList } from './SourceList'
import type { ViewTarget } from './Viewer'

interface Props {
  folder: string | null
  llm: boolean
  leading?: ReactNode     // shown before the brand in the top bar (the library toggle)
  onHome: () => void
  onOpen: (target: ViewTarget) => void
}

type Phase = 'searching' | 'reading' | 'writing' | 'retry' | 'done' | 'error' | 'stopped'

// One question and its answer. Each question is answered on its own: the model does not
// see the earlier turns, which stay on screen only as a history.
interface Turn {
  id: number
  question: string
  folder: string | null
  phase: Phase
  sources: Source[]
  streamed: string
  result: AnswerResult | null
  error: string | null
  started: number
}

const WORKING: Partial<Record<Phase, string>> = {
  searching: 'Searching the documents',
  reading: 'Reading the sources',
  writing: 'Writing the answer',
  retry: 'Adding citations',
}

const REFUSALS: Record<string, string> = {
  no_sources: 'There are no documents to search.',
  model: 'The sources found do not contain this information.',
  no_citations: 'The answer cited no source, so it was not shown.',
}

const isBusy = (phase: Phase) => phase in WORKING
const clock = () => Date.now()
const scopeName = (folder: string | null) => (folder === null ? 'All documents' : folder)
const topicQuestion = (topic: Topic) => `Summarize the section “${topic.heading}”.`

export function Chat({ folder, llm, leading, onHome, onOpen }: Props) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [question, setQuestion] = useState('')
  const [now, setNow] = useState(() => Date.now())
  const [focus, setFocus] = useState<{ turn: number; n: number; nonce: number } | null>(null)
  const [topics, setTopics] = useState<Topic[] | null>(null)
  const [topicsRound, setTopicsRound] = useState(0)
  const controller = useRef<AbortController | null>(null)
  const nextId = useRef(1)
  const scroller = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLTextAreaElement>(null)
  const followBottom = useRef(true)

  const busy = turns.some((t) => isBusy(t.phase))

  // A ticking clock while the model works: on a CPU an answer takes about a minute.
  useEffect(() => {
    if (!busy) return
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [busy])

  useEffect(() => () => controller.current?.abort(), [])

  // Starting points taken from the library itself: real section headings, at random,
  // from the folder being searched.
  useEffect(() => {
    let cancelled = false
    api.topics(folder).then(
      (found) => { if (!cancelled) setTopics(found) },
      () => { if (!cancelled) setTopics([]) },
    )
    return () => { cancelled = true }
  }, [folder, topicsRound])

  // Keep the newest text in view, unless the reader has scrolled up to read.
  useEffect(() => {
    const el = scroller.current
    if (el && followBottom.current) el.scrollTo({ top: el.scrollHeight })
  }, [turns])

  const update = (id: number, patch: Partial<Turn>) =>
    setTurns((all) => all.map((t) => (t.id === id ? { ...t, ...patch } : t)))

  async function send(text: string) {
    const q = text.trim()
    if (!q || busy) return
    const id = nextId.current++
    const current = new AbortController()
    controller.current = current
    followBottom.current = true
    const started = clock()
    setTurns((all) => [...all, {
      id, question: q, folder, phase: 'searching', sources: [], streamed: '',
      result: null, error: null, started,
    }])
    setQuestion('')
    setNow(started)
    if (input.current) input.current.style.height = ''

    try {
      if (!llm) {
        const found = await api.search(q, folder)
        update(id, { sources: found, phase: 'done' })
        return
      }
      let streamed = ''
      let finished = false
      for await (const ev of ask(q, folder, current.signal)) {
        if (ev.event === 'sources') {
          update(id, { sources: ev.sources, phase: 'reading' })
        } else if (ev.event === 'delta') {
          streamed += ev.text
          update(id, { streamed, phase: 'writing' })
        } else if (ev.event === 'retry') {
          update(id, { phase: 'retry' })
        } else if (ev.event === 'answer') {
          // The parsed answer replaces the streamed text: it may be a refusal, or a
          // rewrite with citations.
          update(id, { result: ev, phase: 'done' })
          finished = true
        }
      }
      if (!finished) update(id, { phase: 'error', error: 'The answer was interrupted.' })
    } catch (err) {
      if (current.signal.aborted) {
        update(id, { phase: 'stopped' })
      } else {
        update(id, { phase: 'error', error: err instanceof Error ? err.message : String(err) })
      }
    }
  }

  function submit(event?: FormEvent) {
    event?.preventDefault()
    void send(question)
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  function newConversation() {
    controller.current?.abort()
    setTurns([])
    setFocus(null)
    setTopicsRound((r) => r + 1)
    input.current?.focus()
  }

  return (
    <section className="chat">
      <header className="topbar">
        {leading}
        <button type="button" className="brand" onClick={onHome} title="Back to the start page">
          <LogoMark size={26} />
          <span>RAG Library</span>
        </button>
        <div className="topbar-actions">
          <button
            type="button"
            className="icon-btn"
            title="New conversation"
            aria-label="New conversation"
            onClick={newConversation}
            disabled={turns.length === 0}
          >
            <ComposeIcon />
          </button>
        </div>
      </header>

      <div
        className="messages"
        ref={scroller}
        onScroll={(e) => {
          const el = e.currentTarget
          followBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120
        }}
      >
        <div className="messages-inner">
          {turns.length === 0 ? (
            <div className="empty-chat">
              <LogoMark size={44} />
              <h2>What would you like to know?</h2>
              <p>
                Ask in English or Romanian. I search{' '}
                {folder === null ? 'all your documents' : `the “${folder}” folder`} and show you
                the sources.
              </p>
              {topics !== null && topics.length > 0 && (
                <>
                  <div className="topics-head">
                    <span>From your library</span>
                    <button
                      type="button"
                      className="icon-btn subtle"
                      title="Other topics"
                      aria-label="Other topics"
                      onClick={() => setTopicsRound((r) => r + 1)}
                    >
                      <RefreshIcon size={15} />
                    </button>
                  </div>
                  <div className="examples">
                    {topics.map((topic) => (
                      <button
                        key={`${topic.source}|${topic.heading}`}
                        type="button"
                        className="example"
                        onClick={() => void send(topicQuestion(topic))}
                        disabled={busy}
                      >
                        <span className="example-title">{topic.heading}</span>
                        <span className="example-source">
                          <FileIcon size={12} />
                          {topic.source.split('/').pop()}
                          {topic.page ? ` · p. ${topic.page}` : ''}
                        </span>
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          ) : (
            turns.map((turn) => (
              <TurnView
                key={turn.id}
                turn={turn}
                llm={llm}
                now={now}
                focus={focus?.turn === turn.id ? focus : null}
                onCite={(n) => setFocus({ turn: turn.id, n, nonce: Date.now() })}
                onOpen={onOpen}
              />
            ))
          )}
        </div>
      </div>

      <form className="composer-wrap" onSubmit={submit}>
        <div className="composer glass">
          <span className="scope" title="Where I search">
            {folder === null ? <StackIcon size={14} /> : <FolderIcon size={14} />}
            {scopeName(folder)}
          </span>
          <textarea
            ref={input}
            rows={1}
            value={question}
            onChange={(e) => {
              setQuestion(e.target.value)
              e.target.style.height = ''
              e.target.style.height = `${Math.min(e.target.scrollHeight, 180)}px`
            }}
            onKeyDown={onKeyDown}
            placeholder={llm ? 'Ask anything about your documents…' : 'Search your documents…'}
            maxLength={1000}
            aria-label="Question"
          />
          {busy ? (
            <button type="button" className="send stop" onClick={() => controller.current?.abort()} aria-label="Stop">
              <StopIcon size={16} />
            </button>
          ) : (
            <button type="submit" className="send" disabled={!question.trim()} aria-label="Send">
              <ArrowUpIcon size={18} />
            </button>
          )}
        </div>
        <p className="composer-hint">
          {llm
            ? 'Each question is answered on its own, only from your documents.'
            : 'The language model is off: showing matching passages only.'}
        </p>
      </form>
    </section>
  )
}

interface TurnProps {
  turn: Turn
  llm: boolean
  now: number
  focus: { n: number; nonce: number } | null
  onCite: (n: number) => void
  onOpen: (target: ViewTarget) => void
}

function TurnView({ turn, llm, now, focus, onCite, onOpen }: TurnProps) {
  const { phase, result, streamed, sources } = turn
  const seconds = Math.max(0, Math.round((now - turn.started) / 1000))

  return (
    <article className="turn">
      <div className="bubble user">{turn.question}</div>

      <div className="bubble assistant">
        <div className="answer-meta">
          {turn.folder === null ? <StackIcon size={13} /> : <FolderIcon size={13} />}
          {scopeName(turn.folder)}
        </div>

        {isBusy(phase) && result === null && !streamed && (
          <p className="working">
            <span className="pulse" />
            {WORKING[phase]}
            <span className="seconds">{seconds} s</span>
          </p>
        )}

        {result === null && streamed && (
          <>
            <AnswerText text={streamed} sourceCount={sources.length} onCite={onCite} />
            {phase === 'retry' && <p className="working small"><span className="pulse" />Adding citations</p>}
          </>
        )}
        {result?.found && <AnswerText text={result.text} sourceCount={sources.length} onCite={onCite} />}
        {result && !result.found && (
          <div className="refusal">
            <p>{result.text}</p>
            {result.refusal && <p className="refusal-why">{REFUSALS[result.refusal]}</p>}
          </div>
        )}
        {phase === 'stopped' && <p className="muted">Stopped.</p>}
        {turn.error && <p className="note error">{turn.error}</p>}
        {!llm && phase === 'done' && sources.length === 0 && <p className="muted">Nothing found.</p>}

        {result && (
          <p className="timing">
            search {result.timings.retrieval ?? '–'} s · answer {result.timings.generation ?? '–'} s
          </p>
        )}

        <SourceList
          sources={sources}
          cited={result?.cited ?? []}
          focus={focus}
          defaultOpen={!llm}
          onOpen={onOpen}
        />
      </div>
    </article>
  )
}
