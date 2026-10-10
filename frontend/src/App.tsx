import { useCallback, useEffect, useState } from 'react'
import { api, type Folder, type Status } from './api'
import { Backdrop } from './components/Backdrop'
import { Chat } from './components/Chat'
import { Library } from './components/Library'
import { Viewer, type ViewTarget } from './components/Viewer'
import { Welcome } from './components/Welcome'
import { SidebarIcon } from './icons'

// The welcome page is shown once per browser tab session.
const ENTERED = 'rag-entered'

function readEntered(): boolean {
  try {
    return sessionStorage.getItem(ENTERED) === '1'
  } catch {
    return false
  }
}

function writeEntered(value: boolean) {
  try {
    if (value) sessionStorage.setItem(ENTERED, '1')
    else sessionStorage.removeItem(ENTERED)
  } catch {
    // storage unavailable (private mode): the welcome page simply shows again
  }
}

export default function App() {
  const [entered, setEntered] = useState(readEntered)
  const [status, setStatus] = useState<Status | null>(null)
  const [folders, setFolders] = useState<Folder[]>([])
  const [current, setCurrent] = useState<string | null>(null)
  const [offline, setOffline] = useState(false)
  const [libraryOpen, setLibraryOpen] = useState(false)
  const [viewing, setViewing] = useState<ViewTarget | null>(null)
  const closeViewer = useCallback(() => setViewing(null), [])

  // Status and folder counts change together whenever a document comes or goes.
  const apply = useCallback(([s, f]: [Status, Folder[]]) => {
    setStatus(s)
    setFolders(f)
    setOffline(false)
    // The selected folder may have just been deleted.
    setCurrent((c) => (c !== null && !f.some((x) => x.name === c) ? null : c))
  }, [])

  const refresh = useCallback(() => {
    Promise.all([api.status(), api.folders()]).then(apply, () => setOffline(true))
  }, [apply])

  useEffect(() => {
    let cancelled = false
    Promise.all([api.status(), api.folders()]).then(
      (result) => { if (!cancelled) apply(result) },
      () => { if (!cancelled) setOffline(true) },
    )
    return () => { cancelled = true }
  }, [apply])

  const enter = useCallback(() => {
    writeEntered(true)
    setEntered(true)
  }, [])

  return (
    <>
      <Backdrop calm={entered} />

      {!entered ? (
        <Welcome status={status} onEnter={enter} />
      ) : (
        <div className={libraryOpen ? 'workspace library-open' : 'workspace'}>
          <Library
            status={status}
            offline={offline}
            folders={folders}
            current={current}
            onSelect={setCurrent}
            onCreated={(name) => {
              setCurrent(name)
              refresh()
            }}
            onChanged={refresh}
            onFolderDeleted={() => {
              setCurrent(null)
              refresh()
            }}
            onOpen={setViewing}
            onClose={() => setLibraryOpen(false)}
          />
          <div className="scrim only-narrow" onClick={() => setLibraryOpen(false)} />
          <Chat
            folder={current}
            llm={status?.llm ?? false}
            onOpen={setViewing}
            onHome={() => { writeEntered(false); setEntered(false) }}
            leading={
              <button
                type="button"
                className="icon-btn only-narrow"
                title="Library"
                aria-label="Library"
                onClick={() => setLibraryOpen(true)}
              >
                <SidebarIcon />
              </button>
            }
          />
        </div>
      )}

      {viewing && <Viewer key={`${viewing.source}|${viewing.page}|${viewing.span}`} target={viewing} onClose={closeViewer} />}
    </>
  )
}
