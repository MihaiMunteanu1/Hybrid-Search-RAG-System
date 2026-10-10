import { useCallback, useEffect, useState, type DragEvent, type FormEvent } from 'react'
import { api, type DocumentInfo, type Folder, type Status } from '../api'
import { CloseIcon, FileIcon, FolderIcon, PlusIcon, StackIcon, TrashIcon, UploadIcon } from '../icons'
import { useConfirm } from './ConfirmDialog'
import type { ViewTarget } from './Viewer'

interface Props {
  status: Status | null
  offline: boolean
  folders: Folder[]
  current: string | null
  onSelect: (folder: string | null) => void
  onCreated: (folder: string) => void
  onChanged: () => void          // documents came or went: counts need reloading
  onFolderDeleted: () => void
  onOpen: (target: ViewTarget) => void
  onClose: () => void            // narrow screens only: hide the panel
}

const message = (err: unknown) => (err instanceof Error ? err.message : String(err))
const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`

export function Library(props: Props) {
  const { status, offline, folders, current, onSelect, onCreated, onClose } = props
  const [creating, setCreating] = useState(false)
  const [name, setName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const total = folders.reduce((sum, f) => sum + f.documents, 0)

  async function create(event: FormEvent) {
    event.preventDefault()
    const trimmed = name.trim()
    if (!trimmed) return
    try {
      const created = await api.createFolder(trimmed)
      setName('')
      setCreating(false)
      setError(null)
      onCreated(created.name)
    } catch (err) {
      setError(message(err))
    }
  }

  return (
    <aside className="library glass">
      <div className="library-head">
        <h2>Library</h2>
        <button
          type="button"
          className="icon-btn"
          title="New folder"
          aria-label="New folder"
          onClick={() => { setCreating((c) => !c); setError(null) }}
        >
          <PlusIcon />
        </button>
        <button type="button" className="icon-btn only-narrow" aria-label="Close" onClick={onClose}>
          <CloseIcon />
        </button>
      </div>

      {creating && (
        <form className="new-folder" onSubmit={create}>
          <input
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Escape') setCreating(false) }}
            placeholder="Folder name"
            maxLength={64}
            aria-label="New folder name"
          />
          <button type="submit" className="btn primary small" disabled={!name.trim()}>
            Create
          </button>
        </form>
      )}
      {error && <p className="note error">{error}</p>}

      <nav className="folders" aria-label="Folders">
        <button
          type="button"
          className={current === null ? 'folder-row active' : 'folder-row'}
          onClick={() => onSelect(null)}
        >
          <span className="folder-icon all"><StackIcon size={16} /></span>
          <span className="folder-name">All documents</span>
          <span className="count">{total}</span>
        </button>

        {folders.map((f) => (
          <div key={f.name} className={current === f.name ? 'folder open' : 'folder'}>
            <button
              type="button"
              className={current === f.name ? 'folder-row active' : 'folder-row'}
              onClick={() => onSelect(current === f.name ? null : f.name)}
              aria-expanded={current === f.name}
            >
              <span className="folder-icon"><FolderIcon size={16} /></span>
              <span className="folder-name">{f.name}</span>
              <span className="count">{f.documents}</span>
            </button>
            {current === f.name && (
              <FolderContents
                key={f.name}
                folder={f.name}
                maxUploadMb={status?.max_upload_mb ?? 50}
                onChanged={props.onChanged}
                onFolderDeleted={props.onFolderDeleted}
                onOpen={props.onOpen}
              />
            )}
          </div>
        ))}
        {folders.length === 0 && !offline && (
          <p className="note">No folders yet. Create one with +.</p>
        )}
      </nav>

      <footer className="library-foot">
        <span className={offline ? 'dot off' : status?.llm ? 'dot on' : 'dot idle'} />
        {offline
          ? 'Server not responding'
          : status === null
            ? 'Connecting…'
            : status.llm
              ? `${status.model?.replace(/\.gguf$/, '') ?? 'model'} · local`
              : 'No model · search only'}
        {status && !offline && <span className="foot-meta">{plural(status.chunks, 'chunk', 'chunks')}</span>}
      </footer>
    </aside>
  )
}

interface LogLine {
  id: number
  state: 'busy' | 'ok' | 'error'
  text: string
}

interface ContentsProps {
  folder: string
  maxUploadMb: number
  onChanged: () => void
  onFolderDeleted: () => void
  onOpen: (target: ViewTarget) => void
}

// The open folder: its files, upload, and deletion. Mounted per folder (key), so the
// upload log and the replace toggle start clean whenever another folder is opened.
function FolderContents({ folder, maxUploadMb, onChanged, onFolderDeleted, onOpen }: ContentsProps) {
  const [documents, setDocuments] = useState<DocumentInfo[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [log, setLog] = useState<LogLine[]>([])
  const [replace, setReplace] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [dialog, confirm] = useConfirm()

  const load = useCallback(async () => {
    try {
      setDocuments(await api.documents(folder))
      setError(null)
    } catch (err) {
      setError(message(err))
    }
  }, [folder])

  useEffect(() => {
    let cancelled = false
    api.documents(folder).then(
      (docs) => { if (!cancelled) setDocuments(docs) },
      (err: unknown) => { if (!cancelled) setError(message(err)) },
    )
    return () => { cancelled = true }
  }, [folder])

  async function remove(source: string) {
    const file = source.split('/').slice(1).join('/')
    if (!(await confirm('Delete this document?', `“${file}” will be removed from the library and from disk.`))) return
    try {
      await api.deleteDocument(source)
    } catch (err) {
      setError(message(err))
    }
    await load()
    onChanged()
  }

  async function deleteFolder() {
    const count = documents?.length ?? 0
    const ok = await confirm(
      'Delete this folder?',
      count
        ? `“${folder}” and the ${plural(count, 'document', 'documents')} in it will be deleted.`
        : `The empty folder “${folder}” will be deleted.`,
    )
    if (!ok) return
    try {
      await api.deleteFolder(folder, count > 0)
      onFolderDeleted()
    } catch (err) {
      setError(message(err))
    }
  }

  // One file at a time: each is read and embedded on the server's CPU, and each gets
  // its own line with its own result.
  async function upload(files: File[]) {
    if (files.length === 0) return
    setUploading(true)
    for (const file of files) {
      const id = Date.now() + Math.random()
      const line = (state: LogLine['state'], text: string) =>
        setLog((lines) => [{ id, state, text }, ...lines.filter((l) => l.id !== id)].slice(0, 6))
      if (file.size > maxUploadMb * 2 ** 20) {
        line('error', `${file.name}: over the ${maxUploadMb} MB limit`)
        continue
      }
      line('busy', `${file.name}: indexing…`)
      try {
        const [report] = await api.upload(folder, file, replace)
        if (report.ok) {
          line('ok', `${file.name}: ${plural(report.chunks ?? 0, 'chunk', 'chunks')}, ${report.seconds} s`)
        } else {
          line('error', `${file.name}: ${report.error}`)
        }
      } catch (err) {
        line('error', `${file.name}: ${message(err)}`)
      }
      await load()
      onChanged()
    }
    setUploading(false)
  }

  function onDrop(event: DragEvent) {
    event.preventDefault()
    setDragging(false)
    void upload([...event.dataTransfer.files])
  }

  return (
    <div className="folder-contents">
      {error && <p className="note error">{error}</p>}

      {documents === null ? (
        <p className="note">Loading…</p>
      ) : documents.length === 0 ? (
        <p className="note">This folder is empty.</p>
      ) : (
        <ul className="files">
          {documents.map((d) => (
            <li key={d.source} className="file">
              <button
                type="button"
                className="file-open"
                title={`Open ${d.source}`}
                onClick={() => onOpen({ source: d.source })}
              >
                <FileIcon size={15} />
                <span className="file-name">{d.source.split('/').slice(1).join('/')}</span>
                <span className="file-meta">
                  {d.language?.toUpperCase()}
                  {d.pages ? ` · ${d.pages} p.` : ''}
                </span>
              </button>
              <button
                type="button"
                className="icon-btn subtle"
                title="Delete document"
                aria-label={`Delete ${d.source}`}
                disabled={uploading}
                onClick={() => remove(d.source)}
              >
                <TrashIcon size={15} />
              </button>
            </li>
          ))}
        </ul>
      )}

      <label
        className={dragging ? 'dropzone over' : 'dropzone'}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <UploadIcon size={18} />
        <span>
          {uploading ? 'Uploading…' : 'Drop files here or browse'}
          <small>PDF, DOCX, PPTX, XLSX, CSV, HTML, MD, TXT · up to {maxUploadMb} MB</small>
        </span>
        <input
          type="file"
          multiple
          hidden
          onChange={(e) => {
            void upload([...(e.target.files ?? [])])
            e.target.value = ''
          }}
        />
      </label>

      <div className="folder-actions">
        <label className="toggle">
          <input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} />
          <span className="switch" />
          Replace files with the same name
        </label>
        <button type="button" className="link-btn danger" onClick={deleteFolder}>
          Delete folder
        </button>
      </div>

      {log.length > 0 && (
        <ul className="upload-log">
          {log.map((l) => (
            <li key={l.id} className={l.state}>{l.text}</li>
          ))}
        </ul>
      )}
      {dialog}
    </div>
  )
}
