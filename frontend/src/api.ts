// Typed client for the FastAPI server in rag/api/app.py. The shapes below mirror the JSON
// it returns; when an endpoint changes there, it changes here too.

export interface Status {
  documents: number
  chunks: number
  folders: number
  llm: boolean
  model: string | null
  max_upload_mb: number
}

export interface Folder {
  name: string
  documents: number
  chunks: number
}

export interface DocumentInfo {
  source: string // "<folder>/<file>"
  folder: string
  language: string | null
  chunks: number
  chars: number
  pages: number
}

export interface Source {
  n: number // the number the model cites it by, [n]
  source: string
  page: number | null
  heading_path: string
  text: string
  start_char: number // where the passage sits in the document's extracted text
  end_char: number
  score: number
}

// A document's text as the server extracted it, with where each PDF page starts.
export interface DocumentText {
  source: string
  format: string | null
  text: string
  page_starts: number[] | null
}

// A spreadsheet sheet as rows of cell text. A merge anchors at (r, c) and spans
// `rows` x `cols` cells; widths are Excel column widths, in characters.
export interface Sheet {
  name: string
  rows: string[][]
  merges: { r: number; c: number; rows: number; cols: number }[]
  widths: number[]
  truncated: boolean
}

// A section heading from the library, offered as a starting question.
export interface Topic {
  source: string
  heading: string
  page: number | null
}

export interface UploadReport {
  file: string
  ok: boolean
  source?: string
  chunks?: number
  replaced_chunks?: number
  language?: string
  seconds?: number
  error?: string
  status?: number
}

export interface AnswerResult {
  text: string
  found: boolean
  refusal: 'no_sources' | 'model' | 'no_citations' | null
  language: string | null
  cited: number[]
  invalid_citations: number[]
  timings: { retrieval?: number; generation?: number; [key: string]: unknown }
}

// The server-sent events of POST /api/ask, in the order they arrive.
export type AskEvent =
  | { event: 'sources'; language: string | null; sources: Source[] }
  | { event: 'delta'; text: string }
  | { event: 'retry' }
  | ({ event: 'answer' } & AnswerResult)

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) throw await errorOf(response)
  return (await response.json()) as T
}

async function errorOf(response: Response): Promise<ApiError> {
  let message = response.statusText
  try {
    const body = (await response.json()) as { detail?: unknown }
    if (body.detail !== undefined) {
      message = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    }
  } catch {
    // not JSON: keep the status text
  }
  return new ApiError(response.status, message)
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

// A source such as "Practică 2026/ghid.pdf" goes into the URL segment by segment, so the
// "/" between folder and file stays a path separator and everything else is escaped.
const sourcePath = (source: string) => source.split('/').map(encodeURIComponent).join('/')

// The original file, for the browser to show (PDF) or download (other formats).
export const fileUrl = (source: string) => `/api/files/${sourcePath(source)}`

export const api = {
  status: () => request<Status>('/api/status'),

  folders: () => request<Folder[]>('/api/folders'),
  createFolder: (name: string) => request<{ name: string }>('/api/folders', json({ name })),
  deleteFolder: (name: string, withDocuments: boolean) =>
    request<{ documents_removed: number }>(
      `/api/folders/${encodeURIComponent(name)}?with_documents=${withDocuments}`,
      { method: 'DELETE' },
    ),

  documents: (folder: string | null) =>
    request<DocumentInfo[]>(
      folder === null ? '/api/documents' : `/api/documents?folder=${encodeURIComponent(folder)}`,
    ),
  upload: (folder: string, file: File, replace: boolean) => {
    const form = new FormData()
    form.append('files', file)
    form.append('replace', String(replace))
    return request<UploadReport[]>(`/api/folders/${encodeURIComponent(folder)}/documents`, {
      method: 'POST',
      body: form,
    })
  },
  deleteDocument: (source: string) =>
    request<{ removed: string }>(`/api/documents/${sourcePath(source)}`, { method: 'DELETE' }),

  search: (question: string, folder: string | null) =>
    request<Source[]>('/api/search', json({ question, folder })),

  documentText: (source: string) => request<DocumentText>(`/api/text/${sourcePath(source)}`),
  sheets: (source: string) => request<{ sheets: Sheet[] }>(`/api/sheets/${sourcePath(source)}`),

  topics: (folder: string | null) =>
    request<Topic[]>(folder === null ? '/api/topics' : `/api/topics?folder=${encodeURIComponent(folder)}`),
}

// POST /api/ask, read as a stream of server-sent events. EventSource cannot send a POST
// body, so the response is read by hand: events are separated by a blank line, and a
// network chunk can end in the middle of one, hence the buffer.
export async function* ask(
  question: string,
  folder: string | null,
  signal: AbortSignal,
): AsyncGenerator<AskEvent> {
  const response = await fetch('/api/ask', { ...json({ question, folder }), signal })
  if (!response.ok) throw await errorOf(response)
  if (!response.body) throw new ApiError(500, 'the server sent no body')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let cut: number
    while ((cut = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, cut)
      buffer = buffer.slice(cut + 2)
      if (block.startsWith('data: ')) yield JSON.parse(block.slice(6)) as AskEvent
    }
  }
}
