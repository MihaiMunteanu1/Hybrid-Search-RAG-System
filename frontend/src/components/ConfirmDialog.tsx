import { useCallback, useEffect, useState } from 'react'

interface Request {
  title: string
  message: string
  action: string
  resolve: (ok: boolean) => void
}

// A confirmation sheet instead of window.confirm: same promise-based use, styled like the
// rest of the interface. Render `dialog` once; call `confirm(...)` and await the answer.
export function useConfirm() {
  const [request, setRequest] = useState<Request | null>(null)

  const confirm = useCallback(
    (title: string, message: string, action = 'Delete') =>
      new Promise<boolean>((resolve) => setRequest({ title, message, action, resolve })),
    [],
  )

  const close = useCallback(
    (ok: boolean) => {
      request?.resolve(ok)
      setRequest(null)
    },
    [request],
  )

  useEffect(() => {
    if (!request) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [request, close])

  const dialog = request && (
    <div className="sheet-backdrop" onClick={() => close(false)}>
      <div
        className="sheet glass"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="sheet-title"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 id="sheet-title">{request.title}</h3>
        <p>{request.message}</p>
        <div className="sheet-actions">
          <button type="button" className="btn ghost" onClick={() => close(false)}>
            Cancel
          </button>
          <button type="button" className="btn danger" autoFocus onClick={() => close(true)}>
            {request.action}
          </button>
        </div>
      </div>
    </div>
  )

  return [dialog, confirm] as const
}
