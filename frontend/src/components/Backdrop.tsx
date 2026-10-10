import { useEffect, useRef } from 'react'

// Travel of each shape, in pixels, when the pointer is at the edge of the window: the
// shapes lean toward the pointer at different depths (negative ones lean away).
const DEPTHS = [36, -44, 28, -20]

// The abstract shapes behind the interface, a soft light that follows the pointer, and a
// grid of dots that shows only around it. Pointer moves never re-render React: the
// position is written straight to the elements, once per animation frame, and only
// `transform` and a small background offset change, which the browser handles cheaply.
export function Backdrop({ calm }: { calm: boolean }) {
  const light = useRef<HTMLSpanElement>(null)
  const layers = useRef<(HTMLSpanElement | null)[]>([])

  useEffect(() => {
    // The light follows the pointer directly, so it stays on for everyone. The shapes'
    // drift toward it is motion the reader did not ask for: off when the system says so.
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    let frame = 0
    let x = 0
    let y = 0

    const apply = () => {
      frame = 0
      const el = light.current
      if (el) {
        el.style.transform = `translate3d(${x}px, ${y}px, 0)`
        // The dots stay fixed to the page while the window onto them moves.
        el.style.setProperty('--gx', `${-x}px`)
        el.style.setProperty('--gy', `${-y}px`)
        el.classList.add('on')
      }
      if (reduced) return
      const dx = x / window.innerWidth - 0.5
      const dy = y / window.innerHeight - 0.5
      layers.current.forEach((layer, i) => {
        if (layer) layer.style.transform = `translate3d(${dx * DEPTHS[i]}px, ${dy * DEPTHS[i]}px, 0)`
      })
    }
    const onMove = (event: PointerEvent) => {
      x = event.clientX
      y = event.clientY
      if (!frame) frame = requestAnimationFrame(apply)
    }
    const onOut = (event: MouseEvent) => {
      if (!event.relatedTarget) light.current?.classList.remove('on')   // left the window
    }

    window.addEventListener('pointermove', onMove, { passive: true })
    window.addEventListener('mouseout', onOut)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('mouseout', onOut)
      cancelAnimationFrame(frame)
    }
  }, [])

  return (
    <div className={calm ? 'backdrop calm' : 'backdrop'} aria-hidden="true">
      {DEPTHS.map((_, i) => (
        <span key={i} className="parallax" ref={(el) => { layers.current[i] = el }}>
          <span className={`orb orb-${i + 1}`} />
        </span>
      ))}
      <span className="light" ref={light}>
        <span className="dots" />
      </span>
    </div>
  )
}
