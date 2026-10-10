// Line icons drawn inline, so the interface needs no icon library or network access.
import type { ReactNode, SVGProps } from 'react'

type IconProps = SVGProps<SVGSVGElement> & { size?: number }

function Icon({ size = 18, children, ...rest }: IconProps & { children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...rest}
    >
      {children}
    </svg>
  )
}

export const ArrowUpIcon = (p: IconProps) => (
  <Icon {...p}><path d="M12 19V5" /><path d="m5 12 7-7 7 7" /></Icon>
)
export const ArrowRightIcon = (p: IconProps) => (
  <Icon {...p}><path d="M5 12h14" /><path d="m12 5 7 7-7 7" /></Icon>
)
export const StopIcon = (p: IconProps) => (
  <Icon {...p}><rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" stroke="none" /></Icon>
)
export const PlusIcon = (p: IconProps) => (
  <Icon {...p}><path d="M12 5v14" /><path d="M5 12h14" /></Icon>
)
export const FolderIcon = (p: IconProps) => (
  <Icon {...p}><path d="M3 7.5A2.5 2.5 0 0 1 5.5 5h3.6l2 2.2h7.4A2.5 2.5 0 0 1 21 9.7v7.8a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 17.5z" /></Icon>
)
export const StackIcon = (p: IconProps) => (
  <Icon {...p}><path d="m12 3 9 5-9 5-9-5z" /><path d="m3 13 9 5 9-5" /></Icon>
)
export const FileIcon = (p: IconProps) => (
  <Icon {...p}><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5" /></Icon>
)
export const TrashIcon = (p: IconProps) => (
  <Icon {...p}><path d="M4 7h16" /><path d="M10 11v6" /><path d="M14 11v6" /><path d="M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12" /><path d="M9 7V4h6v3" /></Icon>
)
export const UploadIcon = (p: IconProps) => (
  <Icon {...p}><path d="M12 15V4" /><path d="m7 9 5-5 5 5" /><path d="M5 15v3a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-3" /></Icon>
)
export const ChevronIcon = (p: IconProps) => (
  <Icon {...p}><path d="m9 6 6 6-6 6" /></Icon>
)
export const ComposeIcon = (p: IconProps) => (
  <Icon {...p}><path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z" /></Icon>
)
export const SidebarIcon = (p: IconProps) => (
  <Icon {...p}><rect x="3" y="4" width="18" height="16" rx="3" /><path d="M9 4v16" /></Icon>
)
export const ExternalIcon = (p: IconProps) => (
  <Icon {...p}><path d="M14 4h6v6" /><path d="M20 4 11 13" /><path d="M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4" /></Icon>
)
export const RefreshIcon = (p: IconProps) => (
  <Icon {...p}><path d="M20 11a8 8 0 1 0-2.3 5.7" /><path d="M20 4v7h-7" /></Icon>
)
export const CloseIcon = (p: IconProps) => (
  <Icon {...p}><path d="M6 6l12 12" /><path d="M18 6 6 18" /></Icon>
)

// The app's mark: three overlapping pages, the corpus.
export const LogoMark = ({ size = 28 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
    <defs>
      <linearGradient id="logo-a" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stopColor="#7c5cff" />
        <stop offset="1" stopColor="#2f8cff" />
      </linearGradient>
      <linearGradient id="logo-b" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stopColor="#ff7ab6" />
        <stop offset="1" stopColor="#ff9f5a" />
      </linearGradient>
    </defs>
    <rect x="3" y="7" width="17" height="21" rx="5" fill="url(#logo-b)" opacity="0.85" />
    <rect x="12" y="4" width="17" height="21" rx="5" fill="url(#logo-a)" opacity="0.92" />
    <path d="M16.5 11.5h8M16.5 15.5h8M16.5 19.5h5" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" />
  </svg>
)
