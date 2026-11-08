import type { SVGProps } from "react";

export type IconName =
  | "architecture"
  | "wiki"
  | "copilot"
  | "tasks"
  | "documents"
  | "settings"
  | "search"
  | "plus"
  | "arrow"
  | "spark"
  | "send"
  | "trash"
  | "upload"
  | "source"
  | "person"
  | "calendar"
  | "check"
  | "external"
  | "menu"
  | "close"
  | "chevron"
  | "zoomIn"
  | "zoomOut"
  | "fit"
  | "lens"
  | "enter"
  | "level-up"
  | "route"
  | "port"
  | "sealed";

const paths: Record<IconName, React.ReactNode> = {
  architecture: <><rect x="3" y="4" width="7" height="6" rx="1"/><rect x="14" y="4" width="7" height="6" rx="1"/><rect x="8.5" y="15" width="7" height="5" rx="1"/><path d="M6.5 10v2.5h11V10M12 12.5V15"/></>,
  wiki: <><path d="M4 5.5A3.5 3.5 0 0 1 7.5 2H11v17H7.5A3.5 3.5 0 0 0 4 22V5.5ZM20 5.5A3.5 3.5 0 0 0 16.5 2H13v17h3.5A3.5 3.5 0 0 1 20 22V5.5Z"/></>,
  copilot: <><path d="M12 2v4M12 18v4M4.9 4.9l2.8 2.8M16.3 16.3l2.8 2.8M2 12h4M18 12h4M4.9 19.1l2.8-2.8M16.3 7.7l2.8-2.8"/><circle cx="12" cy="12" r="4"/></>,
  tasks: <><path d="M9 5h11M9 12h11M9 19h11"/><path d="m3 5 1.5 1.5L7 3.5M3 12l1.5 1.5L7 10.5M3 19l1.5 1.5L7 17.5"/></>,
  documents: <><path d="M6 2h8l4 4v16H6zM14 2v5h4"/><path d="M9 12h6M9 16h6"/></>,
  settings: <><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1A1.7 1.7 0 0 0 9 4.6 1.7 1.7 0 0 0 10 3V2.8h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1Z"/></>,
  search: <><circle cx="11" cy="11" r="7"/><path d="m16 16 5 5"/></>,
  plus: <path d="M12 5v14M5 12h14"/>,
  arrow: <path d="m9 5 7 7-7 7"/>,
  spark: <><path d="m12 2 1.4 5.6L19 9l-5.6 1.4L12 16l-1.4-5.6L5 9l5.6-1.4L12 2Z"/><path d="m19 15 .7 2.3L22 18l-2.3.7L19 21l-.7-2.3L16 18l2.3-.7L19 15Z"/></>,
  send: <><path d="m3 3 18 9-18 9 4-9-4-9Z"/><path d="M7 12h14"/></>,
  trash: <><path d="M4 7h16M9 7V4h6v3M7 7l1 14h8l1-14"/><path d="M10 11v6M14 11v6"/></>,
  upload: <><path d="M12 16V3m-5 5 5-5 5 5M4 14v7h16v-7"/></>,
  source: <><path d="M8 3 3 8l5 5M16 11l5 5-5 5M14 3l-4 18"/></>,
  person: <><circle cx="12" cy="8" r="4"/><path d="M4 22c.6-5.3 3.3-8 8-8s7.4 2.7 8 8"/></>,
  calendar: <><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 2v6M17 2v6M3 10h18"/></>,
  check: <path d="m4 12 5 5L20 6"/>,
  external: <><path d="M14 4h6v6M20 4 10 14"/><path d="M18 13v7H4V6h7"/></>,
  menu: <path d="M4 7h16M4 12h16M4 17h16"/>,
  close: <path d="m5 5 14 14M19 5 5 19"/>,
  chevron: <path d="m9 6 6 6-6 6"/>,
  zoomIn: <><circle cx="11" cy="11" r="7"/><path d="M11 8v6M8 11h6M16 16l5 5"/></>,
  zoomOut: <><circle cx="11" cy="11" r="7"/><path d="M8 11h6M16 16l5 5"/></>,
  fit: <><path d="M3 9V3h6M21 9V3h-6M3 15v6h6M21 15v6h-6"/></>,
  lens: <><path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 13 9 5 9-5"/></>,
  enter: <><path d="M12 4v12"/><path d="m7 11 5 5 5-5"/><path d="M4 20h16"/></>,
  "level-up": <><path d="M12 20V8"/><path d="m7 13 5-5 5 5"/><path d="M4 4h16"/></>,
  route: <><circle cx="5" cy="6" r="2"/><circle cx="19" cy="18" r="2"/><path d="M5 8v4a4 4 0 0 0 4 4h8"/></>,
  port: <><rect x="4" y="9" width="16" height="6" rx="1"/><path d="M8 9V6M16 15v3"/></>,
  sealed: <><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 4l16 16M20 4 4 20"/></>,
};

export function Icon({ name, ...props }: { name: IconName } & SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" {...props}>
      {paths[name]}
    </svg>
  );
}
