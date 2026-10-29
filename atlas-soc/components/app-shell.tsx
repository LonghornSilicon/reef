"use client";

import Image from "next/image";
import Link from "next/link";
import { signOut } from "next-auth/react";
import { usePathname } from "next/navigation";
import { useState } from "react";
import type { Actor } from "@/lib/domain";
import { Icon, type IconName } from "@/components/icons";

const navigation: Array<{ href: string; label: string; icon: IconName }> = [
  { href: "/architecture", label: "Architecture", icon: "architecture" },
  { href: "/wiki", label: "Wiki", icon: "wiki" },
  { href: "/copilot", label: "Copilot", icon: "copilot" },
  { href: "/tasks", label: "Tasks", icon: "tasks" },
  { href: "/documents", label: "Documents", icon: "documents" },
];

export function AppShell({
  actor,
  anonymous,
  canSignIn,
  children,
}: {
  actor: Actor;
  /** Public read-only visitor: no account, so no account menu. */
  anonymous: boolean;
  /** Whether an identity provider exists to sign in with. */
  canSignIn: boolean;
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);
  const current = navigation.find((item) => pathname.startsWith(item.href)) ?? navigation[0];

  return (
    <div className="atlas-shell">
      <header className="workspace-bar">
        <button className="mobile-menu" type="button" aria-label="Open navigation" onClick={() => setMobileOpen(true)}>
          <Icon name="menu" />
        </button>
        <Link href="/architecture" className="atlas-wordmark" aria-label="Atlas home">
          <Image src="/logo.png" width={36} height={36} alt="" priority />
          <span>Atlas</span>
        </Link>
        <div className="workspace-identity">
          <span>Workspace</span>
          <strong>Coral SoC</strong>
        </div>
        <div className="surface-location" aria-live="polite">
          <Icon name={current.icon} />
          <span>{current.label}</span>
        </div>
        <div className="workspace-tools">
          <Link className="command-button" href="/wiki" aria-label="Search Atlas">
            <Icon name="search" />
            <span>Find evidence</span>
            <kbd>/</kbd>
          </Link>
          <span className={`role-stamp role-${anonymous ? "public" : actor.role}`}>
            {anonymous ? "read only" : actor.role}
          </span>
          {anonymous ? (
            canSignIn && (
              <Link className="command-button" href="/signin">
                <Icon name="settings" />
                <span>Sign in</span>
              </Link>
            )
          ) : (
            <details className="account-menu">
              <summary aria-label={`Open account menu for ${actor.name}`}>
                <span className="avatar">{actor.name.split(/\s+/).map((part) => part[0]).slice(0, 2).join("")}</span>
              </summary>
              <div className="account-popover">
                <strong>{actor.name}</strong>
                <span>{actor.email}</span>
                <small>{actor.role}</small>
                <button type="button" onClick={() => void signOut({ redirectTo: "/signin" })}>Sign out</button>
              </div>
            </details>
          )}
        </div>
      </header>

      <aside className={`instrument-nav${mobileOpen ? " open" : ""}`} aria-label="Atlas sections">
        <button className="mobile-close" type="button" aria-label="Close navigation" onClick={() => setMobileOpen(false)}>
          <Icon name="close" />
        </button>
        <nav>
          {navigation.map((item) => {
            const active = pathname.startsWith(item.href);
            return (
              <Link key={item.href} href={item.href} className={active ? "active" : ""} aria-current={active ? "page" : undefined} onClick={() => setMobileOpen(false)}>
                <Icon name={item.icon} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
        <Link href="/admin" className={pathname.startsWith("/admin") ? "active nav-admin" : "nav-admin"} onClick={() => setMobileOpen(false)}>
          <Icon name="settings" />
          <span>Admin</span>
        </Link>
      </aside>
      {mobileOpen && <button className="nav-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
      <main className="atlas-main">{children}</main>
    </div>
  );
}
