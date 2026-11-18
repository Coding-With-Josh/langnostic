"use client";

import { Fragment, useEffect, useState, type ComponentType } from "react";
import Link from "next/link";
import { AnimatePresence, motion } from "motion/react";
import { Code, CodeXmlIcon, Home } from "lucide-react";
// Phase 3 (import path): the pasted bare path was malformed; use the alias.
import { MenuToggle } from "@/components/effects/menu-toggle";

// Phase 1 (link boundary): X and GitHub marks are local inline SVGs, so no
// remote asset or untrusted URL ever reaches an anchor or an <img> src.
const XLogo = ({ className }: { className?: string }) => (
  <svg
    viewBox="0 0 24 24"
    fill="currentColor"
    className={className}
    aria-hidden="true"
  >
    <path d="M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83L0 1.154h7.594l5.243 6.932ZM17.61 20.644h2.039L6.486 3.24H4.298Z" />
  </svg>
);

const GithubLogo = ({ className }: { className?: string }) => (
  <svg
    viewBox="0 0 24 24"
    fill="currentColor"
    className={className}
    aria-hidden="true"
  >
    <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
  </svg>
);

type MenuItem = {
  label: string;
  href: string;
  Icon: ComponentType<{ className?: string }>;
  external?: boolean;
  dividerAfter?: boolean;
};

// Phase 3 (dead-route integrity): only destinations that actually exist.
// Every href is a module constant; no prop or user input can substitute one.
const ITEMS: MenuItem[] = [
  { label: "home", href: "/", Icon: Home, dividerAfter: true },
  {
    label: "check me out X",
    href: "https://x.com/josh_scriptz",
    Icon: XLogo,
    external: true,
  },
  {
    label: "my github",
    href: "https://github.com/Coding-With-Josh",
    Icon: GithubLogo,
    external: true,
  },
  {
    label: "source code",
    href: "https://github.com/Coding-With-Josh/langnostic",
    Icon: CodeXmlIcon,
    external: true,
  },
];

export default function Menu({ className }: { className?: string }) {
  const [open, setOpen] = useState(false);

  // Phase 3 (overlay interception): listener only exists while open, and the
  // only possible state change from Escape is a close.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const itemClass =
    "flex items-center gap-3 px-2 py-1 rounded-xl text-sm font-medium tracking-tight text-neutral-700 hover:bg-neutral-100 hover:text-neutral-900 transition-all duration-200 active:scale-98";

  return (
    // Phase 3 (overlay interception): backdrop and panel are mounted only in
    // the open branch; unmounting removes every click-target immediately.
    <div className={`group fixed z-2000 bottom-4 right-4 ${className}`}>
      <AnimatePresence>
        {open && (
          <>
            <motion.div
              className="fixed inset-0 z-2000"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              onClick={() => setOpen(false)}
            />
            <motion.nav
              role="menu"
              aria-label="Site menu"
              className="absolute bottom-20 right-0 z-2001 w-48 rounded-2xl bg-white border border-neutral-200 shadow-xl p-1.5 flex flex-col gap-0.5"
              initial={{ opacity: 0, scale: 0.95, y: 6 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 6 }}
              transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
            >
              {ITEMS.map(({ label, href, Icon, external, dividerAfter }) => {
                const icon = <Icon className="w-4 h-4 shrink-0" />;
                const item = external ? (
                  // Phase 3 (reverse tabnabbing): explicit noopener + noreferrer
                  // on every external destination, not implicit modern behavior.
                  <a
                    key={label}
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    role="menuitem"
                    className={itemClass}
                    onClick={() => setOpen(false)}
                  >
                    {icon}
                    {label}
                  </a>
                ) : (
                  <Link
                    key={label}
                    href={href}
                    role="menuitem"
                    className={itemClass}
                    onClick={() => setOpen(false)}
                  >
                    {icon}
                    {label}
                  </Link>
                );
                return (
                  <Fragment key={label}>
                    {item}
                    {dividerAfter && (
                      <div className="border-t border-neutral-100 mx-1 my-1" />
                    )}
                  </Fragment>
                );
              })}
            </motion.nav>
          </>
        )}
      </AnimatePresence>
      <MenuToggle
        strokeWidth={3}
        open={open}
        onOpenChange={setOpen}
        className="size-10 lg:size-16 text-neutral-400 group-hover:text-black"
      />
    </div>
  );
}
