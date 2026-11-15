"use client";

import NumberFlow from "@number-flow/react";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";

/** Length of the count itself. */
const COUNT_MS = 2000;
/** Beat at 100% before it leaves — the page arrives during this window. */
const HOLD_MS = 600;
/** Must match the `duration-` class on the wrapper below. */
const EXIT_MS = 500;

// A store that never changes: it exists only to ask React whether this is the
// client render. On the server the snapshot is false, so a no-JS visitor never
// sees a frozen "0%" in the corner and there is no markup to mismatch on
// hydration.
const subscribe = () => () => {};
const getClientSnapshot = () => true;
const getServerSnapshot = () => false;

type Phase = "counting" | "holding" | "exiting" | "done";

export function LoadingCounter() {
  const mounted = useSyncExternalStore(
    subscribe,
    getClientSnapshot,
    getServerSnapshot,
  );
  const [percent, setPercent] = useState(0);
  const [phase, setPhase] = useState<Phase>("counting");
  const finished = useRef(false);

  useEffect(() => {
    if (!mounted) return;
    // Terminal-state guard: a media-query flip after the exit has been
    // scheduled must not re-arm the timers and make the corner reappear.
    if (finished.current) return;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const timers: ReturnType<typeof setTimeout>[] = [];

    const scheduleExit = () => {
      finished.current = true;
      // The page-load gate. This is the only place content becomes visible,
      // and it happens at 100% rather than at unmount so the page is already
      // there to receive the loader's departure. Content is NOT unmounted or
      // hidden by JS: `globals.css` keeps it in the DOM and hidden from paint
      // until this attribute exists, which also means a failure here falls
      // through to the stylesheet's own 4s fallback instead of a blank page.
      document.documentElement.dataset.revealed = "true";
      setPhase("holding");
      timers.push(
        setTimeout(() => {
          setPhase("exiting");
          timers.push(setTimeout(() => setPhase("done"), EXIT_MS));
        }, HOLD_MS),
      );
    };

    if (reduced) {
      // The count is the motion; under reduced motion show the end state and
      // leave without animating digits or the lift. Deferred to a frame
      // callback so this effect never calls setState in its own body.
      const raf = requestAnimationFrame(() => {
        setPercent(100);
        scheduleExit();
      });
      return () => {
        cancelAnimationFrame(raf);
        timers.forEach(clearTimeout);
      };
    }

    let raf = 0;
    const start = performance.now();
    let last = -1;

    const tick = (now: number) => {
      // Clamped before rounding: a late or throttled frame must not render
      // 170% or a float tail at the terminal tick.
      const t = Math.min((now - start) / COUNT_MS, 1);
      const next = Math.round(t * 100);
      if (next !== last) {
        last = next;
        setPercent(next);
      }
      if (t < 1) {
        raf = requestAnimationFrame(tick);
        return;
      }
      scheduleExit();
    };

    raf = requestAnimationFrame(tick);

    // One effect owns the loop and every timer, so StrictMode's double mount
    // and a client-side unmount can never leave a second counter running.
    return () => {
      cancelAnimationFrame(raf);
      timers.forEach(clearTimeout);
    };
  }, [mounted]);

  if (!mounted || phase === "done") return null;

  return (
    <div
      data-loading-counter
      // Decorative: it is not reporting real work, so it is hidden from
      // assistive tech rather than announced as a progress value.
      aria-hidden="true"
      className={`pointer-events-none select-none transition-transform transition-opacity duration-500 ease-out motion-reduce:transition-none ${
        phase === "exiting" ? "opacity-0" : "opacity-100"
      }`}
      style={{
        transform:
          phase === "exiting" ? "translateY(-0.75em)" : "translateY(0)",
      }}
    >
      <NumberFlow
        // Percent format takes 0–1, hence the division. Seeded at 0 on mount,
        // so the first value is not animated — every later update is.
        value={percent / 100}
        format={{ style: "percent", maximumFractionDigits: 0 }}
        // Upward trend, so digits travel the same direction as the exit.
        trend={1}
        // 101 updates across 2s: this is the "frequently updating" case the
        // library's will-change hint exists for.
        willChange
        className="tabular-nums [font:inherit] [--number-flow-mask-height:0.12em] [--number-flow-mask-width:0.18em]"
      />
    </div>
  );
}
