import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

// Phase 3 (class-injection / conflicting utilities): merge duplicate Tailwind
// utilities so a caller-supplied size/color deterministically wins over the
// component defaults instead of both classes reaching the DOM.
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
