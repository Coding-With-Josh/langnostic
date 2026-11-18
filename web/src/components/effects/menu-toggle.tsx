"use client";

import React from "react";
// Phase 3 (import path): the pasted bare "/lib/utils" does not resolve here;
// use the project's "@/" alias instead.
import { cn } from "@/lib/utils";

type MenuToggleProps = React.ComponentProps<"svg"> & {
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

export function MenuToggle({
  open,
  onOpenChange,
  className,
  fill = "none",
  stroke = "currentColor",
  strokeWidth = 2,
  strokeLinecap = "round",
  strokeLinejoin = "round",
  ...props
}: MenuToggleProps) {
  return (
    <label className="flex cursor-pointer items-center justify-center">
      {/* Phase 3 (open-state desync): controlled by parent state so Escape and
          backdrop closes also uncheck the box; onChange can only flip boolean. */}
      <input
        className="hidden"
        type="checkbox"
        onChange={() => onOpenChange(!open)}
        checked={open}
        aria-label="Toggle menu"
      />
      <svg
        strokeWidth={strokeWidth}
        fill={fill}
        stroke={stroke}
        viewBox="0 0 32 32"
        strokeLinecap={strokeLinecap}
        strokeLinejoin={strokeLinejoin}
        className={cn(
          "size-4 transition-transform duration-600 ease-out",
          open && "-rotate-45",
          className,
        )}
        {...props}
      >
        <path
          className={cn(
            "transition-all duration-600 ease-out",
            open
              ? "[stroke-dasharray:20_300] [stroke-dashoffset:-32.42px]"
              : "[stroke-dasharray:12_63]",
          )}
          d="M27 10 13 10C10.8 10 9 8.2 9 6 9 3.5 10.8 2 13 2 15.2 2 17 3.8 17 6L17 26C17 28.2 18.8 30 21 30 23.2 30 25 28.2 25 26 25 23.8 23.2 22 21 22L7 22"
        />
        <path d="M7 16 27 16" />
      </svg>
    </label>
  );
}
