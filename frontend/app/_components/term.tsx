"use client";

import React, { useEffect, useRef, useState, useSyncExternalStore } from "react";

/* ------------------------------------------------------------------ motion */

const REDUCED_QUERY = "(prefers-reduced-motion: reduce)";

function subscribeReduced(callback: () => void) {
  const mq = window.matchMedia(REDUCED_QUERY);
  mq.addEventListener("change", callback);
  return () => mq.removeEventListener("change", callback);
}

export function useReducedMotion() {
  return useSyncExternalStore(
    subscribeReduced,
    () => window.matchMedia(REDUCED_QUERY).matches,
    () => false,
  );
}

/** Eased count toward `target`, continuing from the last value shown. */
export function useCountUp(target: number | null | undefined, duration = 700): number | null {
  const reduced = useReducedMotion();
  const [value, setValue] = useState(0);
  const last = useRef(0);
  const valid = typeof target === "number" && Number.isFinite(target);

  useEffect(() => {
    if (!valid || reduced) return;
    const to = target as number;
    const from = last.current;
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      last.current = from + (to - from) * eased;
      setValue(last.current);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, valid, reduced, duration]);

  if (!valid) return null;
  return reduced ? (target as number) : value;
}

/** Reveals `total` items one at a time, like output arriving in a terminal. */
export function useReveal(total: number, stepMs = 90) {
  const reduced = useReducedMotion();
  const [shown, setShown] = useState(0);
  useEffect(() => {
    if (reduced || shown >= total) return;
    const id = window.setTimeout(() => setShown((s) => s + 1), shown === 0 ? 80 : stepMs);
    return () => window.clearTimeout(id);
  }, [shown, total, stepMs, reduced]);
  return reduced ? total : Math.min(shown, total);
}

export function useInView<T extends Element>() {
  const ref = useRef<T | null>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || inView) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          io.disconnect();
        }
      },
      { threshold: 0.1 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [inView]);
  return [ref, inView] as const;
}

/* -------------------------------------------------------------- formatting */

export const fmt = {
  num(v: number | null | undefined, digits = 2) {
    if (v === null || v === undefined || !Number.isFinite(v)) return "-";
    return v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  },
  int(v: number | null | undefined) {
    if (v === null || v === undefined || !Number.isFinite(v)) return "-";
    return Math.round(v).toLocaleString("en-US");
  },
  sci(v: number | null | undefined, digits = 1) {
    if (v === null || v === undefined || !Number.isFinite(v)) return "-";
    return v === 0 ? "0" : v.toExponential(digits);
  },
  dur(seconds: number | null | undefined) {
    if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "-";
    return seconds < 1 ? `${(seconds * 1000).toFixed(1)} ms` : `${seconds.toFixed(2)} s`;
  },
  pct(v: number | null | undefined, digits = 1) {
    if (v === null || v === undefined || !Number.isFinite(v)) return "-";
    return `${v.toFixed(digits)}%`;
  },
  compact(v: number | null | undefined) {
    if (v === null || v === undefined || !Number.isFinite(v)) return "-";
    const a = Math.abs(v);
    if (a >= 1e4) return fmt.num(v, 0);
    if (a >= 1 || a === 0) return fmt.num(v, 3);
    if (a >= 1e-3) return v.toFixed(4);
    return fmt.sci(v, 2);
  },
  words(key: string) {
    return key.replace(/_/g, " ").toLowerCase();
  },
};

/* ------------------------------------------------------------------- tone */

export type Tone = "ok" | "warn" | "bad" | "dim";

export const toneText: Record<Tone, string> = {
  ok: "text-green",
  warn: "text-amber",
  bad: "text-red",
  dim: "text-dim",
};

/* -------------------------------------------------------------- primitives */

/** Box with its title set into the top border, like a rich panel in the CLI. */
export function Panel({
  title,
  right,
  children,
  className = "",
}: {
  title?: React.ReactNode;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`relative min-w-0 border border-line px-4 pb-4 pt-5 sm:px-5 ${className}`}>
      {title ? (
        <h2 className="absolute -top-[10px] left-3 bg-bg px-1.5 text-[12.5px] leading-5 text-dim">{title}</h2>
      ) : null}
      {right ? <div className="absolute -top-[10px] right-3 bg-bg px-1.5 text-[12.5px] leading-5 text-dim">{right}</div> : null}
      {children}
    </section>
  );
}

export function Comment({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <p className={`text-[13px] leading-relaxed text-dim ${className}`}>
      <span className="select-none text-faint"># </span>
      {children}
    </p>
  );
}

/** key .......... value */
export function Leader({ label, children, hint }: { label: React.ReactNode; children: React.ReactNode; hint?: React.ReactNode }) {
  return (
    <div className="flex min-w-0 items-baseline gap-2 py-[3px] text-[13.5px]">
      <span className="min-w-0 truncate text-dim">
        {label}
        {hint ? <span className="text-faint"> {hint}</span> : null}
      </span>
      <span aria-hidden className="min-w-4 flex-1 -translate-y-[4px] border-b border-dotted border-faint/70" />
      <span className="shrink-0 text-right text-fg">{children}</span>
    </div>
  );
}

type BtnVariant = "primary" | "bracket";

export function Btn({
  variant = "bracket",
  busy = false,
  className = "",
  children,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: BtnVariant; busy?: boolean }) {
  const base =
    "group inline-flex items-center gap-1.5 whitespace-nowrap text-[13.5px] transition-colors duration-100 disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-amber";
  if (variant === "primary") {
    return (
      <button
        className={`${base} h-8 bg-amber px-3.5 font-medium text-bg hover:bg-amber-hi active:translate-y-px ${className}`}
        disabled={busy || props.disabled}
        {...props}
      >
        {busy ? <Spinner className="text-bg" /> : null}
        {children}
      </button>
    );
  }
  return (
    <button className={`${base} px-0.5 text-fg hover:bg-fg hover:text-bg ${className}`} disabled={busy || props.disabled} {...props}>
      <span className="text-faint group-hover:text-bg">[</span>
      {busy ? <Spinner /> : null}
      {children}
      <span className="text-faint group-hover:text-bg">]</span>
    </button>
  );
}

/** [x] label */
export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (next: boolean) => void; label: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="group whitespace-pre text-[13.5px] text-dim hover:text-fg focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-amber"
    >
      <span className="text-faint">[</span>
      <span className="text-amber">{checked ? "x" : " "}</span>
      <span className="text-faint">]</span> {label}
    </button>
  );
}

export function Select({
  value,
  onChange,
  options,
  label,
}: {
  value: string;
  onChange: (next: string) => void;
  options: Array<{ value: string; label: string }>;
  label: string;
}) {
  return (
    <label className="relative inline-flex items-center">
      <span className="sr-only">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-8 appearance-none border border-line bg-bg pl-2.5 pr-7 text-[13.5px] text-fg transition-colors hover:border-faint focus:border-amber focus:outline-none"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      <span aria-hidden className="pointer-events-none absolute right-2.5 text-[11px] text-dim">
        v
      </span>
    </label>
  );
}

export function Kbd({ children }: { children: React.ReactNode }) {
  return <kbd className="border border-line px-1 py-px text-[11.5px] leading-none text-dim">{children}</kbd>;
}

const FRAMES = ["|", "/", "-", "\\"];

export function Spinner({ className = "" }: { className?: string }) {
  const reduced = useReducedMotion();
  const [i, setI] = useState(0);
  useEffect(() => {
    if (reduced) return;
    const id = window.setInterval(() => setI((x) => (x + 1) % FRAMES.length), 90);
    return () => window.clearInterval(id);
  }, [reduced]);
  return (
    <span aria-hidden className={`inline-block w-[1ch] text-center text-amber ${className}`}>
      {reduced ? "*" : FRAMES[i]}
    </span>
  );
}

export function CountUp({
  value,
  format,
  duration,
}: {
  value: number | null | undefined;
  format: (v: number | null) => string;
  duration?: number;
}) {
  const shown = useCountUp(value, duration);
  return <span>{format(shown)}</span>;
}

/** [██████····] that fills in when scrolled into view. */
export function AsciiBar({ value, width = 24, tone = "warn" }: { value: number; width?: number; tone?: Tone }) {
  const [ref, inView] = useInView<HTMLSpanElement>();
  const target = inView ? Math.round(Math.max(0, Math.min(1, value)) * width) : 0;
  const filled = Math.round(useCountUp(target, 650) ?? 0);
  return (
    <span ref={ref} aria-hidden className="whitespace-pre">
      <span className="text-faint">[</span>
      <span className={toneText[tone]}>{"█".repeat(filled)}</span>
      <span className="text-faint/70">{"·".repeat(Math.max(0, width - filled))}</span>
      <span className="text-faint">]</span>
    </span>
  );
}

export function ViewTabs<T extends number>({
  items,
  active,
  onChange,
}: {
  items: Array<{ id: T; label: string; done?: boolean }>;
  active: T;
  onChange: (id: T) => void;
}) {
  return (
    <div role="tablist" className="flex flex-wrap items-center gap-x-1 gap-y-1.5 border-b border-line pb-2.5">
      {items.map((item) => {
        const isActive = item.id === active;
        return (
          <button
            key={item.id}
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(item.id)}
            className={`whitespace-pre px-2 py-0.5 text-[13.5px] transition-colors duration-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-amber ${
              isActive ? "bg-fg text-bg" : "text-dim hover:bg-line-soft hover:text-fg"
            }`}
          >
            <span className={isActive ? "text-bg/60" : "text-faint"}>{item.id}</span> {item.label}
            <span className={isActive ? "text-bg" : "text-green"}>{item.done ? " ✓" : "  "}</span>
          </button>
        );
      })}
    </div>
  );
}
