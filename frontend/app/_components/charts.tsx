"use client";

import React, { useMemo, useRef, useState } from "react";
import type { ModelMetadata, SparsityPoint, TreeTraceNode } from "./types";
import { fmt } from "./term";

/* ---------------------------------------------------------- matrix view */

/** Small matrices print as a character grid (+ − ·); large ones as a dot plot. */
export function MatrixView({ model }: { model: ModelMetadata }) {
  const [hover, setHover] = useState<SparsityPoint | null>(null);
  const n = Math.max(model.num_variables, 1);
  const m = Math.max(model.num_constraints, 1);
  const truncated = model.sparsity_sample.length < model.num_nonzeros;
  const asText = n <= 72 && m <= 60 && !truncated;

  const cells = useMemo(() => {
    const map = new Map<number, number>();
    for (const p of model.sparsity_sample) map.set(p.row * n + p.col, p.val);
    return map;
  }, [model, n]);

  const cell = 10;
  const W = n * cell;
  const H = m * cell;
  const r = Math.min(cell * 0.34, (4 * W) / 900);

  return (
    <figure>
      {asText ? (
        <div className="overflow-x-auto pb-1" onMouseLeave={() => setHover(null)}>
          <div className="inline-block text-[13px] leading-[1.25]">
            {Array.from({ length: m }, (_, row) => (
              <div key={row} className="animate-type flex whitespace-pre" style={{ animationDelay: `${Math.min(row * 22, 900)}ms` }}>
                <span className={`w-[5ch] shrink-0 pr-2 text-right ${hover?.row === row ? "text-amber" : "text-faint"}`}>{row}</span>
                {Array.from({ length: n }, (_, col) => {
                  const v = cells.get(row * n + col);
                  const lit = hover !== null && (hover.row === row || hover.col === col);
                  return (
                    <span
                      key={col}
                      onMouseEnter={() => setHover({ row, col, val: v ?? 0 })}
                      className={`inline-block w-[2ch] cursor-crosshair text-center ${lit ? "bg-line-soft" : ""} ${
                        v === undefined ? "text-faint/60" : v >= 0 ? "text-fg" : "text-red"
                      }`}
                    >
                      {v === undefined ? "·" : v >= 0 ? "+" : "−"}
                    </span>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="border border-line-soft p-3">
          <svg
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="xMidYMid meet"
            className="block h-auto max-h-[440px] w-full"
            onMouseLeave={() => setHover(null)}
            role="img"
            aria-label={`Sparsity pattern of the ${m} by ${n} constraint matrix`}
          >
            {hover ? (
              <g>
                <rect x={0} y={hover.row * cell} width={W} height={cell} className="fill-line-soft" />
                <rect x={hover.col * cell} y={0} width={cell} height={H} className="fill-line-soft" />
              </g>
            ) : null}
            {model.sparsity_sample.map((p, i) => (
              <circle
                key={`${model.name}-${i}`}
                cx={(p.col + 0.5) * cell}
                cy={(p.row + 0.5) * cell}
                r={r}
                className={`animate-fade cursor-crosshair ${p.val >= 0 ? "fill-fg" : "fill-red"}`}
                style={{ animationDelay: `${Math.min(i * 3, 900)}ms` }}
                onMouseEnter={() => setHover(p)}
              />
            ))}
          </svg>
        </div>
      )}
      <figcaption className="mt-3 flex flex-wrap justify-between gap-x-6 gap-y-1 text-[12.5px] text-dim">
        <span className="flex flex-wrap gap-x-4">
          <span>
            <span className="text-fg">+</span> positive
          </span>
          <span>
            <span className="text-red">−</span> negative
          </span>
          {asText ? (
            <span>
              <span className="text-faint">·</span> zero
            </span>
          ) : null}
          {truncated ? <span>first {model.sparsity_sample.length} of {fmt.int(model.num_nonzeros)} entries</span> : null}
        </span>
        <span className="text-fg">
          {hover ? `a[${hover.row}, ${hover.col}] = ${fmt.compact(hover.val)}` : "hover a cell to read its coefficient"}
        </span>
      </figcaption>
    </figure>
  );
}

/* ----------------------------------------------------------- line chart */

export interface ChartSeries {
  name: string;
  points: Array<{ x: number; y: number }>;
  stroke: string; // tailwind stroke-* class
  text: string; // tailwind text-* class for the legend
}

export function LineChart({
  series,
  height = 230,
  log = false,
  ariaLabel,
}: {
  series: ChartSeries[];
  height?: number;
  log?: boolean;
  ariaLabel: string;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const prepared = useMemo(
    () =>
      series
        .map((s) => ({
          ...s,
          pts: s.points
            .filter((p) => Number.isFinite(p.y))
            .map((p) => ({ x: p.x, raw: p.y, y: log ? Math.log10(Math.max(Math.abs(p.y), 1e-16)) : p.y })),
        }))
        .filter((s) => s.pts.length > 1),
    [series, log],
  );

  if (prepared.length === 0) return null;

  const W = 640;
  const H = height;
  const padL = 78;
  const padR = 12;
  const padT = 10;
  const padB = 26;

  const xs = prepared.flatMap((s) => s.pts.map((p) => p.x));
  const ys = prepared.flatMap((s) => s.pts.map((p) => p.y));
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs) === x0 ? x0 + 1 : Math.max(...xs);
  let y0 = Math.min(...ys);
  let y1 = Math.max(...ys);
  if (y0 === y1) {
    const d = Math.abs(y0) * 0.05 || 1;
    y0 -= d;
    y1 += d;
  } else if (!log) {
    const pad = (y1 - y0) * 0.08;
    y0 -= pad;
    y1 += pad;
  }
  if (log) {
    y0 = Math.floor(y0);
    y1 = Math.max(Math.ceil(y1), y0 + 1);
  }
  const sx = (x: number) => padL + ((x - x0) / (x1 - x0)) * (W - padL - padR);
  const sy = (y: number) => padT + (1 - (y - y0) / (y1 - y0)) * (H - padT - padB);

  const yTicks = log
    ? Array.from({ length: Math.floor((y1 - y0) / Math.max(1, Math.ceil((y1 - y0) / 4))) + 1 }, (_, i) => y0 + i * Math.max(1, Math.ceil((y1 - y0) / 4)))
    : [0, 1, 2, 3].map((i) => y0 + ((y1 - y0) * i) / 3);
  const xTicks = Array.from(new Set([0, 1, 2, 3, 4].map((i) => Math.round(x0 + ((x1 - x0) * i) / 4))));

  const base = prepared[0].pts;
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const xv = x0 + ((((e.clientX - rect.left) / rect.width) * W - padL) / (W - padL - padR)) * (x1 - x0);
    let best = 0;
    base.forEach((p, i) => {
      if (Math.abs(p.x - xv) < Math.abs(base[best].x - xv)) best = i;
    });
    setHoverIdx(best);
  };
  const hovered = hoverIdx !== null ? base[hoverIdx] : null;

  return (
    <figure>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="block h-auto w-full touch-none"
        onPointerMove={onMove}
        onPointerLeave={() => setHoverIdx(null)}
        role="img"
        aria-label={ariaLabel}
      >
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={padL} x2={W - padR} y1={sy(t)} y2={sy(t)} className="stroke-line-soft" strokeWidth={1} />
            <text x={padL - 10} y={sy(t)} dy="0.32em" textAnchor="end" fontSize={11} className="fill-dim">
              {log ? `1e${t}` : fmt.compact(t)}
            </text>
          </g>
        ))}
        <line x1={padL} x2={padL} y1={padT} y2={H - padB} className="stroke-line" strokeWidth={1} />
        {xTicks.map((t) => (
          <text key={`x${t}`} x={sx(t)} y={H - 7} textAnchor="middle" fontSize={11} className="fill-dim">
            {t}
          </text>
        ))}
        {prepared.map((s, si) => (
          <path
            key={s.name}
            d={s.pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("")}
            fill="none"
            pathLength={1}
            strokeWidth={1.6}
            strokeLinejoin="round"
            className={`${s.stroke} animate-draw`}
            style={{ strokeDasharray: 1, animationDelay: `${si * 150}ms` }}
          />
        ))}
        {hovered ? (
          <g>
            <line x1={sx(hovered.x)} x2={sx(hovered.x)} y1={padT} y2={H - padB} className="stroke-faint" strokeDasharray="2 3" />
            {prepared.map((s) => {
              const p = s.pts[Math.min(hoverIdx ?? 0, s.pts.length - 1)];
              return <rect key={`h-${s.name}`} x={sx(p.x) - 3} y={sy(p.y) - 3} width={6} height={6} className={s.stroke.replace("stroke-", "fill-")} />;
            })}
          </g>
        ) : null}
      </svg>
      <figcaption className="mt-2 flex flex-wrap justify-between gap-x-6 gap-y-1 text-[12.5px] text-dim">
        <span className="flex flex-wrap gap-x-4">
          {prepared.map((s) => (
            <span key={s.name}>
              <span className={s.text}>──</span> {s.name}
            </span>
          ))}
        </span>
        <span className="text-fg">
          {hovered
            ? `iter ${hovered.x}  ` + prepared.map((s) => `${s.name} ${fmt.compact(s.pts[Math.min(hoverIdx ?? 0, s.pts.length - 1)].raw)}`).join("  ")
            : "hover to read values"}
        </span>
      </figcaption>
    </figure>
  );
}

/* ------------------------------------------------------------ tree view */

const TREE_WORDS: Record<string, { text: string; cls: string }> = {
  ROOT: { text: "root", cls: "text-amber" },
  BRANCH: { text: "branched", cls: "text-dim" },
  PRUNED_BOUND: { text: "pruned by bound", cls: "text-faint" },
  INTEGER_INCUMBENT: { text: "integer solution ✓", cls: "text-green" },
  INFEASIBLE: { text: "infeasible", cls: "text-red" },
};

export function treeWord(status: string) {
  return TREE_WORDS[status] ?? { text: status.toLowerCase(), cls: "text-dim" };
}

function flattenTree(nodes: TreeTraceNode[]) {
  const byId = new Map<number, TreeTraceNode>();
  for (const node of nodes) byId.set(node.node_id, { ...(byId.get(node.node_id) ?? {}), ...node });
  const kids = new Map<number, number[]>();
  const roots: number[] = [];
  for (const node of byId.values()) {
    if (node.parent_id !== null && node.parent_id !== node.node_id && byId.has(node.parent_id)) {
      kids.set(node.parent_id, [...(kids.get(node.parent_id) ?? []), node.node_id]);
    } else {
      roots.push(node.node_id);
    }
  }
  const lines: Array<{ node: TreeTraceNode; prefix: string; isRoot: boolean }> = [];
  const seen = new Set<number>();
  const walk = (id: number, prefix: string, last: boolean, isRoot: boolean) => {
    const node = byId.get(id);
    if (!node || seen.has(id)) return;
    seen.add(id);
    lines.push({ node, prefix: isRoot ? "" : prefix + (last ? "└─ " : "├─ "), isRoot });
    const children = kids.get(id) ?? [];
    const next = isRoot ? "" : prefix + (last ? "   " : "│  ");
    children.forEach((k, i) => walk(k, next, i === children.length - 1, false));
  };
  roots.forEach((r) => walk(r, "", true, true));
  return lines;
}

/** Branch-and-bound search printed like the `tree` command. */
export function TreeView({
  nodes,
  selected,
  onSelect,
}: {
  nodes: TreeTraceNode[];
  selected: number | null;
  onSelect: (id: number) => void;
}) {
  const lines = useMemo(() => flattenTree(nodes), [nodes]);
  return (
    <div className="overflow-x-auto text-[13.5px] leading-[1.75]">
      {lines.map(({ node, prefix, isRoot }, i) => {
        const word = treeWord(node.status);
        const isSel = selected === node.node_id;
        return (
          <button
            key={`${node.node_id}-${i}`}
            onClick={() => onSelect(node.node_id)}
            className={`animate-type flex w-full min-w-max items-baseline whitespace-pre pr-2 text-left transition-colors ${
              isSel ? "bg-line-soft" : "hover:bg-line-soft/60"
            }`}
            style={{ animationDelay: `${Math.min(i * 40, 1500)}ms` }}
          >
            <span className="text-faint">{prefix}</span>
            <span className="text-fg">{isRoot ? "root relaxation" : node.branch_condition}</span>
            <span className={`ml-3 ${word.cls}`}>{isRoot && node.status === "ROOT" ? "" : word.text}</span>
            <span className="ml-auto pl-8 text-dim">z = {fmt.compact(node.lower_bound)}</span>
          </button>
        );
      })}
    </div>
  );
}
