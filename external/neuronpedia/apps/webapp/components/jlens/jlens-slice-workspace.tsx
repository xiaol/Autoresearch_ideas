'use client';

import { LensMode, LensTokenMessage, LensType, LensTypeSlice } from '@/lib/utils/lens';
import { Pin } from 'lucide-react';
import { CSSProperties, useEffect, useMemo, useState } from 'react';
import { COLOR_RGB, COLOR_RGB_DARK, MAX_SELECT, SELECT_COLORS, SelectColor } from './jlens-analysis';
import { lensTypesForMode } from './jlens-lens-mode';
import { displayToken, normKey } from './jlens-token-popup';
import { JlensAnalysis } from './use-jlens-analysis';

type ReadoutItem = {
  type: LensType;
  token: string;
  tokenId?: number;
  prob?: number;
  rank?: number;
  slot: number;
};

type CellCoord = {
  position: number;
  layer: number;
};

type PinToken = {
  key: string;
  type: LensType;
  color: SelectColor;
  label: string;
};

const GRID_CELL_W = 78;
const GRID_CELL_H = 28;

function sliceFor(token: LensTokenMessage | undefined, type: LensType): LensTypeSlice | undefined {
  return token?.results.find((result) => result.type === type);
}

function rowIndexForLayer(layersByType: Record<string, number[]>, type: LensType, layer: number): number {
  return (layersByType[type] ?? []).indexOf(layer);
}

function rowItems(
  slice: LensTypeSlice | undefined,
  type: LensType,
  rowIndex: number,
  limit = Number.POSITIVE_INFINITY,
): ReadoutItem[] {
  if (!slice || rowIndex < 0) {
    return [];
  }
  const tokens = slice.top_tokens[rowIndex] ?? [];
  const probs = slice.top_probs[rowIndex] ?? [];
  const ids = slice.top_token_ids?.[rowIndex] ?? [];
  const ranks = slice.top_ranks?.[rowIndex] ?? [];
  return tokens.slice(0, limit).map((token, slot) => ({
    type,
    token,
    tokenId: ids[slot],
    prob: probs[slot],
    rank: ranks[slot],
    slot,
  }));
}

function topItemAt(
  token: LensTokenMessage | undefined,
  layersByType: Record<string, number[]>,
  type: LensType,
  layer: number,
): ReadoutItem | null {
  return rowItems(sliceFor(token, type), type, rowIndexForLayer(layersByType, type, layer), 1)[0] ?? null;
}

function rankLabel(item: ReadoutItem): string {
  if (typeof item.rank === 'number') {
    return `#${item.rank + 1}`;
  }
  return `top-${item.slot + 1}`;
}

function probLabel(prob: number | undefined): string {
  if (typeof prob !== 'number' || !Number.isFinite(prob)) {
    return '';
  }
  if (prob >= 0.1) {
    return `${Math.round(prob * 100)}%`;
  }
  if (prob >= 0.001) {
    return `${(prob * 100).toFixed(1)}%`;
  }
  return '<0.1%';
}

function itemMatchesPin(item: ReadoutItem, pin: PinToken): boolean {
  return item.type === pin.type && normKey(item.token) === pin.key;
}

function rankAt(
  tokens: LensTokenMessage[],
  layersByType: Record<string, number[]>,
  position: number,
  layer: number,
  pin: PinToken,
): { rank: number; source: 'full' | 'slot' } | null {
  const token = tokens.find((t) => t.position === position);
  const slice = sliceFor(token, pin.type);
  const rowIndex = rowIndexForLayer(layersByType, pin.type, layer);
  const items = rowItems(slice, pin.type, rowIndex);
  const item = items.find((candidate) => itemMatchesPin(candidate, pin));
  if (!item) {
    return null;
  }
  if (typeof item.rank === 'number') {
    return { rank: item.rank, source: 'full' };
  }
  return { rank: item.slot, source: 'slot' };
}

function tokenLabelForKey(tokens: LensTokenMessage[], type: LensType, key: string): string {
  for (const token of tokens) {
    const slice = sliceFor(token, type);
    if (!slice) {
      continue;
    }
    for (const row of slice.top_tokens) {
      const found = row.find((candidate) => normKey(candidate) === key);
      if (found != null) {
        return found;
      }
    }
  }
  return key;
}

function clampSelection<T>(items: T[], current: T | null, fallback: T | null): T | null {
  if (current != null && items.includes(current)) {
    return current;
  }
  return fallback;
}

function rankColor(rank: number, maxRank: number): string {
  const denom = Math.log(Math.max(2, maxRank + 1));
  const score = 1 - Math.log(rank + 1) / denom;
  const hue = 178 + score * 34;
  const lightness = 86 - score * 47;
  return `hsl(${hue}, 70%, ${lightness}%)`;
}

function pinTextColor(color: SelectColor): string {
  return COLOR_RGB_DARK[color] ?? COLOR_RGB_DARK.sky;
}

function promptTokenClass(selected: boolean, generated: boolean): string {
  if (selected) {
    return 'border-sky-500 bg-sky-50 text-sky-800';
  }
  if (generated) {
    return 'border-emerald-200 bg-emerald-50 text-emerald-800 hover:border-emerald-300';
  }
  return 'border-slate-200 bg-white text-slate-700 hover:border-sky-300';
}

function ReadoutTokenButton({
  item,
  pinned,
  disabled,
  onToggle,
}: {
  item: ReadoutItem;
  pinned: boolean;
  disabled: boolean;
  onToggle: (item: ReadoutItem) => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => onToggle(item)}
      title={`${pinned ? 'Unpin' : 'Pin'} ${item.token}`}
      className={`group flex min-w-0 items-center gap-x-1 rounded border px-1.5 py-0.5 text-left font-mono text-[10px] leading-tight transition ${
        pinned
          ? 'border-sky-300 bg-sky-50 text-sky-800'
          : 'border-slate-200 bg-white text-slate-700 hover:border-sky-300 hover:bg-sky-50'
      } disabled:cursor-not-allowed disabled:opacity-45`}
    >
      <span className="shrink-0 text-[9px] tabular-nums text-slate-400">{rankLabel(item)}</span>
      <span className="truncate">{displayToken(item.token)}</span>
      <span className="ml-auto shrink-0 text-[9px] text-slate-400">{probLabel(item.prob)}</span>
      <Pin className={`h-3 w-3 shrink-0 ${pinned ? 'text-sky-500' : 'text-slate-300 group-hover:text-sky-500'}`} />
    </button>
  );
}

function CellReadout({
  coord,
  tokens,
  layersByType,
  displayedTypes,
  selectedPins,
  pinDisabled,
  onTogglePin,
}: {
  coord: CellCoord | null;
  tokens: LensTokenMessage[];
  layersByType: Record<string, number[]>;
  displayedTypes: LensType[];
  selectedPins: PinToken[];
  pinDisabled: boolean;
  onTogglePin: (item: ReadoutItem) => void;
}) {
  if (!coord) {
    return (
      <div className="flex min-h-[118px] items-center justify-center rounded-md border border-dashed border-slate-200 bg-slate-50 text-[11px] text-slate-400">
        Hover a cell
      </div>
    );
  }
  const token = tokens.find((t) => t.position === coord.position);
  return (
    <div className="flex min-h-[118px] flex-col gap-y-2 rounded-md border border-slate-200 bg-white p-2">
      <div className="flex items-center justify-between gap-x-2 text-[10px] font-semibold uppercase text-slate-400">
        <span>Cell Readout</span>
        <span className="font-mono normal-case text-slate-500">
          pos {coord.position} · layer {coord.layer}
        </span>
      </div>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {displayedTypes.map((type) => {
          const rowIndex = rowIndexForLayer(layersByType, type, coord.layer);
          const items = rowItems(sliceFor(token, type), type, rowIndex, 5);
          return (
            <div key={type} className="flex min-w-0 flex-col gap-y-1">
              <div className="text-[10px] font-semibold text-slate-500">
                {type === LensType.JACOBIAN_LENS ? 'Jacobian' : 'Logit'}
              </div>
              {items.length > 0 ? (
                items.map((item) => {
                  const pinned = selectedPins.some((pin) => itemMatchesPin(item, pin));
                  return (
                    <ReadoutTokenButton
                      key={`${type}-${item.slot}-${item.token}`}
                      item={item}
                      pinned={pinned}
                      disabled={!pinned && pinDisabled}
                      onToggle={onTogglePin}
                    />
                  );
                })
              ) : (
                <div className="rounded border border-dashed border-slate-200 px-2 py-2 text-[10px] text-slate-400">
                  no data
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function RankSparkline({
  title,
  domain,
  ranks,
  color,
}: {
  title: string;
  domain: number[];
  ranks: (number | null)[];
  color: SelectColor;
}) {
  const width = 320;
  const height = 92;
  const pad = 18;
  const known = ranks.filter((r): r is number => r != null);
  const maxRank = Math.max(8, ...known);
  const x = (idx: number) => (domain.length <= 1 ? width / 2 : pad + (idx / (domain.length - 1)) * (width - pad * 2));
  const y = (rank: number) => {
    const denom = Math.log(Math.max(2, maxRank + 1));
    return pad + (Math.log(rank + 1) / denom) * (height - pad * 2);
  };
  const segments: string[][] = [];
  let current: string[] = [];
  ranks.forEach((rank, idx) => {
    if (rank == null) {
      if (current.length > 0) {
        segments.push(current);
        current = [];
      }
      return;
    }
    current.push(`${x(idx).toFixed(1)},${y(rank).toFixed(1)}`);
  });
  if (current.length > 0) {
    segments.push(current);
  }
  return (
    <div className="flex min-w-0 flex-col gap-y-1 rounded-md border border-slate-200 bg-white p-2">
      <div className="flex items-center justify-between gap-x-2 text-[10px] font-semibold uppercase text-slate-400">
        <span>{title}</span>
        <span className="normal-case text-slate-500">{known.length}/{domain.length} known</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-[92px] w-full overflow-visible">
        <line x1={pad} x2={width - pad} y1={pad} y2={pad} stroke="#e2e8f0" />
        <line x1={pad} x2={width - pad} y1={height - pad} y2={height - pad} stroke="#e2e8f0" />
        <text x={2} y={pad + 3} className="fill-slate-400 text-[8px]">
          #1
        </text>
        <text x={2} y={height - pad + 3} className="fill-slate-400 text-[8px]">
          deep
        </text>
        {segments.map((points, idx) => (
          <polyline
            key={idx}
            points={points.join(' ')}
            fill="none"
            stroke={`rgb(${COLOR_RGB[color]})`}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ))}
        {ranks.map((rank, idx) =>
          rank == null ? (
            <circle key={domain[idx]} cx={x(idx)} cy={height - pad} r="1.4" fill="#cbd5e1" />
          ) : (
            <circle key={domain[idx]} cx={x(idx)} cy={y(rank)} r="2.5" fill={`rgb(${COLOR_RGB[color]})`}>
              <title>
                {domain[idx]}: #{rank + 1}
              </title>
            </circle>
          ),
        )}
      </svg>
    </div>
  );
}

function RankHeatmap({
  tokens,
  layers,
  layersByType,
  pin,
  selectedCoord,
  onSelect,
}: {
  tokens: LensTokenMessage[];
  layers: number[];
  layersByType: Record<string, number[]>;
  pin: PinToken | null;
  selectedCoord: CellCoord;
  onSelect: (coord: CellCoord) => void;
}) {
  const knownRanks = useMemo(() => {
    if (!pin) {
      return [];
    }
    const ranks: number[] = [];
    for (const layer of layers) {
      for (const token of tokens) {
        const result = rankAt(tokens, layersByType, token.position, layer, pin);
        if (result) {
          ranks.push(result.rank);
        }
      }
    }
    return ranks;
  }, [layers, layersByType, pin, tokens]);
  const maxRank = Math.max(8, ...knownRanks);
  if (!pin) {
    return (
      <div className="flex min-h-[170px] items-center justify-center rounded-md border border-dashed border-slate-200 bg-slate-50 text-[11px] text-slate-400">
        Pin a readout token
      </div>
    );
  }
  return (
    <div className="flex min-h-[170px] flex-col gap-y-1 rounded-md border border-slate-200 bg-white p-2">
      <div className="flex items-center justify-between gap-x-2 text-[10px] font-semibold uppercase text-slate-400">
        <span>Rank Heatmap</span>
        <span className="min-w-0 truncate font-mono normal-case" style={{ color: `rgb(${pinTextColor(pin.color)})` }}>
          {displayToken(pin.label)}
        </span>
      </div>
      <div className="overflow-auto">
        <div
          className="grid min-w-max gap-px"
          style={{
            gridTemplateColumns: `46px repeat(${tokens.length}, 22px)`,
          }}
        >
          <div />
          {tokens.map((token) => (
            <button
              type="button"
              key={`rank-col-${token.position}`}
              onClick={() => onSelect({ position: token.position, layer: selectedCoord.layer })}
              className={`h-5 truncate rounded-sm border text-[8px] tabular-nums ${
                selectedCoord.position === token.position
                  ? 'border-sky-400 bg-sky-50 text-sky-700'
                  : 'border-transparent text-slate-400 hover:bg-slate-100'
              }`}
              title={`${token.position}: ${token.token}`}
            >
              {token.position}
            </button>
          ))}
          {layers.map((layer) => (
            <div key={`rank-layer-group-${layer}`} className="contents">
              <button
                type="button"
                key={`rank-row-${layer}`}
                onClick={() => onSelect({ position: selectedCoord.position, layer })}
                className={`h-5 rounded-sm border px-1 text-right text-[9px] tabular-nums ${
                  selectedCoord.layer === layer
                    ? 'border-sky-400 bg-sky-50 text-sky-700'
                    : 'border-transparent text-slate-400 hover:bg-slate-100'
                }`}
              >
                L{layer}
              </button>
              {tokens.map((token) => {
                const result = rankAt(tokens, layersByType, token.position, layer, pin);
                const selected = selectedCoord.position === token.position && selectedCoord.layer === layer;
                return (
                  <button
                    type="button"
                    key={`rank-${layer}-${token.position}`}
                    onClick={() => onSelect({ position: token.position, layer })}
                    className={`h-5 w-[22px] rounded-sm border ${
                      selected ? 'border-sky-500 ring-1 ring-sky-500' : 'border-white'
                    }`}
                    style={{ backgroundColor: result ? rankColor(result.rank, maxRank) : '#f1f5f9' }}
                    title={
                      result
                        ? `pos ${token.position}, layer ${layer}: rank #${result.rank + 1}${
                            result.source === 'slot' ? ' (returned slot)' : ''
                          }`
                        : `pos ${token.position}, layer ${layer}: not in returned top-k`
                    }
                  />
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function SliceGrid({
  tokens,
  layers,
  layersByType,
  lensMode,
  displayedTypes,
  selectedCoord,
  hoveredCoord,
  onHover,
  onSelect,
}: {
  tokens: LensTokenMessage[];
  layers: number[];
  layersByType: Record<string, number[]>;
  lensMode: LensMode;
  displayedTypes: LensType[];
  selectedCoord: CellCoord;
  hoveredCoord: CellCoord | null;
  onHover: (coord: CellCoord | null) => void;
  onSelect: (coord: CellCoord) => void;
}) {
  const diffMode = lensMode === LensMode.DIFF && displayedTypes.length > 1;
  return (
    <div className="overflow-auto rounded-md border border-slate-200 bg-white">
      <div
        className="grid min-w-max gap-px bg-slate-100 p-1"
        style={
          {
            gridTemplateColumns: `52px repeat(${tokens.length}, ${GRID_CELL_W}px)`,
          } as CSSProperties
        }
      >
        <div className="sticky left-1 z-10 h-6 bg-slate-100" />
        {tokens.map((token) => (
          <button
            type="button"
            key={`grid-head-${token.position}`}
            onClick={() => onSelect({ position: token.position, layer: selectedCoord.layer })}
            className={`h-6 truncate rounded border bg-white px-1 font-mono text-[9px] leading-none ${
              selectedCoord.position === token.position ? 'border-sky-400 text-sky-700' : 'border-slate-200 text-slate-500'
            }`}
            title={`${token.position}: ${token.token}`}
          >
            {displayToken(token.token)}
          </button>
        ))}
        {layers.map((layer) => (
          <div key={`grid-layer-group-${layer}`} className="contents">
            <button
              type="button"
              key={`grid-layer-${layer}`}
              onClick={() => onSelect({ position: selectedCoord.position, layer })}
              className={`sticky left-1 z-10 rounded border bg-white px-1 text-right text-[10px] font-semibold tabular-nums ${
                selectedCoord.layer === layer ? 'border-sky-400 text-sky-700' : 'border-slate-200 text-slate-500'
              }`}
              style={{ height: GRID_CELL_H }}
            >
              L{layer}
            </button>
            {tokens.map((token) => {
              const coord = { position: token.position, layer };
              const jTop = topItemAt(token, layersByType, LensType.JACOBIAN_LENS, layer);
              const logitTop = topItemAt(token, layersByType, LensType.LOGIT_LENS, layer);
              const singleType = displayedTypes[0];
              const singleTop = singleType ? topItemAt(token, layersByType, singleType, layer) : null;
              const selected = selectedCoord.position === token.position && selectedCoord.layer === layer;
              const hovered = hoveredCoord?.position === token.position && hoveredCoord.layer === layer;
              const divergent = diffMode && jTop && logitTop && jTop.token !== logitTop.token;
              return (
                <button
                  type="button"
                  key={`grid-${layer}-${token.position}`}
                  onMouseEnter={() => onHover(coord)}
                  onMouseLeave={() => onHover(null)}
                  onFocus={() => onHover(coord)}
                  onBlur={() => onHover(null)}
                  onClick={() => onSelect(coord)}
                  className={`flex min-w-0 flex-col justify-center rounded-sm border px-1 text-left font-mono leading-tight transition ${
                    selected
                      ? 'border-sky-500 bg-sky-50 ring-1 ring-sky-500'
                      : hovered
                        ? 'border-sky-300 bg-sky-50'
                        : divergent
                          ? 'border-amber-200 bg-amber-50'
                          : 'border-white bg-white hover:border-sky-200 hover:bg-sky-50'
                  }`}
                  style={{ width: GRID_CELL_W, height: GRID_CELL_H }}
                  title={`pos ${token.position}, layer ${layer}`}
                >
                  {diffMode ? (
                    <>
                      <span className="truncate text-[9px] text-sky-700">J {displayToken(jTop?.token ?? '')}</span>
                      <span className="truncate text-[9px] text-emerald-700">
                        L {displayToken(logitTop?.token ?? '')}
                      </span>
                    </>
                  ) : (
                    <span className="truncate text-[10px] text-slate-700">{displayToken(singleTop?.token ?? '')}</span>
                  )}
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

function LayerReadout({
  tokens,
  layers,
  layersByType,
  displayedTypes,
  position,
  activePin,
  onSelectLayer,
}: {
  tokens: LensTokenMessage[];
  layers: number[];
  layersByType: Record<string, number[]>;
  displayedTypes: LensType[];
  position: number;
  activePin: PinToken | null;
  onSelectLayer: (layer: number) => void;
}) {
  const token = tokens.find((t) => t.position === position);
  return (
    <div className="flex min-h-0 flex-col rounded-md border border-slate-200 bg-white">
      <div className="flex h-7 shrink-0 items-center justify-between border-b border-slate-100 px-2 text-[10px] font-semibold uppercase text-slate-400">
        <span>Layers At Position</span>
        <span className="font-mono normal-case text-slate-500">pos {position}</span>
      </div>
      <div className="max-h-[220px] overflow-auto p-1">
        {layers.map((layer) => (
          <button
            type="button"
            key={`layer-readout-${layer}`}
            onClick={() => onSelectLayer(layer)}
            className="grid w-full grid-cols-[38px_1fr] gap-x-2 rounded px-1.5 py-1 text-left hover:bg-slate-50"
          >
            <span className="text-right text-[10px] font-semibold tabular-nums text-slate-400">L{layer}</span>
            <span className="flex min-w-0 flex-col gap-y-0.5">
              {displayedTypes.map((type) => {
                const items = rowItems(sliceFor(token, type), type, rowIndexForLayer(layersByType, type, layer), 3);
                return (
                  <span key={type} className="flex min-w-0 items-center gap-x-1 text-[10px]">
                    <span className={type === LensType.JACOBIAN_LENS ? 'text-sky-600' : 'text-emerald-600'}>
                      {type === LensType.JACOBIAN_LENS ? 'J' : 'L'}
                    </span>
                    {items.map((item) => (
                      <span
                        key={`${type}-${item.slot}-${item.token}`}
                        className={`min-w-0 truncate rounded px-1 font-mono ${
                          activePin && itemMatchesPin(item, activePin)
                            ? 'bg-sky-100 text-sky-800'
                            : 'bg-slate-100 text-slate-700'
                        }`}
                      >
                        {displayToken(item.token)}
                      </span>
                    ))}
                  </span>
                );
              })}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

function PositionReadout({
  tokens,
  layer,
  layersByType,
  displayedTypes,
  activePin,
  onSelectPosition,
}: {
  tokens: LensTokenMessage[];
  layer: number;
  layersByType: Record<string, number[]>;
  displayedTypes: LensType[];
  activePin: PinToken | null;
  onSelectPosition: (position: number) => void;
}) {
  return (
    <div className="flex min-h-0 flex-col rounded-md border border-slate-200 bg-white">
      <div className="flex h-7 shrink-0 items-center justify-between border-b border-slate-100 px-2 text-[10px] font-semibold uppercase text-slate-400">
        <span>Positions At Layer</span>
        <span className="font-mono normal-case text-slate-500">layer {layer}</span>
      </div>
      <div className="max-h-[220px] overflow-auto p-1">
        {tokens.map((token) => (
          <button
            type="button"
            key={`position-readout-${token.position}`}
            onClick={() => onSelectPosition(token.position)}
            className="grid w-full grid-cols-[44px_1fr] gap-x-2 rounded px-1.5 py-1 text-left hover:bg-slate-50"
          >
            <span className="text-right text-[10px] font-semibold tabular-nums text-slate-400">P{token.position}</span>
            <span className="flex min-w-0 flex-col gap-y-0.5">
              {displayedTypes.map((type) => {
                const items = rowItems(sliceFor(token, type), type, rowIndexForLayer(layersByType, type, layer), 3);
                return (
                  <span key={type} className="flex min-w-0 items-center gap-x-1 text-[10px]">
                    <span className={type === LensType.JACOBIAN_LENS ? 'text-sky-600' : 'text-emerald-600'}>
                      {type === LensType.JACOBIAN_LENS ? 'J' : 'L'}
                    </span>
                    {items.map((item) => (
                      <span
                        key={`${type}-${item.slot}-${item.token}`}
                        className={`min-w-0 truncate rounded px-1 font-mono ${
                          activePin && itemMatchesPin(item, activePin)
                            ? 'bg-sky-100 text-sky-800'
                            : 'bg-slate-100 text-slate-700'
                        }`}
                      >
                        {displayToken(item.token)}
                      </span>
                    ))}
                  </span>
                );
              })}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

export default function JlensSliceWorkspace({ analysis, tokens }: { analysis: JlensAnalysis; tokens: LensTokenMessage[] }) {
  const layersByType = analysis.layersByType;
  const availableTypes = useMemo(
    () => lensTypesForMode(analysis.lensMode).filter((type) => (layersByType[type]?.length ?? 0) > 0),
    [analysis.lensMode, layersByType],
  );
  const primaryType = availableTypes[0] ?? LensType.LOGIT_LENS;
  const layers = useMemo(() => layersByType[primaryType] ?? [], [layersByType, primaryType]);
  const positions = useMemo(() => tokens.map((token) => token.position), [tokens]);
  const selectedFromTranscript = useMemo(() => Array.from(analysis.selectedPositions), [analysis.selectedPositions]);
  const [selectedPosition, setSelectedPosition] = useState<number | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [hoveredCoord, setHoveredCoord] = useState<CellCoord | null>(null);
  const [activePinKey, setActivePinKey] = useState<string | null>(null);

  useEffect(() => {
    const transcriptPos = selectedFromTranscript.find((pos) => positions.includes(pos));
    setSelectedPosition((current) => clampSelection(positions, transcriptPos ?? current, positions[0] ?? null));
  }, [positions, selectedFromTranscript]);

  useEffect(() => {
    setSelectedLayer((current) => clampSelection(layers, current, layers[0] ?? null));
  }, [layers]);

  const selectedPins = useMemo<PinToken[]>(
    () =>
      analysis.selected
        .map((pin, index) => ({
          ...pin,
          color: SELECT_COLORS[Math.min(index, SELECT_COLORS.length - 1)],
          label: tokenLabelForKey(tokens, pin.type, pin.key),
        }))
        .filter((pin) => (layersByType[pin.type]?.length ?? 0) > 0),
    [analysis.selected, layersByType, tokens],
  );
  useEffect(() => {
    if (selectedPins.length === 0) {
      setActivePinKey(null);
      return;
    }
    if (!activePinKey || !selectedPins.some((pin) => `${pin.type}:${pin.key}` === activePinKey)) {
      setActivePinKey(`${selectedPins[0].type}:${selectedPins[0].key}`);
    }
  }, [activePinKey, selectedPins]);

  if (tokens.length === 0 || layers.length === 0 || availableTypes.length === 0 || selectedPosition == null || selectedLayer == null) {
    return null;
  }

  const activePin = selectedPins.find((pin) => `${pin.type}:${pin.key}` === activePinKey) ?? null;
  const selectedCoord = { position: selectedPosition, layer: selectedLayer };
  const readoutCoord = hoveredCoord ?? selectedCoord;
  const activeRankLayers = activePin ? (layersByType[activePin.type] ?? []) : [];
  const rankLayer = activeRankLayers.includes(selectedLayer) ? selectedLayer : (activeRankLayers[0] ?? selectedLayer);
  const layerRanks =
    activePin && activeRankLayers.length > 0
      ? activeRankLayers.map((layer) => rankAt(tokens, layersByType, selectedPosition, layer, activePin)?.rank ?? null)
      : [];
  const positionRanks =
    activePin && activeRankLayers.length > 0
      ? tokens.map((token) => rankAt(tokens, layersByType, token.position, rankLayer, activePin)?.rank ?? null)
      : [];
  const hasFullRanksForReturnedTokens = tokens.some((token) =>
    token.results.some((slice) => slice.top_ranks?.some((row) => row.some((rank) => typeof rank === 'number'))),
  );

  const selectCoord = (coord: CellCoord) => {
    setSelectedPosition(coord.position);
    setSelectedLayer(coord.layer);
    analysis.setSelectedPositions(new Set([coord.position]));
  };
  const selectPosition = (position: number) => {
    setSelectedPosition(position);
    analysis.setSelectedPositions(new Set([position]));
  };
  const togglePin = (item: ReadoutItem) => {
    analysis.toggleSelect(normKey(item.token), item.type);
  };
  const pinDisabled = analysis.selected.length >= MAX_SELECT;

  return (
    <section className="flex w-full flex-col gap-y-3 rounded-lg border border-slate-200 bg-slate-50 p-2 shadow-sm sm:p-3">
      <div className="flex flex-col gap-y-2">
        <div className="flex items-center justify-between gap-x-3">
          <div className="min-w-0">
            <div className="text-[11px] font-semibold uppercase text-slate-400">Slice Workspace</div>
            <div className="truncate text-[12px] text-slate-600">
              {analysis.lensMode === LensMode.DIFF ? 'Jacobian vs Logit' : analysis.lensModeLabel} · returned top-
              {tokens[0]?.results[0]?.top_tokens[0]?.length ?? 0}
              {hasFullRanksForReturnedTokens ? ' · full ranks for returned tokens' : ' · returned slots only'}
            </div>
          </div>
          <div className="flex min-w-0 flex-wrap justify-end gap-1">
            {selectedPins.length > 0 ? (
              selectedPins.map((pin) => {
                const active = activePinKey === `${pin.type}:${pin.key}`;
                return (
                  <button
                    type="button"
                    key={`${pin.type}:${pin.key}`}
                    onClick={() => setActivePinKey(`${pin.type}:${pin.key}`)}
                    className={`flex max-w-[160px] items-center gap-x-1 rounded border px-2 py-1 font-mono text-[10px] leading-none ${
                      active ? 'border-sky-400 bg-white shadow-sm' : 'border-slate-200 bg-slate-100 hover:bg-white'
                    }`}
                    style={{ color: `rgb(${pinTextColor(pin.color)})` }}
                  >
                    <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ backgroundColor: `rgb(${COLOR_RGB[pin.color]})` }} />
                    <span className="truncate">{displayToken(pin.label)}</span>
                  </button>
                );
              })
            ) : (
              <span className="rounded border border-dashed border-slate-200 bg-white px-2 py-1 text-[10px] text-slate-400">
                no pinned readouts
              </span>
            )}
          </div>
        </div>

        <div className="flex gap-x-1 overflow-x-auto rounded-md border border-slate-200 bg-white p-1">
          {tokens.map((token) => (
            <button
              type="button"
              key={`prompt-token-${token.position}`}
              onClick={() => selectPosition(token.position)}
              className={`h-8 max-w-[92px] shrink-0 truncate rounded border px-2 font-mono text-[10px] leading-none ${promptTokenClass(
                selectedPosition === token.position,
                token.is_generated,
              )}`}
              title={`${token.position}: ${token.token}`}
            >
              {displayToken(token.token)}
            </button>
          ))}
        </div>
      </div>

      <SliceGrid
        tokens={tokens}
        layers={layers}
        layersByType={layersByType}
        lensMode={analysis.lensMode}
        displayedTypes={availableTypes}
        selectedCoord={selectedCoord}
        hoveredCoord={hoveredCoord}
        onHover={setHoveredCoord}
        onSelect={selectCoord}
      />

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(300px,380px)]">
        <div className="flex min-w-0 flex-col gap-y-3">
          <CellReadout
            coord={readoutCoord}
            tokens={tokens}
            layersByType={layersByType}
            displayedTypes={availableTypes}
            selectedPins={selectedPins}
            pinDisabled={pinDisabled}
            onTogglePin={togglePin}
          />
          <RankHeatmap
            tokens={tokens}
            layers={activePin ? (layersByType[activePin.type] ?? []) : layers}
            layersByType={layersByType}
            pin={activePin}
            selectedCoord={selectedCoord}
            onSelect={selectCoord}
          />
          {activePin && (
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
              <RankSparkline
                title={`Rank Across Layers · pos ${selectedPosition}`}
                domain={activeRankLayers}
                ranks={layerRanks}
                color={activePin.color}
              />
              <RankSparkline
                title={`Rank Across Positions · layer ${rankLayer}`}
                domain={tokens.map((token) => token.position)}
                ranks={positionRanks}
                color={activePin.color}
              />
            </div>
          )}
        </div>
        <div className="grid min-h-0 grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-1">
          <LayerReadout
            tokens={tokens}
            layers={layers}
            layersByType={layersByType}
            displayedTypes={availableTypes}
            position={selectedPosition}
            activePin={activePin}
            onSelectLayer={(layer) => setSelectedLayer(layer)}
          />
          <PositionReadout
            tokens={tokens}
            layer={selectedLayer}
            layersByType={layersByType}
            displayedTypes={availableTypes}
            activePin={activePin}
            onSelectPosition={selectPosition}
          />
        </div>
      </div>
    </section>
  );
}
