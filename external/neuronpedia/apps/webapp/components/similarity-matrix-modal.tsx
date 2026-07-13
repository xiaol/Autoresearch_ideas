'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import { NEXT_PUBLIC_URL } from '@/lib/env';
import * as Dialog from '@radix-ui/react-dialog';
import copy from 'copy-to-clipboard';
import { Grid, RotateCcw, Share } from 'lucide-react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import { LoadingSquare } from './svg/loading-square';

// Demo button configurations
export const DEMO_BUTTONS = [
  {
    id: 'alice-maya',
    label: '👯‍♀️ Alice & Maya',
    text: "Once upon a time, a little girl named Alice loved looking at the night sky. 'I wish I could count all the stars!' Alice said to her best friend Maya. The two girls stood on a big grass field as the moon rose from the trees. Suddenly, Maya had a striking idea. She opened her laptop and started typing:\n```python\narray = []\nfor i in range(1, 6):\n    s = int(input(f'num_stars:'))\n    array.append(s)\ntot = sum(array)\navg = tot / len(array)\nprint(f'Avg / night: {avg:.1f}')",
  },
  {
    id: 'mech-interp',
    label: '🧠 Mech Interp',
    text: 'Mechanistic interpretability (often abbreviated as mech interp, mechinterp, or MI) is a subfield of research within explainable artificial intelligence that aims to understand the internal workings of neural networks by analyzing the mechanisms present in their computations. The approach seeks to analyze neural networks in a manner similar to how binary computer programs can be reverse-engineered to understand their functions.',
  },
  {
    id: 'audrey-job',
    label: "👔 Audrey's Job",
    text: 'AUDREY was three days into her new job and still elated about gaining her position at Fayburns, the world-famous West End department store. Glamour and glitz were returning to the big stores, which were starting to lavish attention on their window displays as they had before the war. As a recent arts school graduate, Audrey was well qualified to fill the new role of window dresser and display co-ordinator. Having said that, beginning her job in the run-up to Christmas felt like jumping in at the deep end. She\'d been charged with creating a series of extravagant tableaux, in the three windows facing the street, on the theme of the wonderful old poem, "The Night Before Christmas".',
  },
  {
    id: 'obama-ice-cream',
    label: '🍦 Ice Cream',
    text: 'In a LinkedIn post published today, President Obama announced his "Summer Opportunity Project" by talking about his first job: scooping ice cream.\n\n"Scooping ice cream is tougher than it looks. Rows and rows of rock-hard ice cream can be brutal on the wrists," he wrote. "As a teenager working behind the counter at Baskin-Robbins in Honolulu, I was less interested in what the job meant for my future and more concerned about what it meant for my jump shot."',
  },
  {
    id: 'huck-finn',
    label: '📖 Huck Finn',
    text: "You don't know about me without you have read a book by the name of The Adventures of Tom Sawyer; but that ain't no matter.  That book was made by Mr. Mark Twain, and he told the truth, mainly.  There was things which he stretched, but mainly he told the truth.  That is nothing.",
  },
  {
    id: 'dr-seuss',
    label: '🐱 Dr. Seuss',
    text: "You have brains in your head.\nYou have feet in your shoes.\nYou can steer yourself in any direction you choose.\nYou're on your own.\nAnd you know what you know.\nYou are the guy who'll decide where to go.\n~Dr. Seuss",
  },
];

export default function SimilarityMatrixModal() {
  const { similarityMatrixSource, similarityMatrixText, similarityMatrixModalOpen, setSimilarityMatrixModalOpen } =
    useGlobalContext();

  const router = useRouter();
  const searchParams = useSearchParams();

  // State for tokens and similarity matrix
  const [tokens, setTokens] = useState<string[]>([]);
  const [similarityMatrix, setSimilarityMatrix] = useState<number[][]>([]);
  const [selectedToken, setSelectedToken] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [customText, setCustomText] = useState<string>(similarityMatrixText || DEMO_BUTTONS[0].text);
  const [cellSize, setCellSize] = useState<number>(0);
  const [lastGeneratedText, setLastGeneratedText] = useState<string>('');

  const containerRef = useRef<HTMLDivElement>(null);

  // Function to reset all state
  const resetState = () => {
    setTokens([]);
    setSimilarityMatrix([]);
    setSelectedToken(null);
    setError(null);
    setCustomText(DEMO_BUTTONS[0].text);
    setLoading(false);
    setLastGeneratedText('');
  };

  // Function to clear URL params when closing modal
  const clearUrlParams = () => {
    const params = new URLSearchParams(searchParams.toString());
    params.delete('simMatrix');
    params.delete('simMatrixDemo');
    // pathname is the NEXT_PUBLIC_URL + modelId + sourceId
    const pathname = `${NEXT_PUBLIC_URL}/${similarityMatrixSource?.modelId}/${similarityMatrixSource?.id}/`;
    router.replace(`${pathname}?${params.toString()}`, { scroll: false });
  };

  // Function to close modal and clean up
  const closeModal = () => {
    resetState();
    clearUrlParams();
    setSimilarityMatrixModalOpen(false);
  };

  // Function to fetch similarity matrix
  const fetchSimilarityMatrix = async (text: string) => {
    try {
      // Clear tokens and matrix immediately
      setTokens([]);
      setSimilarityMatrix([]);
      setLoading(true);
      setError(null);
      setSelectedToken(null);

      if (!similarityMatrixSource?.modelId) {
        alert('No model ID found for similarity matrix source');
        setLoading(false);
        return;
      }
      if (!similarityMatrixSource?.id) {
        alert('No source ID found for similarity matrix source');
        setLoading(false);
        return;
      }

      const { modelId } = similarityMatrixSource;
      const sourceId = similarityMatrixSource.id;

      // Skip if text is blank
      if (!text.trim()) {
        setLoading(false);
        return;
      }

      const resp = await fetch('/api/similarity-matrix-pred', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modelId, sourceId, text }),
      });
      if (!resp.ok) {
        const detail = await resp.text();
        throw new Error(detail || 'Request failed');
      }
      const data = await resp.json();
      const newTokens: string[] = data.tokens || [];
      const matrix: number[][] = data.similarity_matrix || [];
      setTokens(newTokens);
      setSimilarityMatrix(matrix);
      setLastGeneratedText(text);

      // Update URL search params to make it shareable
      const params = new URLSearchParams(searchParams.toString());

      // Check if the text matches any demo button
      const matchingDemo = DEMO_BUTTONS.find((demo) => demo.text === text);

      if (matchingDemo) {
        // If it matches a demo, use the short demo ID
        params.set('simMatrixDemo', matchingDemo.id);
        params.delete('simMatrix');
      } else {
        // If it's custom text, use the full text
        params.set('simMatrix', text);
        params.delete('simMatrixDemo');
      }

      const pathname = `${NEXT_PUBLIC_URL}/${similarityMatrixSource?.modelId}/${similarityMatrixSource?.id}/`;
      router.replace(`${pathname}?${params.toString()}`, { scroll: false });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  // Update customText when similarityMatrixText changes (from URL params)
  useEffect(() => {
    if (similarityMatrixText) {
      setCustomText(similarityMatrixText);
    }
  }, [similarityMatrixText]);

  // Fetch similarity matrix from server when modal opens
  useEffect(() => {
    if (!similarityMatrixModalOpen) return;

    // Determine what text to use
    const textToGenerate = similarityMatrixText || customText;

    // Only auto-fetch if we haven't generated this text yet
    if (textToGenerate && textToGenerate !== lastGeneratedText) {
      fetchSimilarityMatrix(textToGenerate);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [similarityMatrixModalOpen, similarityMatrixText]);

  // Calculate cell size based on available height
  useEffect(() => {
    const calculateCellSize = () => {
      if (!containerRef.current || tokens.length === 0) return;

      const availableHeight = containerRef.current.clientHeight;
      const availableWidth = containerRef.current.clientWidth;
      // Account for mobile vs desktop padding
      const isMobile = window.innerWidth < 640; // sm breakpoint
      const padding = isMobile ? 0 : 32; // 0px each side = 0px total for mobile, 16px each side = 32px total for desktop
      const usableHeight = availableHeight - padding;
      const usableWidth = availableWidth - padding;

      // Calculate the maximum cell size that fits both dimensions
      const maxCellSizeByHeight = Math.floor(usableHeight / tokens.length);
      const maxCellSizeByWidth = Math.floor(usableWidth / tokens.length);

      // Use the smaller of the two to ensure it fits
      const calculatedSize = Math.min(maxCellSizeByHeight, maxCellSizeByWidth);

      // Set maximum cell size only (no minimum)
      const finalSize = Math.min(100, calculatedSize);

      setCellSize(finalSize);
    };

    calculateCellSize();

    // Recalculate on window resize
    window.addEventListener('resize', calculateCellSize);
    return () => window.removeEventListener('resize', calculateCellSize);
  }, [tokens.length, similarityMatrix, similarityMatrixModalOpen]);

  // Magma color scheme interpolation
  const getMagmaColor = (value: number) => {
    // Magma colormap approximation: dark purple -> pink -> orange -> yellow
    const colors = [
      { pos: 0.0, r: 0, g: 0, b: 4 },
      { pos: 0.25, r: 60, g: 15, b: 80 },
      { pos: 0.5, r: 180, g: 50, b: 120 },
      { pos: 0.75, r: 250, g: 140, b: 70 },
      { pos: 1.0, r: 252, g: 253, b: 191 },
    ];

    let lower = colors[0];
    let upper = colors[colors.length - 1];

    for (let i = 0; i < colors.length - 1; i += 1) {
      if (value >= colors[i].pos && value <= colors[i + 1].pos) {
        lower = colors[i];
        upper = colors[i + 1];
        break;
      }
    }

    const range = upper.pos - lower.pos;
    const rangePct = range === 0 ? 0 : (value - lower.pos) / range;

    const r = Math.round(lower.r + (upper.r - lower.r) * rangePct);
    const g = Math.round(lower.g + (upper.g - lower.g) * rangePct);
    const b = Math.round(lower.b + (upper.b - lower.b) * rangePct);

    return `rgb(${r}, ${g}, ${b})`;
  };

  return (
    <Dialog.Root open={similarityMatrixModalOpen}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-slate-600/20" />
        <Dialog.Content
          onPointerDownOutside={() => {
            closeModal();
          }}
          className="fixed left-[50%] top-[50%] z-50 flex h-[100vh] max-h-[100vh] w-[100vw] max-w-[100%] translate-x-[-50%] translate-y-[-50%] flex-col overflow-hidden bg-slate-50 shadow-xl focus:outline-none sm:top-[50%] sm:h-[90vh] sm:max-h-[90vh] sm:w-[95vw] sm:max-w-[95%] sm:rounded-md"
        >
          <div
            className="relative flex w-full flex-row items-start justify-between gap-x-4 rounded-t-md border-b bg-white px-2 pb-2 pt-2 sm:px-4"
            style={{ height: '45px' }}
          >
            <button
              type="button"
              className="flex flex-row items-center justify-center gap-x-1 rounded-full bg-slate-300 px-3 py-1.5 text-[11px] text-slate-600 hover:bg-sky-700 hover:text-white focus:outline-none"
              aria-label="Close"
              onClick={() => {
                closeModal();
              }}
            >
              Done
            </button>
            <Dialog.Title className="absolute left-1/2 top-[55%] -translate-x-1/2 -translate-y-1/2 text-center text-sm font-medium text-slate-600">
              Similarity Matrix
            </Dialog.Title>
          </div>

          {/* Text input and tokens - auto height */}

          <div className="flex w-full flex-1 flex-col items-stretch justify-start overflow-hidden bg-white sm:flex-row">
            <div className="hidden flex-shrink-0 flex-col gap-2 bg-slate-50 px-2 pb-3 pt-3 sm:flex sm:w-1/5 sm:px-4">
              <div className="mt-2 text-center text-sm font-medium text-slate-700">Instructions</div>
              <ol className="list-decimal space-y-1 pl-4 text-xs text-slate-600">
                <li>Type some text, ideally at least a sentence.</li>
                <li>Click &quot;Generate&quot;.</li>
                <li>Hover over tokens to highlight their matrix cell.</li>
                <li>Hover over cells to see which token it&apos;s on.</li>
              </ol>
              <div className="mt-3 text-center text-sm font-medium text-slate-700">Click a Demo</div>
              {DEMO_BUTTONS.map((demo, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => {
                    setCustomText(demo.text);
                    fetchSimilarityMatrix(demo.text);
                  }}
                  disabled={customText === demo.text}
                  className="rounded bg-sky-700 px-3 py-3 text-xs text-white hover:bg-sky-800 focus:outline-none focus:ring-2 focus:ring-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {demo.label} {customText === demo.text ? '(Selected)' : ''}
                </button>
              ))}
            </div>
            <div className="flex flex-shrink-0 flex-wrap content-start items-start justify-start gap-1 gap-y-[3px] overflow-y-auto p-4 sm:w-1/3">
              <div className="mb-1.5 w-full text-center font-mono text-xs font-bold uppercase text-slate-600">
                {similarityMatrixSource?.modelId} @ {similarityMatrixSource?.id}
              </div>
              <div className="flex w-full gap-2">
                <textarea
                  value={customText}
                  onChange={(e) => setCustomText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      fetchSimilarityMatrix(customText);
                    }
                  }}
                  placeholder="Enter some text, then click 'Generate'."
                  className="max-h-[80px] flex-1 rounded border border-slate-300 px-3 py-2 text-[12px] leading-normal text-slate-800 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 sm:max-h-none"
                  disabled={loading}
                  rows={10}
                />
                <div className="flex w-[80px] flex-col gap-1.5">
                  <button
                    type="button"
                    onClick={() => fetchSimilarityMatrix(customText)}
                    disabled={loading || !customText.trim()}
                    className="flex w-full flex-1 flex-col items-center justify-center gap-y-1 rounded bg-sky-700 px-2 py-2 text-[11px] font-medium text-white hover:bg-sky-800 focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <Grid className="h-4 w-4" /> Generate
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setCustomText('');
                      setTokens([]);
                      setSimilarityMatrix([]);
                      setSelectedToken(null);
                      setError(null);
                    }}
                    disabled={loading}
                    className="hidden w-full flex-1 flex-col items-center justify-center gap-y-1 rounded bg-slate-600 px-2 py-2 text-[11px] font-medium text-white hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 sm:flex"
                  >
                    <RotateCcw className="h-4 w-4" /> Reset
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      const url = window.location.href;
                      copy(url);
                      alert('URL copied to clipboard!');
                    }}
                    disabled={loading || tokens.length === 0}
                    className="hidden w-full flex-1 flex-col items-center justify-center gap-y-1 rounded bg-emerald-600 px-2 py-2 text-[11px] font-medium text-white hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 sm:flex"
                  >
                    <Share className="h-4 w-4" /> Share
                  </button>
                </div>
              </div>
              {tokens.length > 0 && (
                <div className="mt-4 flex max-h-[120px] w-full flex-shrink-0 flex-wrap content-start items-start justify-start gap-1 gap-y-1.5 overflow-scroll overflow-y-auto border-t px-0 pt-4 sm:max-h-none sm:overflow-auto">
                  {tokens.map((token, i) => (
                    <div
                      key={i}
                      onMouseEnter={() => setSelectedToken(i)}
                      onMouseLeave={() => setSelectedToken(null)}
                      className={`cursor-default select-none rounded py-[1px] font-mono text-[9px] leading-normal transition-all sm:text-[11px] ${
                        selectedToken === i
                          ? 'border border-sky-700 bg-sky-700 text-white'
                          : 'border border-transparent bg-slate-100 text-slate-700 hover:bg-blue-100'
                      } ${
                        token.startsWith(' ') && token.endsWith(' ')
                          ? 'px-2'
                          : token.startsWith(' ')
                            ? 'pl-2 pr-0.5'
                            : token.endsWith(' ')
                              ? 'pl-0.5 pr-2'
                              : 'px-0.5'
                      }`}
                    >
                      {token.includes('\n')
                        ? token.replaceAll('\n', '↵')
                        : token.trim() === ''
                          ? token.replaceAll(' ', '\u00A0')
                          : token}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Matrix container - takes remaining height */}
            <div
              ref={containerRef}
              className="flex flex-1 flex-col items-center justify-start overflow-hidden border-t bg-white pt-5 sm:justify-center sm:border-t-0 sm:pt-0"
            >
              {loading && (
                <div className="flex flex-col items-center justify-center gap-y-3 text-center text-xs text-slate-500">
                  <LoadingSquare className="h-6 w-6 sm:h-8 sm:w-8" size={32} /> Loading similarity matrix…
                </div>
              )}
              {error && <div className="rounded bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>}

              {!loading && !error && (
                <div className="-mt-3 flex flex-col items-center justify-center gap-4">
                  {/* Horizontal Colorbar */}
                  {similarityMatrix.length > 0 && (
                    <div className="flex flex-col items-center gap-0">
                      <div className="mb-1 text-[9px] font-medium uppercase text-slate-400">Similarity</div>
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-[11px] text-slate-600">0.0</span>
                        <div className="border-1 flex overflow-hidden rounded-md border-slate-300">
                          {Array.from({ length: 50 }).map((_, i) => {
                            const value = i / 49; // Go from 0.0 to 1.0 (left to right)
                            return (
                              <div
                                key={i}
                                className="w-[4px] sm:w-[8px]"
                                style={{
                                  height: '24px',
                                  backgroundColor: getMagmaColor(value),
                                }}
                              />
                            );
                          })}
                        </div>
                        <span className="font-mono text-[11px] text-slate-600">1.0</span>
                      </div>
                    </div>
                  )}

                  {/* Heatmap */}
                  <div className="inline-block">
                    <div className="flex">
                      {/* Similarity matrix heatmap */}
                      <div className="inline-block border-[0.5px] border-slate-400">
                        {similarityMatrix.map((row, i) => (
                          <div key={i} className="flex">
                            {row.map((value, j) => {
                              const isInSelectedRow = selectedToken !== null && i === selectedToken;
                              const isInSelectedCol = selectedToken !== null && j === selectedToken;
                              const isFirstInRow = j === 0;
                              const isLastInRow = j === tokens.length - 1;
                              const isFirstInCol = i === 0;
                              const isLastInCol = i === tokens.length - 1;

                              const borderStyle = '0.5px solid rgba(0,0,0,0.2)';
                              let borderTop = borderStyle;
                              let borderBottom = borderStyle;
                              let borderLeft = borderStyle;
                              let borderRight = borderStyle;

                              const highlightBorder = '2px solid #3b82f6';

                              if (isInSelectedRow) {
                                borderTop = highlightBorder;
                                borderBottom = highlightBorder;
                                if (isFirstInRow) borderLeft = highlightBorder;
                                if (isLastInRow) borderRight = highlightBorder;
                              }

                              if (isInSelectedCol) {
                                borderLeft = highlightBorder;
                                borderRight = highlightBorder;
                                if (isFirstInCol) borderTop = highlightBorder;
                                if (isLastInCol) borderBottom = highlightBorder;
                              }

                              return (
                                <div
                                  key={`${i}-${j}`}
                                  onMouseEnter={() => setSelectedToken(Math.min(i, j))}
                                  onMouseLeave={() => setSelectedToken(null)}
                                  className="flex items-center justify-center font-mono text-[9.5px] transition-all"
                                  style={{
                                    width: `${cellSize}px`,
                                    height: `${cellSize}px`,
                                    backgroundColor: getMagmaColor(value),
                                    color: value > 0.5 ? 'black' : 'white',
                                    borderTop,
                                    borderBottom,
                                    borderLeft,
                                    borderRight,
                                  }}
                                >
                                  {/* {value.toFixed(2)} */}
                                </div>
                              );
                            })}
                          </div>
                        ))}
                      </div>
                    </div>
                    {/* 
              <div className="mt-6 text-center text-sm text-gray-600">
                {selectedToken !== null
                  ? `Selected: Token "${tokens[selectedToken]}" (position ${selectedToken})`
                  : 'Click a token or any cell to highlight its row and column'}
              </div> */}
                  </div>
                </div>
              )}
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
