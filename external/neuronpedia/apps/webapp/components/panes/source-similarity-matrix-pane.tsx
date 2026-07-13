'use client';

import { Button } from '@/components/shadcn/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shadcn/card';
import { SourceWithPartialRelations } from '@/prisma/generated/zod';
import { Grid } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useGlobalContext } from '../provider/global-provider';
import { DEMO_BUTTONS } from '../similarity-matrix-modal';

export default function SourceSimilarityMatrixPane({
  source,
  initialSimMatrixText,
  initialSimMatrixDemo,
}: {
  source: SourceWithPartialRelations;
  initialSimMatrixText?: string;
  initialSimMatrixDemo?: string;
}) {
  const { setSimilarityMatrix } = useGlobalContext();

  // Resolve the initial text from either simMatrix or simMatrixDemo
  const resolvedInitialText = initialSimMatrixDemo
    ? DEMO_BUTTONS.find((demo) => demo.id === initialSimMatrixDemo)?.text || ''
    : initialSimMatrixText || '';

  const [customText, setCustomText] = useState<string>(resolvedInitialText);
  const hasOpenedRef = useRef(false);

  // Only open the modal once on mount if there's initial text from URL
  useEffect(() => {
    if (resolvedInitialText && !hasOpenedRef.current) {
      setSimilarityMatrix(source, resolvedInitialText);
      hasOpenedRef.current = true;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex w-full flex-col items-center">
      <Card className="mt-0 w-full max-w-screen-lg bg-white">
        <CardHeader className="w-full pb-3 pt-5">
          <div className="flex w-full flex-row items-center justify-between">
            <CardTitle>Similarity Matrix</CardTitle>
            <a href="https://arxiv.org/abs/2511.01836" target="_blank" rel="noreferrer">
              <Button
                variant="outline"
                size="sm"
                className="flex flex-row gap-x-2 rounded-full text-sm font-semibold text-slate-400 shadow-sm"
              >
                ?
              </Button>
            </a>
          </div>
        </CardHeader>
        <CardContent className="flex flex-col gap-0 pt-0">
          <div className="flex w-full gap-2">
            <textarea
              value={customText}
              onChange={(e) => setCustomText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  setSimilarityMatrix(source, customText);
                }
              }}
              placeholder="Choose a demo below, or enter some text to generate a custom similarity matrix."
              className="flex-1 resize-none rounded border border-slate-300 px-3 py-2 text-[14px] leading-normal text-slate-800 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500"
              rows={3}
            />
            <Button
              className="flex h-auto flex-col gap-y-1 rounded-md bg-sky-700 px-3 text-[9.5px] font-medium text-white hover:bg-sky-800 hover:text-white"
              variant="outline"
              onClick={() => {
                setSimilarityMatrix(source, customText);
              }}
            >
              <Grid className="h-4 w-4" /> GENERATE
            </Button>
          </div>
          <div className="mb-0.5 mt-5 flex flex-row items-center justify-start gap-x-2 text-center font-sans text-[10px] uppercase text-slate-500">
            Run Demo
          </div>
          <div className="grid grid-cols-3 gap-x-1.5 gap-y-1.5 sm:grid-cols-6">
            {DEMO_BUTTONS.map((demo) => (
              <Button
                key={demo.label}
                variant="outline"
                size="sm"
                className="h-12 gap-x-2 shadow-sm"
                onClick={() => {
                  setSimilarityMatrix(source, demo.text);
                }}
              >
                {demo.label}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
