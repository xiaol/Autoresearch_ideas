'use client';

import { Button } from '@/components/shadcn/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shadcn/card';
import { GraphMetadata, ModelWithPartialRelations } from '@/prisma/generated/zod';
import * as Select from '@radix-ui/react-select';
import { ChevronDownIcon, PlusIcon } from 'lucide-react';
import { Fragment } from 'react';

export default function GraphModelPane({
  model,
  graphMetadatas,
}: {
  model: ModelWithPartialRelations;
  graphMetadatas: GraphMetadata[];
}) {
  return (
    <div className="flex w-full flex-col items-center">
      <Card className="mt-0 w-full max-w-screen-lg bg-white">
        <CardHeader className="w-full pb-3 pt-5">
          <div className="flex w-full flex-row items-center justify-between">
            <CardTitle>Circuit Tracing</CardTitle>
            <a href="https://neuronpedia.org/graph/info" target="_blank" rel="noreferrer">
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
        <CardContent className="flex w-full flex-row items-center justify-center gap-x-2">
          <Select.Root
            value={undefined}
            onValueChange={(newVal) => {
              window.location.href = `/${model.id}/graph?slug=${newVal}`;
            }}
          >
            <Select.Trigger
              onKeyDown={(e) => {
                if (e.key === 'g') {
                  e.preventDefault();
                  e.stopPropagation();
                }
              }}
              className="relative inline-flex h-10 max-w-screen-sm flex-1 items-center justify-between gap-1 overflow-x-hidden rounded border border-slate-300 bg-white py-0 pl-3 pr-10 text-sm leading-none focus:outline-none focus:ring-0"
            >
              <div className="w-full flex-1 flex-col items-center justify-center text-slate-400">
                Choose a Featured Graph to Open
              </div>
              <Select.Icon className="absolute right-0 flex h-[100%] w-8 items-center justify-center bg-white">
                <ChevronDownIcon className="ml-1 w-5 text-slate-500" />
              </Select.Icon>
            </Select.Trigger>
            <Select.Portal>
              <Select.Content
                position="popper"
                align="start"
                sideOffset={3}
                className="z-[99999] max-h-[640px] w-full max-w-[800px] overflow-hidden overscroll-y-none rounded-md border bg-white shadow-lg"
              >
                <Select.Viewport className="w-full p-0 text-slate-700">
                  {graphMetadatas.map((graph) => (
                    <Select.Group key={graph.slug} className="divide-y divide-slate-200">
                      <div className="relative flex w-full flex-row items-center hover:bg-sky-100">
                        <Select.Item
                          key={graph.slug}
                          value={graph.slug}
                          className="group relative flex w-full cursor-pointer select-none items-center overflow-x-hidden py-2.5 pl-4 pr-4 text-xs outline-none hover:bg-slate-100 data-[highlighted]:bg-sky-50"
                        >
                          <Select.ItemText className="w-full min-w-full" asChild>
                            <div className="flex w-full min-w-full flex-col items-start justify-start gap-y-0">
                              <div className="flex w-full flex-row items-center justify-between">
                                <div className="font-mono text-[10px] font-medium text-sky-700">{graph.slug}</div>
                              </div>
                              <div className="mt-2 w-full whitespace-pre-line pl-0 text-[10px] leading-tight text-slate-500">
                                <div className="flex flex-wrap">
                                  {graph.promptTokens.map((token, i) => (
                                    <Fragment key={`${token}-${i}`}>
                                      <span className="mx-[1px] mb-1 rounded bg-slate-100 px-[2px] py-0.5 font-mono text-slate-700 group-hover:bg-sky-200 group-hover:text-sky-700 group-data-[highlighted]:bg-sky-200 group-data-[highlighted]:text-sky-700">
                                        {token.replaceAll(' ', '\u00A0')}
                                      </span>
                                      {(token === '⏎' || token === '⏎⏎') && <div className="w-full" />}
                                    </Fragment>
                                  ))}
                                </div>
                              </div>
                            </div>
                          </Select.ItemText>
                        </Select.Item>
                      </div>
                    </Select.Group>
                  ))}
                </Select.Viewport>
                <Select.ScrollDownButton className="flex h-7 cursor-pointer items-center justify-center bg-white text-slate-700 hover:bg-slate-100">
                  <ChevronDownIcon className="w-5 text-slate-500" />
                </Select.ScrollDownButton>
              </Select.Content>
            </Select.Portal>
          </Select.Root>
          <Button
            variant="default"
            size="default"
            className="gap-x-1 text-xs"
            onClick={() => {
              window.location.href = `/${model.id}/graph?generate=true`;
            }}
          >
            <PlusIcon className="h-4 w-4" />
            <span>Generate New Graph</span>
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
