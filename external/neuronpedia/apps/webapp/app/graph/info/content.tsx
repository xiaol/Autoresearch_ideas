'use client';

import { Button } from '@/components/shadcn/button';
import { EnvelopeClosedIcon } from '@radix-ui/react-icons';
import { ArrowUpRight } from 'lucide-react';
import { useRef } from 'react';
import Markdown from 'react-markdown';
import ScrollSpy from 'react-scrollspy-navigation';
import rehypeRaw from 'rehype-raw';
import rehypeSlug from 'rehype-slug';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { markdownData } from './markdown-data';
import { MarkdownPlotComponent } from './plots';

export function GraphInfoContent({ title, randomizedOrgs }: { title: string; randomizedOrgs: React.ReactNode[] }) {
  const scrollSpyParentRef = useRef<HTMLDivElement>(null);

  const onClickEach = (e: React.MouseEvent<HTMLAnchorElement>, next: () => void) => {
    const href = (e.target as HTMLAnchorElement).getAttribute('href');

    if (href && href.startsWith('#')) {
      const elementId = href.substring(1);
      const targetElement = document.getElementById(elementId);
      if (targetElement) {
        const targetTop = targetElement.getBoundingClientRect().top;
        if (elementId === 'section-the-landscape-of-interpretability-methods') {
          window.scrollTo({
            top: window.scrollY + targetTop - 850,
            behavior: 'smooth',
          });
        } else {
          window.scrollTo({
            top: window.scrollY + targetTop - 150,
            behavior: 'smooth',
          });
        }

        // update the url with the hash
        window.history.pushState({}, '', `#${elementId}`);
      }
    }
    // scroll e.target to visible
    const targetElement = e.target as HTMLElement;
    if (scrollSpyParentRef.current) {
      const targetRect = targetElement.getBoundingClientRect();
      const parentRect = scrollSpyParentRef.current.getBoundingClientRect();
      const scrollLeft =
        scrollSpyParentRef.current.scrollLeft +
        (targetRect.left - parentRect.left) -
        (parentRect.width - targetRect.width) / 2;
      const isScrolling = scrollSpyParentRef.current.style.scrollBehavior === 'smooth';
      if (!isScrolling) {
        scrollSpyParentRef.current.scrollTo({ left: scrollLeft, behavior: 'smooth' });
      }
    }
    next();
  };

  // we make our own ref because the 'prev' in the library is broken
  const lastActiveId = useRef<string | null>(null);

  const onChangeActiveId = (current: string, prev: string) => {
    if (current === prev || current === lastActiveId.current) {
      return;
    }
    lastActiveId.current = current;
    // Find and scroll to the current active element in the scrollspy navigation
    if (scrollSpyParentRef.current && current) {
      const activeElement = scrollSpyParentRef.current.querySelector(`a[href="#${current}"]`);
      if (activeElement) {
        const targetRect = activeElement.getBoundingClientRect();
        const parentRect = scrollSpyParentRef.current.getBoundingClientRect();
        const scrollLeft =
          scrollSpyParentRef.current.scrollLeft +
          (targetRect.left - parentRect.left) -
          (parentRect.width - targetRect.width) / 2;

        // Check if we're already scrolling to avoid interrupting smooth scroll
        const isScrolling = scrollSpyParentRef.current.style.scrollBehavior === 'smooth';
        if (!isScrolling) {
          scrollSpyParentRef.current.scrollTo({ left: scrollLeft, behavior: 'smooth' });
        }
      }
    }
  };

  return (
    <div className="flex h-full w-full flex-col items-center">
      <div className="sticky top-12 z-20 flex w-full flex-row items-center justify-between border-b border-t border-slate-200 bg-slate-50 pt-3 sm:pt-4">
        <div className="flex w-full flex-col gap-x-3 gap-y-1 leading-none sm:flex-row sm:items-center sm:justify-center">
          <div className="mt-0 w-full flex-col gap-y-1 text-center font-sans text-[13px] font-bold leading-snug text-slate-800 sm:flex sm:text-center sm:text-[18px] sm:leading-none">
            <button
              type="button"
              onClick={() => {
                window.scrollTo({ top: 0, behavior: 'smooth' });
                window.history.pushState({}, '', '#');
              }}
              className="whitespace-nowrap text-center text-[12px] text-sky-900 sm:text-[18px]"
            >
              {title}
            </button>
            <div className="mb-1 mt-1.5 flex h-[6px] max-h-[6px] min-h-[6px] flex-row items-end justify-center whitespace-nowrap px-3 text-[9px] font-medium text-slate-500 sm:mb-0 sm:text-[11px]">
              {randomizedOrgs.length > 0 && randomizedOrgs}
              <div className="ml-1 hidden sm:block"> · August 2025</div>
            </div>

            <ScrollSpy
              activeAttr
              onClickEach={onClickEach}
              onChangeActiveId={onChangeActiveId}
              threshold={Array.from({ length: 21 }, (_, i) => i * 0.05)}
            >
              <div
                ref={scrollSpyParentRef}
                className="mb-[-1px] mt-2 flex h-11 w-full max-w-full flex-row items-center justify-start gap-x-2 overflow-x-scroll px-3 sm:justify-center sm:gap-x-2 sm:px-0"
              >
                <a
                  href="#section-the-landscape-of-interpretability-methods"
                  className="relative flex h-11 min-w-[155px] max-w-[168px] select-none flex-col items-center justify-center gap-y-[3px] rounded-b-none rounded-t-md border border-b border-slate-200 bg-white px-3 py-2 text-[10px] font-medium leading-[1.325] text-sky-700 shadow-none transition-colors hover:bg-sky-50 data-[active=true]:border-b-sky-200 data-[active=true]:bg-sky-200 data-[active=true]:text-sky-800 sm:min-w-[170px] sm:max-w-[180px] sm:pl-6 sm:pr-3 sm:text-[11px]"
                >
                  <div className="pointer-events-none absolute left-0.5 top-0.5 flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-medium text-sky-700 sm:left-2 sm:top-auto sm:text-sm sm:font-bold">
                    1
                  </div>
                  The Landscape of Interpretability Methods
                </a>
                <a
                  href="#section-attribution-graphs-for-studying-model-biology"
                  className="relative flex h-11 min-w-[155px] max-w-[168px] select-none flex-col items-center justify-center gap-y-[3px] rounded-b-none rounded-t-md border border-b border-slate-200 bg-white px-3 py-2 text-[10px] font-medium leading-[1.325] text-sky-700 shadow-none transition-colors hover:bg-sky-50 data-[active=true]:border-b-sky-200 data-[active=true]:bg-sky-200 data-[active=true]:text-sky-800 sm:min-w-[170px] sm:max-w-[180px] sm:pl-6 sm:pr-3 sm:text-[11px]"
                >
                  <div className="pointer-events-none absolute left-0.5 top-0.5 flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-medium text-sky-700 sm:left-2 sm:top-auto sm:text-sm sm:font-bold">
                    2
                  </div>
                  Attribution Graphs for Studying Model Biology
                </a>
                <a
                  href="#section-transcoder-architecture-and-implementation"
                  className="relative flex h-11 min-w-[155px] max-w-[168px] select-none flex-col items-center justify-center gap-y-[3px] rounded-b-none rounded-t-md border border-b border-slate-200 bg-white px-3 py-2 text-[10px] font-medium leading-[1.325] text-sky-700 shadow-none transition-colors hover:bg-sky-50 data-[active=true]:border-b-sky-200 data-[active=true]:bg-sky-200 data-[active=true]:text-sky-800 sm:min-w-[170px] sm:max-w-[180px] sm:pl-6 sm:pr-3 sm:text-[11px]"
                >
                  <div className="pointer-events-none absolute left-0.5 top-0.5 flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-medium text-sky-700 sm:left-2 sm:top-auto sm:text-sm sm:font-bold">
                    3
                  </div>
                  Transcoder Architecture and Implementation
                </a>
                <a
                  href="#section-directions-for-future-work"
                  className="relative flex h-11 min-w-[125px] max-w-[138px] select-none flex-col items-center justify-center gap-y-[3px] rounded-b-none rounded-t-md border border-b border-slate-200 bg-white px-4 py-2 text-[10px] font-medium leading-[1.325] text-sky-700 shadow-none transition-all hover:bg-sky-50 data-[active=true]:border-b-sky-200 data-[active=true]:bg-sky-200 data-[active=true]:text-sky-800 sm:min-w-[130px] sm:max-w-[140px] sm:pl-7 sm:pr-5 sm:text-[11px]"
                >
                  <div className="pointer-events-none absolute left-0.5 top-0.5 flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-medium text-sky-700 sm:left-2 sm:top-auto sm:text-sm sm:font-bold">
                    4
                  </div>
                  Directions for Future Work
                </a>
                <a
                  href="#section-appendix"
                  className="relative flex h-11 min-w-[115px] max-w-[128px] select-none flex-col items-center justify-center gap-y-[3px] rounded-b-none rounded-t-md border border-b border-slate-200 bg-white px-3 py-2 text-[10px] font-medium leading-[1.325] text-sky-700 shadow-none transition-colors hover:bg-sky-50 data-[active=true]:border-b-sky-200 data-[active=true]:bg-sky-200 data-[active=true]:text-sky-800 sm:min-w-[130px] sm:max-w-[140px] sm:pl-5 sm:pr-3 sm:text-[11px]"
                >
                  <div className="pointer-events-none absolute left-0.5 top-0.5 flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-medium text-sky-700 sm:left-2 sm:top-auto sm:text-sm sm:font-bold">
                    5
                  </div>
                  Appendix
                </a>
                <a
                  href="https://neuronpedia.org/graph"
                  className="relative flex h-11 min-w-[170px] max-w-[180px] select-none flex-col items-center justify-center gap-y-[3px] rounded-b-none rounded-t-md border border-b border-emerald-600/30 border-b-sky-200 bg-emerald-50 py-2 pl-7 pr-1 text-[11px] font-bold leading-[1.325] text-emerald-700 shadow-none transition-colors hover:bg-emerald-100 data-[active=true]:border-b-sky-200 data-[active=true]:bg-sky-200 data-[active=true]:text-sky-800"
                >
                  <div className="pointer-events-none absolute left-5 flex h-4 w-4 items-center justify-center rounded-full text-sm font-bold text-sky-700">
                    🚀
                  </div>
                  <div className="flex flex-row items-center">
                    Explore Circuits <ArrowUpRight className="ml-1 h-3 w-3" />
                  </div>
                </a>
              </div>
            </ScrollSpy>
          </div>
        </div>
      </div>
      <div className="sticky top-[142px] z-20 -mb-[142px] h-4 min-h-4 w-full bg-gradient-to-t from-transparent to-sky-200 sm:top-[155px] sm:-mb-[155px]" />

      <div
        className="relative mb-10 flex w-full max-w-screen-lg flex-col items-center bg-white pb-20 pt-[140px] text-slate-800 sm:px-12 sm:pt-[160px]"
        id="group-post-markdown"
      >
        <div className="flex max-w-[900px] flex-row items-center justify-center gap-x-4 gap-y-4">
          <div className="flex w-full max-w-screen-lg flex-1 flex-col px-3 sm:px-0">
            <img
              src="https://neuronpedia.s3.us-east-1.amazonaws.com/site-assets/graph/post/circuits-landscape.jpg"
              title="The paths and locations are the edges and nodes of circuits. Landmarks: statue of La Liberté (Eleutheria) by Nanine Vallain, A decoding Antikythera mechanism, people camping around a bonfire, a sonnet and quill, and two people playing Go. Disregard the creature behind the mountain, I'm sure it's nothing."
              alt="An illustration of a map (landscape) set in medieval times, that has paths connecting a castle, mountains, houses, and a forest. The paths represent edges of a circuit, and the locations are the nodes. Other notable landmarks: a decoding Antikythera mechanism, La Liberté (Eleutheria) by Nanine Vallain, a group of people camping around a bonfire, a sonnet and quill, and two people playing Go. There is a dragon peering out of a mountain in the background."
              className="mb-1.5 mt-2 sm:z-10 sm:-ml-[5%] sm:mt-1 sm:min-w-[110%]"
            />

            <div className="mb-2 flex flex-col items-start justify-between gap-y-2 sm:flex-row sm:items-center sm:gap-x-2 sm:gap-y-0">
              <div className="flex flex-col">
                <div className="mb-0.5 mt-0 text-[11px] leading-relaxed text-slate-500 sm:mb-0.5 sm:text-[12px]">
                  Jack&nbsp;Lindsey*, Emmanuel&nbsp;Ameisen*, Neel&nbsp;Nanda*, Stepan&nbsp;Shabalin*,
                  Mateusz&nbsp;Piotrowski*, Tom&nbsp;McGrath*, Michael&nbsp;Hanna*, Owen&nbsp;Lewis*, Curt&nbsp;Tigges*,
                  Jack&nbsp;Merullo*, Connor&nbsp;Watts*, Gonçalo&nbsp;Paulo, Joshua&nbsp;Batson, Liv&nbsp;Gorton,
                  Elana&nbsp;Simon, Max&nbsp;Loeffler, Callum&nbsp;McDougall, Johnny&nbsp;Lin
                </div>
                <div className="mb-0 text-[9px] italic text-slate-400 sm:text-[10px]"> *Core Contributor</div>
              </div>
              <Button
                variant="outline"
                className="z-10 cursor-pointer border-sky-600 px-3 text-[11px] text-sky-700 hover:border-sky-400 hover:bg-sky-50 hover:text-sky-800"
                onClick={() => {
                  window.location.href = 'mailto:jacklindsey@anthropic.com';
                }}
              >
                <EnvelopeClosedIcon className="mr-1.5 h-3.5 w-3.5" /> Contact Authors
              </Button>
            </div>

            <Markdown
              rehypePlugins={[rehypeRaw, rehypeSlug]}
              components={{
                div: MarkdownPlotComponent,
              }}
              remarkPlugins={[remarkGfm, remarkMath]}
            >
              {markdownData}
            </Markdown>
          </div>
        </div>
      </div>
    </div>
  );
}
