'use client';

import dynamic from 'next/dynamic';
import { dataMSEvsPerTokenL0, dataRepVsMSE, dataRepVsPerTokenL0 } from './plots-data';

const isMobile = typeof window !== 'undefined' && window.innerWidth < 640;
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

// =================== Replacement Score vs Mean Per-Token L0 Sparsity ===================

const dataRepVsPerTokenL0Data = dataRepVsPerTokenL0.data.map((d) => ({
  ...d,
  hoverlabel: {
    bgcolor: '#e2e8f0',
    font: {
      color: '#334155',
      size: 12,
    },
    align: 'left',
    bordercolor: '#cbd5e1',
  },
  hovertemplate: d.x.map(
    (x, i) => `${d.name}<br><br>Mean Per-Token L0 Sparsity: ${x}<br>Replacement Score: ${d.y[i]}<extra></extra>`,
  ),
}));

const repVsPerTokenL0 = {
  ...dataRepVsPerTokenL0,
  data: dataRepVsPerTokenL0Data,
};

export const OVERALL_FINDINGS_PLOT_CLASSNAME = 'OVERALL_FINDINGS_PLOT';

export const OverallFindingsPlot = (
  <div className="mb-4 mt-3 flex flex-col rounded-xl border-slate-200 bg-slate-50 px-2 py-5 sm:px-8">
    <div className="text-center text-sm font-semibold text-slate-600">
      <a
        href="https://transformer-circuits.pub/2025/attribution-graphs/methods.html#evaluating-graphs-comparing"
        target="_blank"
        rel="noopener noreferrer"
        className="cursor-pointer font-semibold"
        title="The fraction of end-to-end graph paths, weighted by strength, that proceed from embedding nodes to logit nodes
      via feature nodes, rather than error nodes. Higher is better."
      >
        Replacement Score
      </a>{' '}
      vs Mean Per-Token L0 Sparsity (GPT2-Small)
    </div>

    <Plot
      data={repVsPerTokenL0.data as any}
      layout={{
        legend: {
          itemclick: false,
        },
        xaxis: {
          title: {
            text: repVsPerTokenL0.xaxis_title,
            standoff: 20,
            font: {
              size: 11,
              color: '#475569',
            },
          },
          gridcolor: '#cbd5e1',
          fixedrange: true,
          tickfont: {
            color: '#475569',
            size: 10,
          },
        },
        yaxis: {
          title: {
            text: repVsPerTokenL0.yaxis_title,
            standoff: 20,
            font: {
              size: 11,
              color: '#475569',
            },
          },
          gridcolor: '#cbd5e1',
          fixedrange: true,
          tickfont: {
            color: '#475569',
            size: 10,
          },
        },
        barmode: 'relative',
        bargap: 0.05,
        showlegend: !isMobile,
        margin: {
          l: 60,
          r: 0,
          b: 60,
          t: 15,
          pad: 3,
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
      }}
      //   onClick={(event) => {
      //     if (!isMobile) {
      //       const point = event.points[0];
      //       const url = point.customdata as string;
      //       if (url) window.open(url, '_blank');
      //     }
      //   }}
      config={{
        responsive: false,
        displayModeBar: false,
        editable: false,
        scrollZoom: false,
      }}
      className="w-full"
    />
    <div className="text-xs italic leading-relaxed text-slate-500">
      We train various transcoder architectures on GPT-2 for 100M tokens, using roughly parameter-matched configurations
      (2<sup>17</sup> width for transcoders, 2<sup>15</sup> for CLTs, 3*2<sup>15</sup> for incremental CLTs).
      Replacement scores are evaluated on 260 curated prompts. To address evaluation artifacts from GPT-2&apos;s high
      first-token norm, we prepend an EOS token to each prompt and exclude it from attribution (this does not affect
      GPT-2 loss). Different architectural choices result in distinct sparsity-performance trade-offs, with CLTs
      generally outperforming PLTs and other variants.
    </div>
  </div>
);

// =================== Replacement Score vs Normalized MSE ===================

const dataRepVsMSEData = dataRepVsMSE.data.map((d) => ({
  ...d,
  hoverlabel: {
    bgcolor: '#e2e8f0',
    font: {
      color: '#334155',
      size: 12,
    },
    align: 'left',
    bordercolor: '#cbd5e1',
  },
  hovertemplate: d.x.map(
    (x, i) => `${d.name}<br><br>Mean Normalized MSE: ${x}<br>Replacement Score: ${d.y[i]}<extra></extra>`,
  ),
}));

const repVsMSE = {
  ...dataRepVsMSE,
  data: dataRepVsMSEData,
};

export const MSE_VS_REPLACEMENT_SCORE_PLOT_CLASSNAME = 'MSE_VS_REPLACEMENT_SCORE_PLOT';

export const MSEvsReplacementScorePlot = (
  <div className="mb-4 mt-3 flex flex-col rounded-xl border-slate-200 bg-slate-50 px-2 py-5 sm:px-8">
    <div className="text-center text-sm font-semibold text-slate-600">
      <a
        href="https://transformer-circuits.pub/2025/attribution-graphs/methods.html#evaluating-graphs-comparing"
        target="_blank"
        rel="noopener noreferrer"
        className="cursor-pointer font-semibold"
        title="The fraction of end-to-end graph paths, weighted by strength, that proceed from embedding nodes to logit nodes
      via feature nodes, rather than error nodes. Higher is better."
      >
        Replacement Score
      </a>{' '}
      vs Mean Normalized MSE (GPT2-Small)
    </div>

    <Plot
      data={repVsMSE.data as any}
      layout={{
        legend: {
          itemclick: false,
        },
        xaxis: {
          title: {
            text: repVsMSE.xaxis_title,
            standoff: 20,
            font: {
              size: 11,
              color: '#475569',
            },
          },
          gridcolor: '#cbd5e1',
          fixedrange: true,
          tickfont: {
            color: '#475569',
            size: 10,
          },
        },
        yaxis: {
          title: {
            text: repVsMSE.yaxis_title,
            standoff: 20,
            font: {
              size: 11,
              color: '#475569',
            },
          },
          gridcolor: '#cbd5e1',
          fixedrange: true,
          tickfont: {
            color: '#475569',
            size: 10,
          },
        },
        barmode: 'relative',
        bargap: 0.05,
        showlegend: !isMobile,
        margin: {
          l: 60,
          r: 0,
          b: 60,
          t: 15,
          pad: 3,
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
      }}
      //   onClick={(event) => {
      //     if (!isMobile) {
      //       const point = event.points[0];
      //       const url = point.customdata as string;
      //       if (url) window.open(url, '_blank');
      //     }
      //   }}
      config={{
        responsive: false,
        displayModeBar: false,
        editable: false,
        scrollZoom: false,
      }}
      className="w-full"
    />
    <div className="text-xs italic leading-relaxed text-slate-500">
      In general we find these are highly correlated for a given architecture, but the relationship varies between
      architectures. Notably, the skip-versions of each architecture have a lower intercept than the regular versions.
    </div>
  </div>
);

// =================== MSE vs Per-Token L0 Sparsity ===================

const dataMSEvsPerTokenL0Data = dataMSEvsPerTokenL0.data.map((d) => ({
  ...d,
  hoverlabel: {
    bgcolor: '#e2e8f0',
    font: {
      color: '#334155',
      size: 12,
    },
    align: 'left',
    bordercolor: '#cbd5e1',
  },
  hovertemplate: d.x.map(
    (x, i) => `${d.name}<br><br>Mean Normalized MSE: ${x}<br>Mean Per-Token L0 Sparsity: ${d.y[i]}<extra></extra>`,
  ),
}));

const mseVsPerTokenL0 = {
  ...dataMSEvsPerTokenL0,
  data: dataMSEvsPerTokenL0Data,
};

export const MSE_VS_PER_TOKEN_L0_SPARSITY_PLOT_CLASSNAME = 'MSE_VS_PER_TOKEN_L0_SPARSITY_PLOT';

export const MSEvsPerTokenL0SparsityPlot = (
  <div className="mb-4 mt-3 flex flex-col rounded-xl border-slate-200 bg-slate-50 px-2 py-5 sm:px-8">
    <div className="text-center text-sm font-semibold text-slate-600">
      Mean Normalized MSE vs Mean Per-Token L0 Sparsity (GPT2-Small)
    </div>

    <Plot
      data={mseVsPerTokenL0.data as any}
      layout={{
        legend: {
          itemclick: false,
        },
        xaxis: {
          title: {
            text: mseVsPerTokenL0.xaxis_title,
            standoff: 20,
            font: {
              size: 11,
              color: '#475569',
            },
          },
          gridcolor: '#cbd5e1',
          fixedrange: true,
          tickfont: {
            color: '#475569',
            size: 10,
          },
        },
        yaxis: {
          title: {
            text: mseVsPerTokenL0.yaxis_title,
            standoff: 20,
            font: {
              size: 11,
              color: '#475569',
            },
          },
          gridcolor: '#cbd5e1',
          fixedrange: true,
          tickfont: {
            color: '#475569',
            size: 10,
          },
        },
        barmode: 'relative',
        bargap: 0.05,
        showlegend: !isMobile,
        margin: {
          l: 60,
          r: 0,
          b: 60,
          t: 15,
          pad: 3,
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
      }}
      //   onClick={(event) => {
      //     if (!isMobile) {
      //       const point = event.points[0];
      //       const url = point.customdata as string;
      //       if (url) window.open(url, '_blank');
      //     }
      //   }}
      config={{
        responsive: false,
        displayModeBar: false,
        editable: false,
        scrollZoom: false,
      }}
      className="w-full"
    />
  </div>
);

const createPlotComponent = () =>
  function PlotComponent({ children, className }: { className?: string; children?: React.ReactNode }) {
    if (className === OVERALL_FINDINGS_PLOT_CLASSNAME) {
      return OverallFindingsPlot;
    }
    if (className === MSE_VS_REPLACEMENT_SCORE_PLOT_CLASSNAME) {
      return MSEvsReplacementScorePlot;
    }
    if (className === MSE_VS_PER_TOKEN_L0_SPARSITY_PLOT_CLASSNAME) {
      return MSEvsPerTokenL0SparsityPlot;
    }

    return <div className={className}>{children}</div>;
  };

export const MarkdownPlotComponent = createPlotComponent();
