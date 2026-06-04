<script lang="ts">
	import {
		tokens,
		modelMeta,
		headContentHeight,
		headGap,
		attentionHeadIdx,
		blockIdx,
		modelData,
		rootRem,
		weightPopover,
		rwkvStateTokenIdx,
		rwkvStateViewMode
	} from '~/store';
	import classNames from 'classnames';
	import HeadStack from './HeadStack.svelte';
	import Matrix from './common/Matrix.svelte';
	import { Tooltip } from 'flowbite-svelte';
	import TextbookTooltip from './common/TextbookTooltip.svelte';
	import * as d3 from 'd3';
	import resolveConfig from 'tailwindcss/resolveConfig';
	import tailwindConfig from '../../tailwind.config';
	import Katex from '~/utils/Katex.svelte';
	import { ExpandOutline } from 'flowbite-svelte-icons';

	export let className: string | undefined = undefined;

	const { theme } = resolveConfig(tailwindConfig);
	const decayColor = 'bg-rose-200';
	const eraseColor = 'bg-red-200';
	const writeColor = 'bg-emerald-200';
	const outputColor = 'bg-purple-300';
	type RwkvTransitionKey = keyof NonNullable<RwkvBlockTrace['stateTransition']>;
	let hoveredTokenIndex: number | null = null;

	$: blockTrace = $modelData?.rwkvBlocks?.[$blockIdx];
	const headSeries = (
		trace: RwkvBlockTrace | undefined,
		key: keyof RwkvBlockTrace['timeMix'],
		headIndex: number
	) => (trace?.timeMix?.[key] || []).map((row) => row[headIndex] ?? 0);

	$: stateRows = blockTrace?.state?.norms || [];
	$: deltaRows = blockTrace?.state?.deltas || [];
	$: rmsRows = blockTrace?.state?.rms || [];
	$: rankRows = blockTrace?.state?.stableRanks || [];
	$: matrixData = stateRows.length
		? normalizeRows(stateRows)
		: placeholderMatrix($tokens.length, $modelMeta.attention_head_num);
	$: deltaData = deltaRows.length
		? normalizeRows(deltaRows)
		: placeholderMatrix($tokens.length, $modelMeta.attention_head_num);
	$: rmsData = rmsRows.length
		? normalizeRows(rmsRows)
		: placeholderMatrix($tokens.length, $modelMeta.attention_head_num);
	$: rankData = rankRows.length
		? normalizeRows(rankRows)
		: placeholderMatrix($tokens.length, $modelMeta.attention_head_num);
	$: selectedDecay = headSeries(blockTrace, 'decayedStateNorms', $attentionHeadIdx);
	$: selectedErase = headSeries(blockTrace, 'eraseNorms', $attentionHeadIdx);
	$: selectedWrite = headSeries(blockTrace, 'writeNorms', $attentionHeadIdx);
	$: selectedBefore = headSeries(blockTrace, 'stateBeforeNorms', $attentionHeadIdx);
	$: selectedAfter = headSeries(blockTrace, 'stateAfterNorms', $attentionHeadIdx);
	$: selectedRead = headSeries(blockTrace, 'readNorms', $attentionHeadIdx);
	$: selectedBonus = headSeries(blockTrace, 'bonusNorms', $attentionHeadIdx);
	$: selectedOutput = headSeries(blockTrace, 'outputNorms', $attentionHeadIdx);
	$: activeTokenIndex = clamp(
		hoveredTokenIndex ?? $rwkvStateTokenIdx ?? Math.max($tokens.length - 1, 0),
		0,
		Math.max($tokens.length - 1, 0)
	);
	$: activeToken = $tokens[activeTokenIndex] || '';
	$: transitionRawMaps = [
		transitionMatrix(
			blockTrace,
			'stateBefore',
			activeTokenIndex,
			$attentionHeadIdx,
			valueAt(selectedBefore, activeTokenIndex)
		),
		transitionMatrix(
			blockTrace,
			'decayed',
			activeTokenIndex,
			$attentionHeadIdx,
			valueAt(selectedDecay, activeTokenIndex)
		),
		transitionMatrix(
			blockTrace,
			'erase',
			activeTokenIndex,
			$attentionHeadIdx,
			valueAt(selectedErase, activeTokenIndex)
		),
		transitionMatrix(
			blockTrace,
			'write',
			activeTokenIndex,
			$attentionHeadIdx,
			valueAt(selectedWrite, activeTokenIndex)
		),
		transitionMatrix(
			blockTrace,
			'stateAfter',
			activeTokenIndex,
			$attentionHeadIdx,
			valueAt(selectedAfter, activeTokenIndex)
		)
	];
	$: transitionMaps = normalizedTransitionMaps(transitionRawMaps);
	$: transitionSignalMax = Math.max(
		...[
			valueAt(selectedBefore, activeTokenIndex),
			valueAt(selectedDecay, activeTokenIndex),
			valueAt(selectedErase, activeTokenIndex),
			valueAt(selectedWrite, activeTokenIndex),
			valueAt(selectedAfter, activeTokenIndex)
		].map((value) => Math.abs(value)),
		1e-6
	);
	$: calculationTerms = [
		{
			label: 'S_prev',
			shape: '64 x 64',
			data: transitionMaps[0],
			color: 'sky',
			value: valueAt(selectedBefore, activeTokenIndex),
			signal: signalFill(valueAt(selectedBefore, activeTokenIndex)),
			operator: 'decay'
		},
		{
			label: 'S*w',
			shape: '64 x 64',
			data: transitionMaps[1],
			color: 'rose',
			value: valueAt(selectedDecay, activeTokenIndex),
			signal: signalFill(valueAt(selectedDecay, activeTokenIndex)),
			operator: '-'
		},
		{
			label: 'erase',
			shape: '64 x 64',
			data: transitionMaps[2],
			color: 'red',
			value: valueAt(selectedErase, activeTokenIndex),
			signal: signalFill(valueAt(selectedErase, activeTokenIndex)),
			operator: '+'
		},
		{
			label: 'write',
			shape: '64 x 64',
			data: transitionMaps[3],
			color: 'emerald',
			value: valueAt(selectedWrite, activeTokenIndex),
			signal: signalFill(valueAt(selectedWrite, activeTokenIndex)),
			operator: '='
		},
		{
			label: 'S_t',
			shape: '64 x 64',
			data: transitionMaps[4],
			color: 'purple',
			value: valueAt(selectedAfter, activeTokenIndex),
			signal: signalFill(valueAt(selectedAfter, activeTokenIndex)),
			operator: null
		}
	];
	$: transitionMetrics = [
		{ label: 'S_prev', value: valueAt(selectedBefore, activeTokenIndex), color: 'bg-sky-200' },
		{ label: 'S*w', value: valueAt(selectedDecay, activeTokenIndex), color: decayColor },
		{ label: 'erase', value: valueAt(selectedErase, activeTokenIndex), color: eraseColor },
		{ label: 'write', value: valueAt(selectedWrite, activeTokenIndex), color: writeColor },
		{ label: 'S_t', value: valueAt(selectedAfter, activeTokenIndex), color: 'bg-purple-200' },
		{ label: 'read', value: valueAt(selectedRead, activeTokenIndex), color: 'bg-violet-200' },
		{ label: 'bonus', value: valueAt(selectedBonus, activeTokenIndex), color: 'bg-amber-200' },
		{ label: 'out', value: valueAt(selectedOutput, activeTokenIndex), color: outputColor }
	];
	$: cellSize = Math.min(18, Math.max((1 / Math.max($tokens.length, 1)) * rootRem * 5, 8));

	const placeholderMatrix = (tokenCount: number, headCount: number) =>
		Array.from({ length: tokenCount }, () => Array.from({ length: headCount }, () => 0));

	const normalizeRows = (rows: number[][]) => {
		const maxValue = Math.max(...rows.flat(), 1e-6);
		return rows.map((row) => row.map((value) => value / maxValue));
	};

	const scaleValue = (values: number[], index: number) => {
		const maxValue = Math.max(...values, 1e-6);
		return Math.max(0.05, Math.min(1, (values[index] ?? 0) / maxValue));
	};

	const valueAt = (values: number[], index: number) => values[index] ?? 0;
	const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));
	const emptyMap = () => Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => 0));
	const normFallbackMap = (value: number) => {
		if (!Number.isFinite(value) || value <= 0) return emptyMap();
		return [
			[value, value * 0.45, value * 0.2, value * 0.1],
			[value * 0.5, value * 0.3, value * 0.14, value * 0.07],
			[value * 0.24, value * 0.16, value * 0.1, value * 0.05],
			[value * 0.12, value * 0.08, value * 0.05, value * 0.03]
		];
	};
	const transitionMatrix = (
		trace: RwkvBlockTrace | undefined,
		key: RwkvTransitionKey,
		tokenIndex: number,
		headIndex: number,
		fallback: number
	) => trace?.stateTransition?.[key]?.[tokenIndex]?.[headIndex] || normFallbackMap(fallback);
	const normalizedTransitionMaps = (matrices: number[][][]) => {
		return matrices.map((matrix) => {
			const maxValue = Math.max(...matrix.flat().map((value) => Math.abs(value)), 1e-6);
			return matrix.map((row) => row.map((value) => Math.abs(value) / maxValue));
		});
	};
	const formatMetric = (value: number) => {
		if (!Number.isFinite(value)) return '0';
		if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(1);
		if (Math.abs(value) >= 100) return value.toFixed(0);
		if (Math.abs(value) >= 10) return value.toFixed(1);
		if (Math.abs(value) >= 1) return value.toFixed(2);
		return value.toFixed(3);
	};

	const colorScale = (value: number) => d3.interpolate('white', theme.colors.purple[700])(value);
	const calcColorScale = (color: string) => (value: number) =>
		d3.interpolate('white', theme.colors[color][600])(value);
	const signalFill = (value: number) => {
		const scaled = Math.abs(value) / transitionSignalMax;
		if (scaled <= 0) return 0;
		return Math.max(0.08, Math.min(1, Math.sqrt(scaled)));
	};
	const selectToken = (index: number) => {
		rwkvStateTokenIdx.set($rwkvStateTokenIdx === index ? null : index);
	};
	const openStateCalculation = (event: MouseEvent, viewMode: 'matrix' | 'manifold' = 'matrix') => {
		event.stopPropagation();
		rwkvStateViewMode.set(viewMode);
		weightPopover.set('rwkvState');
	};
</script>

<div class={classNames('attention', 'rwkv-state', className)} data-click="rwkv-state-step">
	<div class="title" role="group" data-click="rwkv-state-title">
		<div class="w-max">
			<TextbookTooltip id="rwkv-state">Recurrent State</TextbookTooltip>
		</div>
	</div>
	<div class="content relative">
		<div
			class="bounding attention-bounding"
			style={`padding-bottom:${$modelMeta.attention_head_num * headGap.y}px`}
		></div>
		<div class="state-rule">
			<div class="rule-main">
				<Katex
					math={'S_t = S_{t-1}\\operatorname{diag}(w_t) - (S_{t-1}\\hat{\\kappa}_t)(a_t\\odot\\hat{\\kappa}_t)^\\top + v_t\\tilde{k}_t^\\top'}
				/>
			</div>
			<div class="rule-sub">
				<Katex
					math={'y_t = r_t S_t + ((r_t\\odot\\tilde{k}_t\\odot r_k)\\mathbf{1})v_t'}
				/>
			</div>
			<div class="transition-readout">
				<span class="context">State Head {$attentionHeadIdx + 1}, token {activeTokenIndex + 1} {activeToken}</span>
				{#each transitionMetrics as metric}
					<span class="metric-chip" title={`${metric.label}: ${formatMetric(metric.value)}`}>
						<span class={`swatch ${metric.color}`}></span>
						<span>{metric.label}</span>
					</span>
				{/each}
			</div>
		</div>
		<div class="heads">
			<HeadStack>
				<div class="head-block flex w-full items-center justify-between px-2" style={`height:${$headContentHeight}px;`}>
					<div class="qkv flex h-full flex-col justify-center gap-[3.5rem] pl-[6rem]">
						<div class="column key">
							<div class="head1 title"><TextbookTooltip id="rwkv-state">S*w</TextbookTooltip></div>
							{#each $tokens as token, index}
								<div
									class="head1 key cell x1-12 text-xs"
									class:active-token={activeTokenIndex === index}
									on:mouseenter={() => {
										hoveredTokenIndex = index;
									}}
									on:mouseleave={() => {
										hoveredTokenIndex = null;
									}}
									on:click={() => selectToken(index)}
									role="button"
									tabindex="0"
									on:keydown={(event) => {
										if (event.key === 'Enter' || event.key === ' ') selectToken(index);
									}}
								>
									<span class="label float">{token}</span>
									<div class={`vector x1-12 ${decayColor}`}>
										<div class="metric" style={`--fill:${scaleValue(selectedDecay, index)}`}></div>
									</div>
								</div>
							{/each}
						</div>
						<div class="column query">
							<div class="head1 title"><TextbookTooltip id="rwkv-state">Erase</TextbookTooltip></div>
							{#each $tokens as token, index}
								<div
									class="head1 cell x1-12 query text-xs"
									class:active-token={activeTokenIndex === index}
									on:mouseenter={() => {
										hoveredTokenIndex = index;
									}}
									on:mouseleave={() => {
										hoveredTokenIndex = null;
									}}
									on:click={() => selectToken(index)}
									role="button"
									tabindex="0"
									on:keydown={(event) => {
										if (event.key === 'Enter' || event.key === ' ') selectToken(index);
									}}
								>
									<span class="label float">{token}</span>
									<div class={`vector x1-12 ${eraseColor}`}>
										<div class="metric" style={`--fill:${scaleValue(selectedErase, index)}`}></div>
									</div>
								</div>
							{/each}
						</div>
						<div class="column value">
							<div class="head1 title"><TextbookTooltip id="rwkv-state">Write</TextbookTooltip></div>
							{#each $tokens as token, index}
								<div
									class="head1 cell x1-12 text-xs"
									class:last={index === $tokens.length - 1}
									class:active-token={activeTokenIndex === index}
									on:mouseenter={() => {
										hoveredTokenIndex = index;
									}}
									on:mouseleave={() => {
										hoveredTokenIndex = null;
									}}
									on:click={() => selectToken(index)}
									role="button"
									tabindex="0"
									on:keydown={(event) => {
										if (event.key === 'Enter' || event.key === ' ') selectToken(index);
									}}
								>
									<span class="label float">{token}</span>
									<div class={`vector x1-12 ${writeColor}`}>
										<div class="metric" style={`--fill:${scaleValue(selectedWrite, index)}`}></div>
									</div>
								</div>
							{/each}
						</div>
					</div>
					<div class="resize-watch attention-matrix flex">
						<div class="attention-matrix-container relative flex flex-col">
							<div class="state-view-actions" aria-label="Open RWKV state view">
								<button
									type="button"
									class:active={$rwkvStateViewMode === 'matrix'}
									title="Open matrix view"
									aria-label="Open matrix view"
									on:click={(event) => openStateCalculation(event, 'matrix')}
								>
									Matrix
								</button>
								<button
									type="button"
									class="manifold"
									class:active={$rwkvStateViewMode === 'manifold'}
									title="Open manifold view"
									aria-label="Open manifold view"
									on:click={(event) => openStateCalculation(event, 'manifold')}
								>
									Manifold
								</button>
							</div>
							<div class="token-picker" aria-label="Select RWKV state-transition token">
								{#each $tokens as token, index}
									<button
										type="button"
										class:active={activeTokenIndex === index}
										on:mouseenter={() => {
											hoveredTokenIndex = index;
										}}
										on:mouseleave={() => {
											hoveredTokenIndex = null;
										}}
										on:click={() => selectToken(index)}
										title={`Show state transition for token ${index + 1}: ${token}`}
									>
										<span class="token-index">{index + 1}</span>
										<span class="token-text">{token}</span>
									</button>
								{/each}
								<button
									type="button"
									class="open-calc"
									title="Open state calculation"
									aria-label="Open state calculation"
									on:click={(event) => openStateCalculation(event, $rwkvStateViewMode)}
								>
									<ExpandOutline class="h-3 w-3" />
								</button>
							</div>
							<div class="state-calculation">
								{#each calculationTerms as term}
									<div
										class="calc-term"
										style={`--signal:${term.signal}`}
										title={`${term.label}: ${formatMetric(term.value)} · ${term.shape}`}
									>
										<div class="calc-label">{term.label}</div>
										<Matrix
											className="calc-matrix"
											data={term.data}
											showSize={false}
											cellHeight={5}
											cellWidth={5}
											rowGap={1}
											colGap={1}
											colorScale={calcColorScale(term.color)}
										/>
										<div class={`signal-bar ${term.color}`}><span></span></div>
									</div>
									{#if term.operator}
										<div class="calc-operator">{term.operator}</div>
									{/if}
								{/each}
							</div>
							<div class="state-summary-row">
								<div class="attention-matrix attention-result attention-initial attention-out flex flex-col items-center">
									<Matrix
										className="main"
										data={matrixData}
										showSize={false}
										cellHeight={cellSize}
										cellWidth={cellSize}
										rowGap={3}
										colGap={2}
										shape="circle"
										{colorScale}
									/>
									<div class="matrix-label">State S</div>
								</div>
								<div class="attention-matrix state-delta flex flex-col items-center">
									<Matrix
										className="main"
										data={deltaData}
										showSize={false}
										cellHeight={cellSize}
										cellWidth={cellSize}
										rowGap={3}
										colGap={2}
										shape="circle"
										{colorScale}
									/>
									<div class="matrix-label">Delta S</div>
								</div>
								<div class="attention-matrix state-rms flex flex-col items-center">
									<Matrix
										className="main"
										data={rmsData}
										showSize={false}
										cellHeight={cellSize}
										cellWidth={cellSize}
										rowGap={3}
										colGap={2}
										shape="circle"
										{colorScale}
									/>
									<div class="matrix-label">RMS</div>
								</div>
								<div class="attention-matrix state-rank flex flex-col items-center">
									<Matrix
										className="main"
										data={rankData}
										showSize={false}
										cellHeight={cellSize}
										cellWidth={cellSize}
										rowGap={3}
										colGap={2}
										shape="circle"
										{colorScale}
									/>
									<div class="matrix-label">Rank</div>
								</div>
							</div>
						</div>
					</div>
					<div class="head-out mx-[2rem]">
						<div class="column out">
							<div class="head1 title">Out</div>
							{#each $tokens as token, index}
								<div
									class="head1 cell x1-12"
									class:last={index === $tokens.length - 1}
									class:active-token={activeTokenIndex === index}
									on:mouseenter={() => {
										hoveredTokenIndex = index;
									}}
									on:mouseleave={() => {
										hoveredTokenIndex = null;
									}}
									on:click={() => selectToken(index)}
									role="button"
									tabindex="0"
									on:keydown={(event) => {
										if (event.key === 'Enter' || event.key === ' ') selectToken(index);
									}}
								>
									<div class={`vector x1-12 ${outputColor}`}>
										<div class="metric" style={`--fill:${scaleValue(selectedOutput, index)}`}></div>
									</div>
								</div>
							{/each}
						</div>
					</div>
				</div>
			</HeadStack>
		</div>
		<Tooltip class="popover" triggeredBy={'.rwkv-state .attention-result'} placement="right"
			>State-head recurrent state norm, state delta, RMS, and stable rank by token</Tooltip
		>
		<Tooltip class="popover" triggeredBy={'.step.rwkv-state .key .cell'} placement="right"
			>Decayed previous state term S*w for State Head {$attentionHeadIdx + 1}</Tooltip
		>
		<Tooltip class="popover" triggeredBy={'.step.rwkv-state .query .cell'} placement="right"
			>Erase term from the normalized removal key and in-context learning gate for State Head {$attentionHeadIdx + 1}</Tooltip
		>
		<Tooltip class="popover" triggeredBy={'.step.rwkv-state .value .cell'} placement="right"
			>Write term v outer k~ for State Head {$attentionHeadIdx + 1}</Tooltip
		>
	</div>
</div>

<style lang="scss">
	.rwkv-state {
		.attention-bounding {
			top: -0.5rem;
			padding: 0.5rem 0;
			left: -0.3rem;
			width: calc(100% + 1rem);
			height: calc(100%);
		}
		.content {
			display: grid;
			grid-template-columns: auto 0;
		}
		.state-rule {
			position: absolute;
			left: 7rem;
			right: 4rem;
			top: -3.6rem;
			z-index: $COLUMN_TITLE_INDEX;
			display: grid;
			gap: 0.22rem;
			color: theme('colors.gray.500');
			font-size: 0.68rem;
			line-height: 1.15;
			pointer-events: none;
		}
		.rule-main {
			color: theme('colors.gray.700');
			font-size: 0.76rem;
			font-weight: 600;
		}
		.rule-sub {
			display: flex;
			flex-wrap: wrap;
			gap: 0.35rem;
			align-items: center;
		}
		.transition-readout {
			display: flex;
			flex-wrap: wrap;
			gap: 0.25rem 0.45rem;
			align-items: center;
		}
		.context {
			color: theme('colors.gray.600');
			font-weight: 600;
			max-width: 13rem;
			overflow: hidden;
			text-overflow: ellipsis;
			white-space: nowrap;
		}
		.metric-chip {
			display: inline-flex;
			align-items: center;
			gap: 0.18rem;
			white-space: nowrap;
			color: theme('colors.gray.500');

		}
		.swatch {
			width: 0.45rem;
			height: 0.45rem;
			border: 1px solid rgba(0, 0, 0, 0.08);
		}
		.heads {
			padding: 0 7rem 0 8rem;
		}
		.column {
			.label {
				font-size: 0.7rem;
				color: theme('colors.gray.600');
			}
			.title {
				z-index: $COLUMN_TITLE_INDEX;
				position: absolute;
				top: -1.7rem;
				left: 50%;
				transform: translateX(-50%);
				font-size: 0.85rem;
				transition: none;
				color: theme('colors.gray.500');
			}
		}
		.head1.cell {
			justify-content: center;
			width: 0;

			&.active-token {
				.vector {
					outline: 2px solid theme('colors.gray.500');
					outline-offset: 1px;
				}
				.label {
					color: theme('colors.gray.900');
					font-weight: 600;
				}
			}
		}
		.head1.cell[role='button'] {
			cursor: pointer;
		}
		.metric {
			position: absolute;
			inset: auto 0 0 0;
			height: calc(var(--fill) * 100%);
			background: rgba(255, 255, 255, 0.5);
		}
		.attention-matrix-container {
			gap: 0.62rem;
			padding: 0.85rem 1rem 1rem 1rem;
		}
		.state-view-actions {
			position: relative;
			z-index: $COLUMN_TITLE_INDEX;
			display: flex;
			justify-content: center;
			gap: 0.35rem;
			padding: 0 0.35rem;
			pointer-events: auto;

			button {
				display: inline-flex;
				align-items: center;
				justify-content: center;
				min-width: 4.2rem;
				border: 1px solid theme('colors.gray.200');
				background: white;
				color: theme('colors.gray.700');
				font-size: 0.72rem;
				font-weight: 700;
				line-height: 1;
				padding: 0.28rem 0.55rem;
				cursor: pointer;
				box-shadow: 0 1px 2px rgb(0 0 0 / 0.06);

				&.active {
					border-color: theme('colors.sky.300');
					background: theme('colors.sky.50');
					color: theme('colors.sky.700');
				}

				&.manifold.active {
					border-color: theme('colors.emerald.300');
					background: theme('colors.emerald.50');
					color: theme('colors.emerald.700');
				}
			}
		}
		.state-calculation,
		.state-summary-row {
			display: flex;
			align-items: center;
			justify-content: center;
			gap: 0.45rem;
		}
		.token-picker {
			display: flex;
			justify-content: center;
			gap: 0.25rem;
			padding: 0 0.35rem 0.2rem 0.35rem;
			pointer-events: auto;

			button {
				display: inline-flex;
				align-items: center;
				max-width: 4.6rem;
				gap: 0.18rem;
				border: 1px solid theme('colors.gray.200');
				background: white;
				color: theme('colors.gray.500');
				font-size: 0.58rem;
				line-height: 1;
				padding: 0.14rem 0.22rem;
				cursor: pointer;

				&.active {
					border-color: theme('colors.purple.300');
					color: theme('colors.purple.700');
					background: theme('colors.purple.50');
				}

				&.open-calc {
					padding: 0.14rem;
					color: theme('colors.gray.500');
				}
			}

			.token-index {
				font-weight: 700;
				color: theme('colors.gray.400');
			}

			.token-text {
				overflow: hidden;
				text-overflow: ellipsis;
				white-space: nowrap;
			}
		}
		.state-calculation {
			padding: 0.15rem 0.4rem 0.45rem 0.4rem;
			border-bottom: 1px solid theme('colors.gray.100');
		}
		.calc-term {
			display: grid;
			justify-items: center;
			gap: 0.16rem;
			font-size: 0.56rem;
			line-height: 1;
			color: theme('colors.gray.500');
		}
		.calc-label {
			font-size: 0.66rem;
			font-weight: 700;
			color: theme('colors.gray.600');
			white-space: nowrap;
		}
		.signal-bar {
			width: 1.65rem;
			height: 0.22rem;
			overflow: hidden;
			background: theme('colors.gray.100');
			border: 1px solid theme('colors.gray.200');

			span {
				display: block;
				width: calc(var(--signal) * 100%);
				height: 100%;
				background: currentColor;
			}
			&.sky {
				color: theme('colors.sky.500');
			}
			&.rose {
				color: theme('colors.rose.500');
			}
			&.red {
				color: theme('colors.red.500');
			}
			&.emerald {
				color: theme('colors.emerald.500');
			}
			&.purple {
				color: theme('colors.purple.500');
			}
		}
		.calc-operator {
			align-self: center;
			margin-top: -0.8rem;
			min-width: 1rem;
			text-align: center;
			font-weight: 700;
			color: theme('colors.gray.500');
		}
		.matrix-label {
			white-space: nowrap;
			color: theme('colors.gray.400');
		}
	}
</style>
