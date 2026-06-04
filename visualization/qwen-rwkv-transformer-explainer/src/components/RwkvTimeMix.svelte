<script lang="ts">
	import { tokens, modelMeta, attentionHeadIdx, vectorHeight, blockIdx, modelData } from '~/store';
	import classNames from 'classnames';
	import { onMount } from 'svelte';
	import VectorCanvas from './common/VectorCanvas.svelte';
	import OperationGroup from './OperationGroup.svelte';
	import { Tooltip } from 'flowbite-svelte';
	import TextbookTooltip from './common/TextbookTooltip.svelte';

	export let className: string | undefined = undefined;

	const inputColor = 'bg-gray-300';
	const streams = [
		{
			key: 'receptanceNorms',
			short: 'r',
			label: 'read receptance',
			className: 'query',
			color: 'bg-sky-200'
		},
		{
			key: 'decayMeans',
			short: 'w',
			label: 'state decay',
			className: 'key',
			color: 'bg-rose-200'
		},
		{
			key: 'replacementKeyNorms',
			short: 'k~',
			label: 'replacement/write key',
			className: 'replacement',
			color: 'bg-emerald-200'
		},
		{
			key: 'removalKeyNorms',
			short: 'k-',
			label: 'removal key before normalization',
			className: 'removal',
			color: 'bg-red-200'
		},
		{
			key: 'valueNorms',
			short: 'v',
			label: 'write value',
			className: 'value',
			color: 'bg-lime-200'
		},
		{
			key: 'valueResidualGateMeans',
			short: 'vr',
			label: 'value residual gate',
			className: 'value-residual',
			color: 'bg-cyan-200'
		},
		{
			key: 'writeGateMeans',
			short: 'a',
			label: 'in-context learning erase gate',
			className: 'write',
			color: 'bg-amber-200'
		},
		{ key: 'gateNorms', short: 'g', label: 'output gate', className: 'gate', color: 'bg-violet-200' }
	];

	let vectorHoverIdx: number | null = null;
	const headCursors = {};

	$: blockTrace = $modelData?.rwkvBlocks?.[$blockIdx];

	const rawMetric = (key: string, tokenIndex: number) =>
		blockTrace?.timeMix?.[key]?.[tokenIndex]?.[$attentionHeadIdx] ?? 0;

	const scaledMetric = (key: string, tokenIndex: number) => {
		const value = rawMetric(key, tokenIndex);
		if (key === 'decayMeans' || key === 'writeGateMeans' || key === 'valueResidualGateMeans') {
			return Math.max(0.05, Math.min(1, value));
		}
		return Math.max(0.05, Math.min(1, value / (value + 8)));
	};

	onMount(() => {
		const unsubscribe = attentionHeadIdx.subscribe(async (newIdx) => {
			Object.values(headCursors).forEach((el) => {
				el.style.top = `${($vectorHeight / $modelMeta.attention_head_num) * newIdx}px`;
			});
		});

		return () => {
			unsubscribe();
		};
	});
</script>

<div class={classNames('qkv', 'rwkv-time-mix', className)} role="none" data-click="rwkv-time-mix-step">
	<div class="title" role="group" data-click="rwkv-time-mix-title">
		<div class="w-max">
			<TextbookTooltip id="rwkv-time-mix">Time Mix</TextbookTooltip>
		</div>
	</div>
	<div class="content relative">
		<div class="vector-column block-start-column relative flex" class:initial-column={$blockIdx === 0}>
			<div class="column vectors embedding-column">
				{#each $tokens as token, index}
					<div
						class={`vector ${$blockIdx !== 0 ? 'bg-blue-200' : inputColor}`}
						class:last={index === $tokens.length - 1}
					>
						<VectorCanvas colorScale={$blockIdx !== 0 ? 'blue' : 'gray'} />
					</div>
				{/each}
			</div>
			<Tooltip class="popover" triggeredBy={'.rwkv-time-mix .embedding-column .vector'} placement="right"
				>hidden x, vector({$modelMeta.dimension})</Tooltip
			>

			<div class="operations flex">
				<OperationGroup type="residual-start" id={'rwkv-time-residual'} />
				<OperationGroup type="ln" id={'rwkv-time-ln'} />
			</div>
		</div>
		<div class="column qkv-column">
			{#each $tokens as token, index}
				<div
					class="qkv-weighted vector x3 flex flex-col"
					class:last={index === $tokens.length - 1}
					on:mouseenter={() => {
						vectorHoverIdx = index;
					}}
					on:mouseleave={() => {
						vectorHoverIdx = null;
					}}
					role="group"
				>
					{#each streams as stream}
						<div
							class={`stream sub-vector ${stream.className} relative flex grow flex-col ${stream.color}`}
							style={`--fill:${scaledMetric(stream.key, index)}`}
							title={`${stream.label}: ${rawMetric(stream.key, index).toFixed(3)}`}
							aria-label={`${stream.label}: ${rawMetric(stream.key, index).toFixed(3)}`}
						>
							<div class="fill"></div>
							{#if stream.className === 'query' || stream.className === 'key' || stream.className === 'value'}
								<div
									class="sub-vector x1-12 head1 absolute"
									bind:this={headCursors[`token${index}_${stream.className}`]}
								></div>
							{/if}
							<div class="sub-vector head-rest">
								{#if vectorHoverIdx !== index}<span>{stream.short}</span>{/if}
							</div>
						</div>
					{/each}
				</div>
			{/each}
			<Tooltip class="popover" triggeredBy={'.rwkv-time-mix .qkv-column .vector'} placement="right"
				>RWKV-7 streams r, w, k~, k-, v, value residual, a, g for State Head {$attentionHeadIdx + 1}</Tooltip
			>
		</div>
	</div>
</div>

<style lang="scss">
	.rwkv-time-mix {
		> .title {
			display: flex;
			justify-content: end;
		}
		.content {
			display: grid;
			grid-template-columns: 1fr 1fr;

			.vector-column {
				position: relative;
				left: 3rem;

				&.initial-column {
					left: -12px;
				}
			}
			.qkv-column {
				position: relative;
				left: 12px;
				display: flex;
				flex-direction: column;
				align-items: end;
			}
		}
		.stream {
			min-height: 0;
			overflow: hidden;
			color: theme('colors.gray.700');

			.fill {
				position: absolute;
				inset: 0;
				width: calc(var(--fill) * 100%);
				background: rgba(255, 255, 255, 0.45);
			}
			.head-rest {
				height: 100%;
				display: flex;
				justify-content: center;
				align-items: center;
				font-size: 0.7rem;
				font-weight: 700;
				text-shadow:
					-1px -1px 0 white,
					1px -1px 0 white,
					-1px 1px 0 white,
					1px 1px 0 white;
			}
		}
	}
</style>
