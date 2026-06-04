<script lang="ts">
	import { tokens, modelMeta, blockIdx, attentionHeadIdx, vectorHeight, modelData } from '~/store';
	import classNames from 'classnames';
	import { onMount, setContext } from 'svelte';
	import OperationGroup from './OperationGroup.svelte';
	import VectorCanvas from './common/VectorCanvas.svelte';
	import { Tooltip } from 'flowbite-svelte';
	import TextbookTooltip from './common/TextbookTooltip.svelte';

	export let className: string | undefined = undefined;

	setContext('block-id', 'mlp');

	const inputColor = 'bg-purple-200';
	const keyColor = 'bg-amber-200';
	const outputColor = 'bg-blue-200';
	let vectorHoverIdx: number | null = null;
	const headCursors = {};

	$: blockTrace = $modelData?.rwkvBlocks?.[$blockIdx];
	$: keyNorms = blockTrace?.channelMix?.keyActivationNorms || [];
	$: outputNorms = blockTrace?.channelMix?.outputNorms || [];

	const scaled = (values: number[], index: number) => {
		const maxValue = Math.max(...values, 1e-6);
		return Math.max(0.05, Math.min(1, (values[index] ?? 0) / maxValue));
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

<div class={classNames('mlp', 'mlpUp', 'mlpDown', 'rwkv-channel-mix', className)} data-click="rwkv-channel-mix-step">
	<div class="title" role="group" data-click="rwkv-channel-mix-title">
		<div class="w-max">
			<TextbookTooltip id="rwkv-channel-mix">Channel Mix</TextbookTooltip>
		</div>
	</div>

	<div class="content relative">
		<div class="bounding mlp-bounding"></div>
		<div class="layer mlpUp first-layer flex">
			<div class="column initial">
				{#each $tokens as token, index}
					<div
						class="cell"
						class:last={index === $tokens.length - 1}
						on:mouseenter={() => {
							vectorHoverIdx = index;
						}}
						on:mouseleave={() => {
							vectorHoverIdx = null;
						}}
						role="group"
					>
						<span class="label float">{token}</span>
						<div class={`vector flex flex-col ${inputColor}`}>
							<VectorCanvas colorScale="purple" active={vectorHoverIdx === index} />
							<div class="sub-vector x1-12 head1 absolute" bind:this={headCursors[`token${index}_out`]}></div>
							<div class="sub-vector head-rest grow"></div>
						</div>
					</div>
				{/each}
			</div>
			<Tooltip triggeredBy={'.step.rwkv-channel-mix .initial .cell'} class="popover" placement="right">
				hidden x, vector({$modelMeta.dimension})</Tooltip
			>
			<OperationGroup type="residual-end" id={'rwkv-time-residual'} />
			<OperationGroup type="ln" id={'rwkv-channel-ln'} />
			<OperationGroup type="residual-start" id={'rwkv-channel-residual'} />
		</div>
		<div class="layer mlpUp mlpDown second-layer flex justify-between">
			<div class="column mlp-mid-column">
				{#each $tokens as token, index}
					<div class={classNames('cell x4', { small: index !== 0 && index !== $tokens.length - 1 })} class:last={index === $tokens.length - 1}>
						<div class={classNames(`vector x4 ${keyColor} opacity-90`)} style={`--fill:${scaled(keyNorms, index)}`}>
							<div class="metric"></div>
							<span>ReLU^2</span>
						</div>
					</div>
				{/each}
			</div>
		</div>
		<Tooltip triggeredBy={'.step.rwkv-channel-mix .mlp-mid-column .cell'} class="popover" placement="right">
			squared-ReLU channel key activation with token-conditioned scaling</Tooltip
		>
		<div class="layer mlpDown out-layer relative flex justify-between">
			<div class="activation">
				<OperationGroup type="activation" id={'mlp-activation'} className="x4" />
			</div>
			<div class="ouputs flex">
				<div class="column out-label">
					{#each $tokens as token, index}
						<div class="cell" class:last={index === $tokens.length - 1}>
							<span class="label float">{token}</span>
						</div>
					{/each}
				</div>
				<OperationGroup type="residual-end" id={'rwkv-channel-residual'} />
				<div class="column out mlp-out-column" class:last-block={$blockIdx === $modelMeta.layer_num - 1}>
					{#each $tokens as token, index}
						<div class="cell" class:last={index === $tokens.length - 1}>
							<div class={`vector ${outputColor}`} style={`--fill:${scaled(outputNorms, index)}`}>
								<div class="metric"></div>
								<VectorCanvas colorScale="blue" />
							</div>
						</div>
					{/each}
				</div>
				<Tooltip triggeredBy={'.step.rwkv-channel-mix .mlp-out-column .cell'} class="popover" placement="right">
					channel-mix value projection, vector({$modelMeta.dimension})</Tooltip
				>
			</div>
		</div>
	</div>
</div>

<style lang="scss">
	.rwkv-channel-mix {
		.mlp-bounding {
			top: -0.5rem;
			padding: 0.5rem 0;
			left: -0.2rem;
			width: calc(100% + 0.2rem);
			height: 100%;
		}
		.content {
			display: grid;
			grid-template-columns: repeat(4, minmax(var(--min-column-width), 1fr));
		}
		.first-layer {
			grid-column: span 2;
		}
		.activation {
			position: relative;
			left: calc(-100% + 0.8rem);
		}
		.vector {
			overflow: hidden;
			display: flex;
			align-items: center;
			justify-content: center;
			color: theme('colors.gray.600');
			font-size: 0.7rem;
			font-weight: 700;
		}
		.metric {
			position: absolute;
			inset: auto 0 0 0;
			height: calc(var(--fill) * 100%);
			background: rgba(255, 255, 255, 0.45);
		}
		.column.out-label {
			.label {
				opacity: 0;
			}
			.last .label {
				opacity: 1;
			}
		}
		.last-block {
			pointer-events: none;
			.vector {
				width: 0;
			}
		}
	}
</style>
