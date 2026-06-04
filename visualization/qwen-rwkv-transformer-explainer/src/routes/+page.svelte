<script lang="ts">
	import {
		tokens,
		expandedBlock,
		vectorHeight,
		inputText,
		rootRem,
		sampling,
		maxVectorHeight,
		minVectorHeight,
		maxVectorScale,
		headContentHeight,
		temperature,
		isFetchingModel,
		selectedExampleIdx,
		isMobile,
		isOnBlockTransition,
		blockIdx,
		isTextbookOpen,
		inputTextExample,
		selectedTraceModel
	} from '~/store';
	import Sankey from '~/components/Sankey.svelte';
	import Attention from '~/components/Attention.svelte';
	import SubsequentBlocks from '~/components/SubsequentBlocks.svelte';
	import LinearSoftmax from '~/components/LinearSoftmax.svelte';
	import Embedding from '~/components/Embedding.svelte';
	import Mlp from '~/components/Mlp.svelte';
	import RwkvTimeMix from '~/components/RwkvTimeMix.svelte';
	import RwkvState from '~/components/RwkvState.svelte';
	import RwkvChannelMix from '~/components/RwkvChannelMix.svelte';

	import { onMount } from 'svelte';
	import classNames from 'classnames';

	import { runTrace } from '~/utils/data';
	import WeightPopovers from '~/components/WeightPopovers.svelte';
	import { fade } from 'svelte/transition';
	import BlockTransition from '~/components/BlockTransition.svelte';
	import QKV from '~/components/QKV.svelte';
	import Textbook from '~/components/textbook/Textbook.svelte';

	let active = false;

	onMount(async () => {
		isFetchingModel.set(false);
		active = true;

		const unsubscribe = subscribeInputs();

		return unsubscribe;
	});

	// Subscribe inputs
	const subscribeInputs = () => {
		const runCurrentTrace = (inputValue = $inputText) => {
			const trimmedInput = inputValue.trim();
			const selectedExampleMatches =
				inputTextExample[$selectedExampleIdx]?.trim() === trimmedInput;
			const matchedExampleIndex = inputTextExample.findIndex(
				(example) => example.trim() === trimmedInput
			);
			const exampleIndex = selectedExampleMatches
				? $selectedExampleIdx
				: matchedExampleIndex === -1
					? undefined
					: matchedExampleIndex;

			runTrace({
				input: trimmedInput,
				temperature: $temperature,
				sampling: $sampling,
				exampleIndex
			});
		};

		const unsubscribeInputText = inputText.subscribe((value) => {
			runCurrentTrace(value);
		});

		let initialTemperature = true; // prevent initial redundant rendering
		const unsubscribeTemperature = temperature.subscribe((value) => {
			if (initialTemperature) {
				initialTemperature = false;
				return;
			}
			runCurrentTrace();
		});

		let initialSampling = true; // prevent initial redundant rendering
		const unsubscribeSmapling = sampling.subscribe((value) => {
			if (initialSampling) {
				initialSampling = false;
				return;
			}
			runCurrentTrace();
		});

		let initialTraceModel = true; // prevent initial redundant rendering
		const unsubscribeTraceModel = selectedTraceModel.subscribe((value) => {
			if (initialTraceModel) {
				initialTraceModel = false;
				return;
			}
			runCurrentTrace();
		});

		return () => {
			unsubscribeInputText();
			unsubscribeTemperature();
			unsubscribeSmapling();
			unsubscribeTraceModel();
		};
	};

	// visual elements
	let vizHeight = 0;
	let titleHeight = rootRem * 5;

	const calculateVectorHeight = () => {
		const gaps = rootRem * 0.5 * ($tokens.length - 1);
		const vectorHeightVal = Math.min(
			Math.max((vizHeight - titleHeight - gaps) / $tokens.length / maxVectorScale, minVectorHeight),
			maxVectorHeight
		);
		vectorHeight.set(vectorHeightVal);
		headContentHeight.set(Math.max($tokens.length * vectorHeightVal * 3 + gaps, rootRem * 20));
	};

	$: if (vizHeight || $tokens.length) {
		calculateVectorHeight();
	}
</script>

<div
	class:active
	class="main-section h-full w-full"
	style={`--vector-height: ${$vectorHeight}px;--title-height: ${titleHeight}px;--content-height:${vizHeight - titleHeight}px;`}
>
	{#if !!$expandedBlock.id}
		<div
			class={classNames('dim', `${$expandedBlock.id || ''}`)}
			transition:fade={{ duration: 100 }}
		></div>
		<div
			class={classNames('dim-partial left', `${$expandedBlock.id || ''}`)}
			transition:fade={{ duration: 100 }}
		></div>
		<div
			class={classNames('dim-partial right', `${$expandedBlock.id || ''}`)}
			transition:fade={{ duration: 100 }}
		></div>
	{/if}
	<div class="sankey opacity-1" class:attention={$expandedBlock.id === 'attention'}>
		<Sankey />
	</div>
	<div class="nodes resize-watch">
		<div class="steps" class:expanded={!!$expandedBlock.id} bind:offsetHeight={vizHeight}>
			<Embedding className="step" />
			<div class="blocks relative">
				<div class="block-steps main" class:initial={$blockIdx === 0}>
					{#if $selectedTraceModel === 'rwkv'}
						<RwkvTimeMix className="step" />
						<RwkvState className="step" />
						<RwkvChannelMix className="step" />
					{:else}
						<QKV className="step" />
						<Attention className="step" />
						<Mlp className="step" />
					{/if}
				</div>
				<div
					class="block-steps next"
					class:hide={!$isOnBlockTransition}
					class:initial={$blockIdx === 0}
				>
					{#if $selectedTraceModel === 'rwkv'}
						<RwkvTimeMix className="step" />
						<RwkvState className="step" />
						<RwkvChannelMix className="step" />
					{:else}
						<QKV className="step" />
						<Attention className="step" />
						<Mlp className="step" />
					{/if}
				</div>
				<div class="transition-watch" class:hide={!$isOnBlockTransition}></div>
			</div>
			<SubsequentBlocks className="step" />
			<LinearSoftmax className="step" />
		</div>
		<WeightPopovers />
		<BlockTransition />
		{#if !$isMobile}
			<Textbook showTextCard={$isTextbookOpen} />
		{/if}
	</div>
</div>

<style lang="scss">
	.main-section {
		opacity: 0;
		&.active {
			opacity: 1;
		}
	}
	.nodes {
		height: 100%;
		width: 100%;
		padding: 1rem 0 3rem 0;
		position: relative;
	}
	.steps {
		position: relative;
		width: 100%;
		height: 100%;
		position: relative;
		display: grid;
		grid-template-columns: auto 3.5fr 0.5fr 0.5fr;

		&.expanded {
			:global(.step > .title) {
				padding-bottom: 3rem;
			}
		}

		.blocks {
			position: relative;
			width: 100%;
			height: 100%;

			.block-steps {
				height: 100%;
				width: 100%;
				position: absolute;
				display: grid;
				grid-template-columns: 0.5fr 2fr 1fr;
			}
			.block-steps.main {
				transform-origin: 3rem center;
				top: 0;
				left: 0;
			}
			.block-steps.next {
				transform-origin: right center;
				justify-content: end;
				top: 0;
				right: 0;
				pointer-events: none;
			}

			.transition-watch {
				position: absolute;
				top: 0;
				left: 0;
				height: 100%;
				width: 100%;
				pointer-events: none;
			}

			.hide {
				display: none;
			}
			&.animate-forward {
				.block-steps,
				.transition-watch {
					animation-duration: 800ms;
					animation-timing-function: ease-in;
				}
				.block-steps.main {
					animation-name: collapse;
					&.initial {
						transform-origin: left center;
					}
				}
				.block-steps.next {
					animation-name: expand;
				}
				.transition-watch {
					animation-name: width-collapse;
				}
			}

			&.animate-backward {
				.block-steps,
				.transition-watch {
					animation-duration: 800ms;
					animation-timing-function: ease-in;
				}
				.block-steps.main {
					animation-name: expand;
					&.initial {
						transform-origin: left center;
					}
				}
				.block-steps.next {
					animation-name: collapse;
				}
				.transition-watch {
					animation-name: width-collapse;
				}
			}
		}
	}
	@keyframes width-collapse {
		0% {
			width: 100%;
		}
		100% {
			width: 0%;
		}
	}
	@keyframes expand {
		0% {
			transform: scaleX(0);
		}
		100% {
			transform: scaleX(1);
		}
	}
	@keyframes collapse {
		0% {
			transform: scaleX(1);
		}
		100% {
			transform: scaleX(0);
		}
	}

	:global(.step) {
		height: 100%;
		display: grid;
		grid-template-rows: var(--title-height) 1fr;
	}

	:global(.step > .title) {
		z-index: $COLUMN_TITLE_INDEX;
		display: flex;
		flex-direction: column;
		justify-content: end;
		grid-row: 1;
		color: theme('colors.gray.400');
		white-space: nowrap;
		padding-bottom: 2rem;
		overflow: visible;
		min-width: 0;
		transition: all 0.5s;
		cursor: default;

		&:hover {
			color: theme('colors.gray.600');
		}
	}

	:global(.step > .title.expandable) {
		cursor: pointer;
	}

	:global(.step .content) {
		grid-row: 2;
		height: fit-content;
	}

	:global(.column) {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		position: relative;

		:global(.cell) {
			height: var(--vector-height);
			display: flex;
			gap: 1rem;
			align-items: center;
			position: relative;
		}

		:global(.subtitle) {
			position: absolute;
			top: 0;
			transform: translateY(calc(-100% - 1rem));
			text-align: center;
			font-size: 0.8rem;
			color: theme('colors.gray.400');
			width: 100%;
			z-index: $COLUMN_TITLE_INDEX;
		}
	}

	:global(.vector),
	:global(.sub-vector) {
		position: relative;
		z-index: $VECTOR_INDEX;
		width: 12px;
		height: var(--vector-height);
		flex-shrink: 0;
		justify-content: start;
	}
	:global(.cell.x1-12),
	:global(.vector.x1-12),
	:global(.sub-vector.x1-12) {
		height: calc(var(--vector-height) / 12);
	}

	:global(.cell.x3),
	:global(.vector.x3),
	:global(.sub-vector.x3) {
		height: calc(var(--vector-height) * 3);
	}
	:global(.cell.x4),
	:global(.vector.x4),
	:global(.sub-vector.x4) {
		height: calc(var(--vector-height) * 3.1);
	}

	:global(.vector.vocab),
	:global(.sub-vector.vocab) {
		height: 100%;
		width: 0;
	}

	:global(.sub-vector.head-rest) {
		flex: 1 0 0;
	}

	:global(.label) {
		font-size: 0.9rem;
		color: theme('colors.gray.700');
		z-index: $VECTOR_INDEX;
		display: inline;
		max-width: 7rem;
		overflow: hidden;
		text-overflow: ellipsis;
		text-align: right;
		line-height: var(--vector-height);
		height: var(--vector-height);
		flex-shrink: 0;
	}
	:global(.label.float) {
		position: absolute;
		left: -0.8rem;
		transform: translateX(-100%);
	}
	:global(.label.float-right) {
		position: absolute;
		left: -0.8rem;
	}

	:global(.ellipsis) {
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	:global(.bounding) {
		position: absolute;
		box-sizing: content-box;
		top: -0.5rem;
		padding: 0.5rem 0;
		left: 0;
		height: 100%;
		border: 2px dashed theme('colors.gray.300');
		border-radius: 0.5rem;
		transition: opacity 0.5s;
		opacity: 0;
		pointer-events: none;
	}
	:global(.bounding.active) {
		opacity: 0.8;
	}

	:global(.popover) {
		z-index: $POPOVER_INDEX;
		width: max-content;
	}

	:global(.tooltip) {
		z-index: $TOOLTIP_INDEX;
		background-color: white !important;
		color: theme('colors.gray.600') !important;
		border: 1px solid theme('colors.gray.200') !important;
		padding: 0.2rem 0.5rem !important;
		font-size: 0.8rem !important;
		white-space: nowrap;
		font-weight: 300 !important;
		border-color: theme('colors.gray.200') !important;
	}
	.dim {
		position: absolute;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		z-index: $DIM_INDEX;
		background-color: white;
		opacity: 0.7;
		user-select: none;

		&.attention {
			z-index: 0;
		}
	}
	.dim-partial {
		user-select: none;
		z-index: $PARTIAL_DIM_INDEX;
		position: absolute;
		top: 0;
		height: 100%;

		&.right {
			right: 0;
			background: linear-gradient(90deg, rgba(255, 255, 255, 0) 0%, rgba(255, 255, 255, 1) 80%);
		}
		&.left {
			left: 0;
			background: linear-gradient(-90deg, rgba(255, 255, 255, 0) 0%, rgba(255, 255, 255, 1) 80%);
		}

		&.embedding {
			&.left {
				display: none;
			}
			&.right {
				width: 60%;
			}
		}
		&.attention {
			&.left {
				width: 20%;
			}
			&.right {
				width: 20%;
			}
		}
		&.softmax {
			&.left {
				width: 60%;
			}
			&.right {
				display: none;
			}
		}
	}
	.sankey {
		position: absolute;
		left: 0;
		top: 0;
		width: 100%;
		height: 100%;

		&.attention {
			:global(.sankey-top) {
				z-index: $EXPANDED_ATTENTION_INDEX !important;
				pointer-events: none;
			}
			// :global(.sankey-top > g) {
			// 	opacity: 0.3;
			// }
			// :global(.sankey-top > g.attention) {
			// 	opacity: 1;
			// }
		}
	}

	:global(svg g.path-group) {
		transition: opacity 0.5s;
	}
	:global(div.step > div) {
		transition: opacity 0.5s;
	}
	:global(div.step .column) {
		transition: opacity 0.5s;
	}
</style>
