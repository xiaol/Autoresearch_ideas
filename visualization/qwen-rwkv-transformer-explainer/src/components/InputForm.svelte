<script lang="ts">
	import { Dropdown, DropdownItem, ButtonGroup } from 'flowbite-svelte';
	import Temperature from './Temperature.svelte';

	import { ChevronDownOutline } from 'flowbite-svelte-icons';
	import {
		inputText,
		isModelRunning,
		predictedToken,
		inputTextExample,
		isFetchingModel,
		expandedBlock,
		selectedExampleIdx,
		isLoaded,
		isOnAnimation,
		isMobile,
		weightPopover,
		sampling,
		attentionHeadIdx,
		blockIdx,
		temperature,
		tokenIds,
		userId,
		traceModelOptions,
		selectedTraceModel
	} from '~/store';
	import LoadingDots from './common/LoadingDots.svelte';
	import classNames from 'classnames';
	import Sampling from './Sampling.svelte';
	import { completeCurrentAnimation } from '~/utils/animation';
	import { textPages } from '~/utils/textbookPages';

	let inputRef: HTMLDivElement;
	let predictRef: HTMLDivElement;

	let useCustomInput = false;

	$: inputTextTemp = $inputText || '';

	$: predictedTokenTemp = $predictedToken?.token || '';

	const wordLimit = 12;
	$: exceedLimit = inputTextTemp.split(' ').length >= wordLimit;

	// Text input
	const syncInputText = ({ acceptPrediction = false } = {}) => {
		const currentText = inputRef?.innerText ?? inputTextTemp;
		const nextText = currentText + (acceptPrediction ? predictedTokenTemp : '');
		const formattedString = nextText.replace(/[\s\n]+/g, ' ');

		inputTextTemp = formattedString;

		predictedTokenTemp = '';
		inputRef.innerText = inputTextTemp;
	};

	const onFocusInput = (e) => {
		syncInputText();
	};

	const acceptPredictedToken = () => {
		syncInputText({ acceptPrediction: true });
	};

	const onInput = (e) => {
		inputTextTemp = inputRef.innerText;
	};

	const handleSubmit = (e) => {
		// Complete any running animation before starting new generation
		completeCurrentAnimation();

		setTimeout(() => {
			syncInputText();
			textPages.find((page) => page.id === 'how-transformers-work')?.complete();

			inputText.update((prev) => {
				if (prev === inputTextTemp.trim()) {
					return inputTextTemp + ' ';
				}
				return inputTextTemp;
			});

			window.dataLayer?.push({
				event: 'generate-next-token',
				attn_head_num: $attentionHeadIdx,
				transformer_block_num: $blockIdx,
				sampling_type: $sampling.type,
				sampling_value: $sampling.value,
				temperature_value: $temperature,
				current_token_length: $tokenIds.length,
				input_word_count: inputTextTemp
					.trim()
					.split(/\s+/)
					.filter((word) => word.length > 0).length,
				use_custom_input: useCustomInput,
				user_id: $userId
			});
		}, 0);
	};

	const onSelectTraceModel = (modelKey: TraceModelKey) => {
		if ($selectedTraceModel === modelKey || selectDisabled) return;
		completeCurrentAnimation();
		expandedBlock.set({ id: null });
		weightPopover.set(null);
		selectedTraceModel.set(modelKey);
		window.dataLayer?.push({
			event: 'select-trace-model',
			trace_model: modelKey,
			user_id: $userId
		});
	};

	const handleKeyDown = (e) => {
		if (e.key === 'Enter') {
			e.preventDefault();
			if (disabled || exceedLimit) return;
			handleSubmit(e);
			return;
		}
		useCustomInput = true;
	};

	// Example select box
	let dropdownOpen = false;
	const onSelectExample = (d, i) => {
		if ($isFetchingModel) {
			textPages.find((page) => page.id === 'how-transformers-work')?.complete();
		}

		dropdownOpen = false;

		selectedExampleIdx.set(i);
		predictedTokenTemp = '';

		inputTextTemp = d;
		inputRef.innerText = inputTextTemp;
		inputText.update((prev) => {
			if (prev === d.trim()) {
				return d + ' ';
			}
			return d.trim();
		});
		useCustomInput = false;
	};

	const moveCursorToEnd = (element) => {
		const range = document.createRange();
		const sel = window.getSelection();
		range.selectNodeContents(element);
		range.collapse(false);
		sel.removeAllRanges();
		sel.addRange(range);
		element.focus();
	};

	$: isLoading = $isFetchingModel || $isModelRunning;
	$: disabled =
		$isFetchingModel ||
		// $isModelRunning ||
		$expandedBlock.id !== null ||
		!!$weightPopover;
	$: selectDisabled = $isModelRunning || $expandedBlock.id !== null || !!$weightPopover;
	$: parameterDisabled = !!$weightPopover;
</script>

<div class="input-area" data-click="input-area">
	<form class="input-form" data-click="input-form">
		<ButtonGroup class="input-btn-group" size="sm">
			<button
				data-click="dropdown-btn"
				type="button"
				disabled={selectDisabled}
				class:selectDisabled
				class="select-button inline-flex shrink-0 items-center justify-center border border-s-0 border-gray-200 bg-white px-3 py-2 text-center text-xs font-medium text-gray-900 first:rounded-s-lg first:border-s last:rounded-e-lg"
			>
				Examples<ChevronDownOutline class="pointer-events-none h-4 w-4 text-gray-500" />
			</button>
			<Dropdown bind:open={dropdownOpen} class="example-dropdown">
				{#each inputTextExample as text, index}
					<DropdownItem
						data-click={`dropdown-item-${index}`}
						class={$selectedExampleIdx === index && 'active'}
						on:click={() => {
							onSelectExample(text, index);
						}}>{text}</DropdownItem
					>
				{/each}
			</Dropdown>

			<div
				data-click="text-input"
				class="input-container"
				class:disabled
				role="none"
				on:keydown={(e) => {
					e.stopPropagation();
					inputRef.focus();
				}}
				on:click={(e) => {
					e.stopPropagation();
					inputRef.focus();
				}}
			>
				<div class={`editable ${!$isModelRunning ? 'w-full' : ''}`}>
					<div
						bind:this={inputRef}
						contenteditable={!disabled}
						class="text-box"
						placeholder="Test your own input text"
						on:focus={onFocusInput}
						on:input={onInput}
						on:keydown={handleKeyDown}
						on:click={(e) => {
							e.stopPropagation();
						}}
						role="input"
					>
						{inputTextTemp}
					</div>
					{#if !$isModelRunning}
						<div
							bind:this={predictRef}
							class="predicted"
							role="none"
							on:click={(e) => {
								e.stopPropagation();
								acceptPredictedToken();
								inputRef.focus();
								moveCursorToEnd(inputRef);
							}}
						>
							<span>{predictedTokenTemp}</span>
						</div>
					{/if}
				</div>
				{#if $isModelRunning}
					<div class="loading"><LoadingDots /></div>
				{/if}
				{#if $isMobile}
					<span class="helper-text"
						>Try the examples. Please use a desktop computer to input Qwen prompts directly.</span
					>
				{:else if $isLoaded && $isFetchingModel}
					<span class="helper-text"
						>Try the examples while trace mode is preparing.</span
					>
				{:else if exceedLimit}
					<span class="helper-text">You can enter up to {wordLimit} words.</span>
				{/if}
			</div>
		</ButtonGroup>
		<button
			data-click="generate-btn"
			disabled={disabled || exceedLimit || exceedLimit}
			class={classNames('generate-button rounded-lg text-center text-sm shadow-sm', {
				disabled: disabled || exceedLimit,
				active: !(disabled || exceedLimit)
			})}
			type="submit"
			on:click={handleSubmit}
		>
			Generate
		</button>
	</form>
	<div class="parameters" data-click="input-parameters">
		<div class="model-selector" data-click="model-selector">
			{#each traceModelOptions as option}
				<button
					type="button"
					class:active={$selectedTraceModel === option.key}
					disabled={selectDisabled}
					on:click={() => onSelectTraceModel(option.key)}
				>
					{option.label}
				</button>
			{/each}
		</div>
		<Temperature disabled={parameterDisabled} />
		<Sampling disabled={parameterDisabled} />
	</div>
</div>

<style lang="scss">
	.parameters {
		display: flex;
		gap: 1rem;
		align-items: center;
	}
	.model-selector {
		display: flex;
		border: 1px solid theme('colors.gray.300');
		border-radius: 0.5rem;
		overflow: hidden;
		flex-shrink: 0;

		button {
			padding: 0.4rem 0.7rem;
			font-size: 0.85rem;
			line-height: 1rem;
			color: theme('colors.gray.700');
			background: white;
			border-right: 1px solid theme('colors.gray.300');
			white-space: nowrap;

			&:last-child {
				border-right: none;
			}
			&.active {
				background: theme('colors.gray.900');
				color: white;
			}
			&:disabled {
				cursor: not-allowed;
				color: theme('colors.gray.400');
			}
		}
	}
	.input-area {
		width: 100%;
		flex-shrink: 0;
		display: flex;
		gap: 1rem;
		padding-left: 1rem;
		padding-right: 1.25rem;

		.input-form {
			width: 100%;
			flex: 1 0 0;
			display: flex;
			align-items: center;
			gap: 0.5rem;

			:global(.input-btn-group) {
				flex: 1 0 0;
				display: flex;
			}
		}
	}
	.input-container {
		position: relative;
		display: flex;
		flex: 1 0 0;
		align-items: center;

		border: 1px solid theme('colors.gray.300');
		color: theme('colors.gray.900');
		border-left: none;
		border-start-end-radius: 0.5rem;
		border-end-end-radius: 0.5rem;
		font-size: 1rem;
		line-height: 1rem;
		padding: 0.5rem;
		white-space: pre-wrap;
		gap: 0.3rem;

		width: 10px; // to keep input box width

		.editable {
			overflow-y: hidden;
			justify-content: end; // to show the last word

			display: flex;
			align-items: center;
			overflow-x: hidden;
			white-space: nowrap;

			.text-box {
				white-space: nowrap;
				br {
					display: none;
				}

				&:focus {
					outline: none;
				}
			}
			.predicted {
				flex: 1 0 0;
				color: var(--predicted-color);
				font-weight: 600;
				span {
					white-space: pre;
				}
			}
		}

		&.disabled {
			cursor: not-allowed;
		}

		.loading {
			flex-shrink: 0;
		}
	}

	.select-button {
		flex-shrink: 0;
		font-size: 0.9rem;
		border: 1px solid theme('colors.gray.300');
		color: theme('colors.gray.900');
		&:hover {
			background-color: theme('colors.gray.100');
		}
		&:focus {
			outline: none;
		}
		&.disabled {
			cursor: not-allowed;
		}
	}
	:global(.example-dropdown) {
		:global(.active) {
			background-color: theme('colors.gray.100') !important;
		}
	}
	.helper-text {
		position: absolute;
		bottom: 0;
		right: 0;
		padding: 0.3rem 0;
		transform: translate(0, 100%);
		color: theme('colors.gray.400');
		font-size: 0.9rem;
	}
	:global(.generate-button) {
		padding: 0.4rem 0.8rem;
		border: 1px solid theme('colors.gray.300');
		color: theme('colors.gray.900');
		transition: all 0.2s;
		flex-shrink: 0;
	}
	:global(.generate-button.active) {
		&:hover {
			border: 1px solid var(--predicted-color);
			color: var(--predicted-color);
		}
	}
	:global(.generate-button.disabled) {
		opacity: 1;
		background-color: theme('colors.gray.100');
		color: theme('colors.gray.400');
		cursor: not-allowed;
	}

	:global(.generate-button):focus {
		border: 1px solid var(--predicted-color);
		color: var(--predicted-color);
	}
</style>
