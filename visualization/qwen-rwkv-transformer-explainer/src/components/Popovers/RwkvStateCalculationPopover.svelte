<script lang="ts">
	import {
		attentionHeadIdx,
		blockIdx,
		modelData,
		rwkvStateTokenIdx,
		rwkvStateViewMode,
		tokens
	} from '~/store';
	import Matrix from '~/components/common/Matrix.svelte';
	import WeightPopoverCard from '~/components/common/WeightPopoverCard.svelte';
	import Katex from '~/utils/Katex.svelte';
	import { gsap } from '~/utils/gsap';
	import * as d3 from 'd3';
	import resolveConfig from 'tailwindcss/resolveConfig';
	import tailwindConfig from '../../../tailwind.config';
	import { onDestroy, onMount } from 'svelte';

	const { theme } = resolveConfig(tailwindConfig);

	let isAnimationActive = false;
	let timeline = gsap.timeline();
	let hoveredTokenIndex: number | null = null;
	let manifoldDetailMode: 'plane' | 'multi' = 'plane';
	const manifoldWidth = 320;
	const manifoldHeight = 190;
	const manifoldPadding = 28;
	const manifoldInnerWidth = manifoldWidth - manifoldPadding * 2;
	const manifoldInnerHeight = manifoldHeight - manifoldPadding * 2;
	const manifoldGrid = Array.from({ length: 5 }, (_, index) => ({
		x: manifoldPadding + (index * manifoldInnerWidth) / 4,
		y: manifoldPadding + (index * manifoldInnerHeight) / 4
	}));
	const multiDimAxes = [
		{ key: 'rowCenter', label: 'row' },
		{ key: 'colCenter', label: 'col' },
		{ key: 'diagonalShare', label: 'diag' },
		{ key: 'antiDiagonalShare', label: 'anti' },
		{ key: 'energyNorm', label: 'energy' }
	];
	type RwkvTransitionKey = keyof NonNullable<RwkvBlockTrace['stateTransition']>;

	$: blockTrace = $modelData?.rwkvBlocks?.[$blockIdx];
	$: activeTokenIndex = clamp(
		hoveredTokenIndex ?? $rwkvStateTokenIdx ?? Math.max($tokens.length - 1, 0),
		0,
		Math.max($tokens.length - 1, 0)
	);
	$: activeToken = $tokens[activeTokenIndex] || '';

	const headSeries = (
		trace: RwkvBlockTrace | undefined,
		key: keyof RwkvBlockTrace['timeMix'],
		headIndex: number
	) => (trace?.timeMix?.[key] || []).map((row) => row[headIndex] ?? 0);

	$: stateBefore = headSeries(blockTrace, 'stateBeforeNorms', $attentionHeadIdx);
	$: decayMeans = headSeries(blockTrace, 'decayMeans', $attentionHeadIdx);
	$: decayed = headSeries(blockTrace, 'decayedStateNorms', $attentionHeadIdx);
	$: erase = headSeries(blockTrace, 'eraseNorms', $attentionHeadIdx);
	$: write = headSeries(blockTrace, 'writeNorms', $attentionHeadIdx);
	$: stateAfter = headSeries(blockTrace, 'stateAfterNorms', $attentionHeadIdx);
	$: read = headSeries(blockTrace, 'readNorms', $attentionHeadIdx);
	$: bonus = headSeries(blockTrace, 'bonusNorms', $attentionHeadIdx);
	$: output = headSeries(blockTrace, 'outputNorms', $attentionHeadIdx);

	$: rawMaps = [
		transitionMatrix(
			blockTrace,
			'stateBefore',
			activeTokenIndex,
			$attentionHeadIdx,
			valueAt(stateBefore, activeTokenIndex)
		),
		decayDiagMap(valueAt(decayMeans, activeTokenIndex)),
		transitionMatrix(
			blockTrace,
			'decayed',
			activeTokenIndex,
			$attentionHeadIdx,
			valueAt(decayed, activeTokenIndex)
		),
		transitionMatrix(
			blockTrace,
			'erase',
			activeTokenIndex,
			$attentionHeadIdx,
			valueAt(erase, activeTokenIndex)
		),
		transitionMatrix(
			blockTrace,
			'write',
			activeTokenIndex,
			$attentionHeadIdx,
			valueAt(write, activeTokenIndex)
		),
		transitionMatrix(
			blockTrace,
			'stateAfter',
			activeTokenIndex,
			$attentionHeadIdx,
			valueAt(stateAfter, activeTokenIndex)
		)
	];
	$: maps = normalizeMaps(rawMaps);
	$: transitionSignalMax = Math.max(
		...[
			valueAt(stateBefore, activeTokenIndex),
			valueAt(decayMeans, activeTokenIndex),
			valueAt(decayed, activeTokenIndex),
			valueAt(erase, activeTokenIndex),
			valueAt(write, activeTokenIndex),
			valueAt(stateAfter, activeTokenIndex)
		].map((value) => Math.abs(value)),
		1e-6
	);
	$: terms = [
		{
			key: 'state-before',
			label: 'S_prev',
			size: '64 x 64',
			value: valueAt(stateBefore, activeTokenIndex),
			signal: signalFill(valueAt(stateBefore, activeTokenIndex)),
			rawIndex: 0,
			data: maps[0],
			color: 'sky',
			operator: 'x'
		},
		{
			key: 'decay',
			label: 'diag(w)',
			size: '64 x 64',
			value: valueAt(decayMeans, activeTokenIndex),
			signal: signalFill(valueAt(decayMeans, activeTokenIndex)),
			rawIndex: 1,
			data: maps[1],
			color: 'rose',
			operator: '='
		},
		{
			key: 'decayed',
			label: 'S*w',
			size: '64 x 64',
			value: valueAt(decayed, activeTokenIndex),
			signal: signalFill(valueAt(decayed, activeTokenIndex)),
			rawIndex: 2,
			data: maps[2],
			color: 'rose',
			operator: '-'
		},
		{
			key: 'erase',
			label: 'erase',
			size: '64 x 64',
			value: valueAt(erase, activeTokenIndex),
			signal: signalFill(valueAt(erase, activeTokenIndex)),
			rawIndex: 3,
			data: maps[3],
			color: 'red',
			operator: '+'
		},
		{
			key: 'write',
			label: 'write',
			size: '64 x 64',
			value: valueAt(write, activeTokenIndex),
			signal: signalFill(valueAt(write, activeTokenIndex)),
			rawIndex: 4,
			data: maps[4],
			color: 'emerald',
			operator: '='
		},
		{
			key: 'state-after',
			label: 'S_t',
			size: '64 x 64',
			value: valueAt(stateAfter, activeTokenIndex),
			signal: signalFill(valueAt(stateAfter, activeTokenIndex)),
			rawIndex: 5,
			data: maps[5],
			color: 'purple',
			operator: null
		}
	];
	$: readoutSignalMax = Math.max(
		...[
			valueAt(read, activeTokenIndex),
			valueAt(bonus, activeTokenIndex),
			valueAt(output, activeTokenIndex)
		].map((value) => Math.abs(value)),
		1e-6
	);
	$: readoutTerms = [
		{
			label: 'read',
			value: valueAt(read, activeTokenIndex),
			signal: readoutSignal(valueAt(read, activeTokenIndex)),
			color: 'violet'
		},
		{
			label: 'bonus',
			value: valueAt(bonus, activeTokenIndex),
			signal: readoutSignal(valueAt(bonus, activeTokenIndex)),
			color: 'amber'
		},
		{
			label: 'out',
			value: valueAt(output, activeTokenIndex),
			signal: readoutSignal(valueAt(output, activeTokenIndex)),
			color: 'purple'
		}
	];
	$: stateTrajectory = buildStateTrajectory($tokens, blockTrace, $attentionHeadIdx, stateAfter);
	$: stateTrajectoryPath = pathFromPoints(stateTrajectory);
	$: stateTrajectorySegments = segmentsFromPoints(stateTrajectory);
	$: activeTrajectorySegment = activeTokenIndex > 0 ? stateTrajectorySegments[activeTokenIndex - 1] : null;
	$: multiDimStatePoints = buildMultiDimStatePoints(
		$tokens,
		blockTrace,
		$attentionHeadIdx,
		stateAfter
	);
	$: selectedTermPoints = terms
		.filter((term) => term.key !== 'decay')
		.map((term) => ({
			...projectMatrix(rawMaps[term.rawIndex]),
			key: term.key,
			label: term.label,
			color: term.color
		}));
	$: selectedStatePoint = selectedTermPoints.find((point) => point.key === 'state-after');
	$: selectedInfluenceSegments = selectedStatePoint
		? selectedTermPoints
				.filter((point) => point.key !== 'state-after')
				.map((point) => ({
					...point,
					d: pathFromPair(point, selectedStatePoint)
				}))
		: [];

	const emptyMap = () => Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => 0));
	const valueAt = (values: number[], index: number) => values[index] ?? 0;
	const decayDiagMap = (value: number) =>
		Array.from({ length: 4 }, (_, row) =>
			Array.from({ length: 4 }, (_, col) => (row === col ? Math.max(value, 0) : 0))
		);
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
	const normalizeMaps = (matrices: number[][][]) => {
		return matrices.map((matrix) => {
			const maxValue = Math.max(...matrix.flat().map((value) => Math.abs(value)), 1e-6);
			return matrix.map((row) => row.map((value) => Math.abs(value) / maxValue));
		});
	};
	const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));
	const signalFill = (value: number) => {
		const scaled = Math.abs(value) / transitionSignalMax;
		if (scaled <= 0) return 0;
		return Math.max(0.08, Math.min(1, Math.sqrt(scaled)));
	};
	const readoutSignal = (value: number) => {
		const scaled = Math.abs(value) / readoutSignalMax;
		if (scaled <= 0) return 0;
		return Math.max(0.08, Math.min(1, Math.sqrt(scaled)));
	};
	const matrixFeatures = (matrix: number[][]) => {
		let total = 0;
		let rowMass = 0;
		let colMass = 0;
		let diagonalMass = 0;
		let antiDiagonalMass = 0;
		const rows = matrix.length;
		const cols = matrix[0]?.length || 0;

		matrix.forEach((row, rowIndex) => {
			row.forEach((value, colIndex) => {
				const magnitude = Math.abs(value || 0);
				total += magnitude;
				rowMass += magnitude * rowIndex;
				colMass += magnitude * colIndex;
				if (rowIndex === colIndex) diagonalMass += magnitude;
				if (rowIndex + colIndex === Math.min(rows, cols) - 1) antiDiagonalMass += magnitude;
			});
		});

		if (total <= 1e-12 || rows === 0 || cols === 0) {
			return {
				rowCenter: 0,
				colCenter: 0,
				diagonalShare: 0,
				antiDiagonalShare: 0,
				energy: 0,
				empty: true
			};
		}

		return {
			rowCenter: rowMass / total / Math.max(rows - 1, 1),
			colCenter: colMass / total / Math.max(cols - 1, 1),
			diagonalShare: diagonalMass / total,
			antiDiagonalShare: antiDiagonalMass / total,
			energy: total,
			empty: false
		};
	};
	const projectMatrix = (matrix: number[][]) => {
		const features = matrixFeatures(matrix);
		if (features.empty) {
			return {
				x: manifoldPadding,
				y: manifoldHeight - manifoldPadding,
				energy: 0,
				empty: true
			};
		}
		const diagonalBias = features.diagonalShare - features.antiDiagonalShare;

		return {
			x: manifoldPadding + features.colCenter * (manifoldWidth - manifoldPadding * 2),
			y: clamp(
				manifoldHeight -
					manifoldPadding -
					features.rowCenter * (manifoldHeight - manifoldPadding * 2) -
					diagonalBias * 12,
				manifoldPadding,
				manifoldHeight - manifoldPadding
			),
			energy: features.energy,
			empty: false
		};
	};
	const buildStateTrajectory = (
		tokenList: string[],
		trace: RwkvBlockTrace | undefined,
		headIndex: number,
		stateAfterValues: number[]
	) => {
		const projected = tokenList.map((token, index) => ({
			...projectMatrix(
				transitionMatrix(
					trace,
					'stateAfter',
					index,
					headIndex,
					valueAt(stateAfterValues, index)
				)
			),
			token,
			index
		}));
		const maxEnergy = Math.max(...projected.map((point) => point.energy), 1e-6);
		return projected.map((point) => ({
			...point,
			radius: point.empty ? 3 : 4 + 7 * Math.sqrt(point.energy / maxEnergy)
		}));
	};
	const pathFromPoints = (points) =>
		points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`).join(' ');
	const pathFromPair = (start, end) => `M ${start.x} ${start.y} L ${end.x} ${end.y}`;
	const segmentsFromPoints = (points) =>
		points.slice(1).map((point, index) => ({
			d: pathFromPair(points[index], point),
			from: points[index],
			to: point,
			index: index + 1
		}));
	const multiAxisX = (index: number) =>
		manifoldPadding + (index * manifoldInnerWidth) / Math.max(multiDimAxes.length - 1, 1);
	const multiAxisY = (value: number) =>
		manifoldPadding + (1 - clamp(value, 0, 1)) * manifoldInnerHeight;
	const buildMultiDimStatePoints = (
		tokenList: string[],
		trace: RwkvBlockTrace | undefined,
		headIndex: number,
		stateAfterValues: number[]
	) => {
		const points = tokenList.map((token, index) => ({
			...matrixFeatures(
				transitionMatrix(
					trace,
					'stateAfter',
					index,
					headIndex,
					valueAt(stateAfterValues, index)
				)
			),
			token,
			index
		}));
		const maxEnergy = Math.max(...points.map((point) => point.energy), 1e-6);
		return points.map((point) => {
			const values = {
				rowCenter: point.rowCenter,
				colCenter: point.colCenter,
				diagonalShare: point.diagonalShare,
				antiDiagonalShare: point.antiDiagonalShare,
				energyNorm: point.energy / maxEnergy
			};
			return {
				...point,
				values,
				path: multiDimAxes
					.map((axis, axisIndex) => {
						const command = axisIndex === 0 ? 'M' : 'L';
						return `${command} ${multiAxisX(axisIndex)} ${multiAxisY(values[axis.key])}`;
					})
					.join(' ')
			};
		});
	};
	const colorFor = (color: string, shade = 600) => theme.colors[color][shade];
	const colorScale = (color: string) => (value: number) =>
		d3.interpolate('white', theme.colors[color][600])(value);
	const formatValue = (value: number) => {
		if (!Number.isFinite(value)) return '0';
		if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(1);
		if (Math.abs(value) >= 100) return value.toFixed(0);
		if (Math.abs(value) >= 10) return value.toFixed(1);
		if (Math.abs(value) >= 1) return value.toFixed(2);
		return value.toFixed(3);
	};
	const selectToken = (index: number) => {
		rwkvStateTokenIdx.set(index);
	};

	const draw = () => {
		timeline.clear();
		const cards = d3.selectAll('.rwkv-state-popover .calc-card').nodes();
		timeline.set(cards, { opacity: 0.35, scale: 0.96 });
		cards.forEach((card) => {
			timeline
				.to(card, { opacity: 1, scale: 1, duration: 0.18 })
				.to(card, { opacity: 0.65, scale: 0.98, duration: 0.1 }, '+=0.12');
		});
		timeline.to(cards, { opacity: 1, scale: 1, duration: 0.18 });
	};

	onMount(() => {
		timeline.eventCallback('onUpdate', () => {
			if (timeline.progress() === 1) isAnimationActive = false;
		});
		setTimeout(() => {
			isAnimationActive = true;
			draw();
		}, 200);
	});

	onDestroy(() => {
		timeline?.kill();
	});
</script>

<WeightPopoverCard
	id="rwkv-state"
	title="RWKV State Calculation"
	className="rwkv-state-popover"
	bind:isAnimationActive
	{timeline}
>
	<div class="rwkv-state-popover-content">
		<div class="token-row">
			<div class="token-context">Layer {$blockIdx + 1} · State Head {$attentionHeadIdx + 1}</div>
			<div class="token-buttons">
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
					>
						<span>{index + 1}</span>{token}
					</button>
				{/each}
			</div>
		</div>
		<div class="view-tabs" role="group" aria-label="RWKV state view">
			<button
				type="button"
				class:active={$rwkvStateViewMode === 'matrix'}
				on:click={() => rwkvStateViewMode.set('matrix')}
				>Matrix</button
			>
			<button
				type="button"
				class:active={$rwkvStateViewMode === 'manifold'}
				on:click={() => rwkvStateViewMode.set('manifold')}
				>Manifold</button
			>
		</div>
		<div class="equation-row">
			<Katex
				math={'S_t = S_{t-1}\\operatorname{diag}(w_t) - (S_{t-1}\\hat{\\kappa}_t)(a_t\\odot\\hat{\\kappa}_t)^\\top + v_t\\tilde{k}_t^\\top'}
			/>
		</div>
		{#if $rwkvStateViewMode === 'matrix'}
			<div class="calc-row">
				{#each terms as term}
					<div
						class="calc-card {term.key}"
						style={`--signal:${term.signal}`}
						title={`${term.label}: ${formatValue(term.value)} · ${term.size}`}
					>
						<div class="calc-title">{term.label}</div>
						<Matrix
							className="rwkv-calc-matrix"
							data={term.data}
							showSize={false}
							cellHeight={18}
							cellWidth={18}
							rowGap={2}
							colGap={2}
							colorScale={colorScale(term.color)}
						/>
						<div class={`energy-strip ${term.color}`}><span></span></div>
						<div class="calc-size">{term.size}</div>
					</div>
					{#if term.operator}
						<div class="operator">{term.operator}</div>
					{/if}
				{/each}
			</div>
			{:else}
				<div class="manifold-view">
					<div class="manifold-mode-tabs" role="group" aria-label="RWKV state projection detail">
						<button
							type="button"
							class:active={manifoldDetailMode === 'plane'}
							on:click={() => {
								manifoldDetailMode = 'plane';
							}}>2D Path</button
						>
						<button
							type="button"
							class:active={manifoldDetailMode === 'multi'}
							on:click={() => {
								manifoldDetailMode = 'multi';
							}}>Multi-Dim</button
						>
					</div>
					{#if manifoldDetailMode === 'plane'}
					<svg
						class="manifold-svg"
						viewBox={`0 0 ${manifoldWidth} ${manifoldHeight}`}
					role="img"
					aria-label="RWKV state matrix trajectory projection"
					>
						<defs>
							<linearGradient id="rwkv-state-plane-gradient" x1="0%" x2="100%" y1="100%" y2="0%">
								<stop offset="0%" stop-color={colorFor('sky', 50)} />
								<stop offset="52%" stop-color={colorFor('gray', 50)} />
								<stop offset="100%" stop-color={colorFor('purple', 50)} />
							</linearGradient>
							<marker
								id="rwkv-state-arrow"
								markerWidth="8"
								markerHeight="8"
								refX="7"
								refY="4"
								orient="auto"
								markerUnits="strokeWidth"
							>
								<path d="M 0 0 L 8 4 L 0 8 z" />
							</marker>
						</defs>
						<rect
							x={manifoldPadding}
							y={manifoldPadding}
							width={manifoldWidth - manifoldPadding * 2}
							height={manifoldHeight - manifoldPadding * 2}
							rx="8"
							class="manifold-plane"
						/>
						{#each manifoldGrid as line}
							<line
								class="manifold-grid"
								x1={line.x}
								x2={line.x}
								y1={manifoldPadding}
								y2={manifoldHeight - manifoldPadding}
							/>
							<line
								class="manifold-grid"
								x1={manifoldPadding}
								x2={manifoldWidth - manifoldPadding}
								y1={line.y}
								y2={line.y}
							/>
						{/each}
						<line
							class="manifold-axis"
							x1={manifoldPadding}
							x2={manifoldWidth - manifoldPadding}
							y1={manifoldHeight - manifoldPadding}
							y2={manifoldHeight - manifoldPadding}
						/>
						<line
							class="manifold-axis"
							x1={manifoldPadding}
							x2={manifoldPadding}
							y1={manifoldPadding}
							y2={manifoldHeight - manifoldPadding}
						/>
						<text class="axis-label" x={manifoldWidth / 2} y={manifoldHeight - 7}>
							state column center
						</text>
						<text
							class="axis-label"
							transform={`translate(10 ${manifoldHeight / 2}) rotate(-90)`}
						>
							state row center
						</text>
						<path class="trajectory" d={stateTrajectoryPath} />
						{#if activeTrajectorySegment}
							<path
								class="trajectory-active"
								d={activeTrajectorySegment.d}
								marker-end="url(#rwkv-state-arrow)"
							/>
						{/if}
						{#if selectedStatePoint}
							<circle
								class="state-target-ring"
								cx={selectedStatePoint.x}
								cy={selectedStatePoint.y}
								r="16"
							/>
						{/if}
						{#each selectedInfluenceSegments as segment}
							<path
								class={`influence-line ${segment.color}`}
								class:subtractive={segment.key === 'erase'}
								d={segment.d}
								marker-end="url(#rwkv-state-arrow)"
							/>
						{/each}
						{#each stateTrajectory as point, index}
							<g class="token-point" class:active={activeTokenIndex === index}>
								<title>Token {index + 1}: {point.token}</title>
							<circle
								cx={point.x}
								cy={point.y}
								r={point.radius}
								fill={activeTokenIndex === index ? colorFor('purple') : colorFor('gray', 300)}
								stroke={activeTokenIndex === index ? colorFor('gray', 900) : 'white'}
							/>
							<text x={point.x} y={point.y + 3}>{index + 1}</text>
						</g>
					{/each}
						{#each selectedTermPoints as point}
							<g class="term-point">
								<title>{point.label}</title>
								<circle
									cx={point.x}
									cy={point.y}
									r={point.empty ? 4 : 7}
									fill={colorFor(point.color)}
								/>
								<text x={point.x + 8} y={point.y - 7}>{point.label}</text>
							</g>
						{/each}
					</svg>
				<div class="manifold-legend">
					{#each terms.filter((term) => term.key !== 'decay') as term}
						<span class={`legend-chip ${term.color}`} title={`${term.label}: ${formatValue(term.value)}`}>
							<span></span>{term.label}
							</span>
						{/each}
					</div>
					{:else}
						<svg
							class="manifold-svg multi-dim-svg"
							viewBox={`0 0 ${manifoldWidth} ${manifoldHeight}`}
							role="img"
							aria-label="RWKV state matrix multi-dimensional projection"
						>
							<rect
								x={manifoldPadding}
								y={manifoldPadding}
								width={manifoldWidth - manifoldPadding * 2}
								height={manifoldHeight - manifoldPadding * 2}
								rx="8"
								class="manifold-plane"
							/>
							{#each multiDimAxes as axis, axisIndex}
								<line
									class="multi-axis"
									x1={multiAxisX(axisIndex)}
									x2={multiAxisX(axisIndex)}
									y1={manifoldPadding}
									y2={manifoldHeight - manifoldPadding}
								/>
								<text
									class="multi-axis-label"
									x={multiAxisX(axisIndex)}
									y={manifoldHeight - 8}
								>
									{axis.label}
								</text>
							{/each}
							{#each multiDimStatePoints as point}
								<path
									class="multi-state-path"
									class:active={activeTokenIndex === point.index}
									d={point.path}
								/>
							{/each}
							{#each multiDimStatePoints as point}
								{#if activeTokenIndex === point.index}
									{#each multiDimAxes as axis, axisIndex}
										<circle
											class="multi-state-value"
											cx={multiAxisX(axisIndex)}
											cy={multiAxisY(point.values[axis.key])}
											r="5"
										/>
									{/each}
								{/if}
							{/each}
						</svg>
						<div class="manifold-legend">
							{#each multiDimStatePoints as point}
								<span
									class="legend-chip state-token"
									class:active={activeTokenIndex === point.index}
									title={`Token ${point.index + 1}: ${point.token}`}
								>
									<span></span>{point.index + 1}
								</span>
							{/each}
						</div>
					{/if}
				</div>
			{/if}
			{#if $rwkvStateViewMode === 'matrix'}
				<div class="readout-row">
					<Katex math={'y_t = r_t S_t + ((r_t\\odot\\tilde{k}_t\\odot r_k)\\mathbf{1})v_t'} />
					<div class="readout-values">
						{#each readoutTerms as term}
							<div
								class={`readout-chip ${term.color}`}
								style={`--signal:${term.signal}`}
								title={`${term.label}: ${formatValue(term.value)}`}
							>
								<span>{term.label}</span>
								<div class="readout-bar"><span></span></div>
							</div>
						{/each}
					</div>
				</div>
			{/if}
		</div>
</WeightPopoverCard>

<style lang="scss">
	:global(.rwkv-state-popover) {
		:global(.content) {
			overflow: visible;
		}
	}

	.rwkv-state-popover-content {
		padding: 1.2rem 1.4rem 1rem 1.4rem;
		display: grid;
		gap: 0.8rem;
		color: theme('colors.gray.600');
	}

	.token-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
	}

	.token-context {
		color: theme('colors.gray.700');
		font-weight: 600;
		white-space: nowrap;
	}

	.view-tabs {
		display: inline-flex;
		justify-self: center;
		border: 1px solid theme('colors.gray.200');
		background: theme('colors.gray.50');

		button {
			border: 0;
			background: transparent;
			color: theme('colors.gray.500');
			font-size: 0.72rem;
			font-weight: 600;
			padding: 0.22rem 0.65rem;
			cursor: pointer;

			&.active {
				background: white;
				color: theme('colors.purple.700');
				box-shadow: 0 1px 2px rgb(0 0 0 / 0.08);
			}
		}
	}

	.token-buttons {
		display: flex;
		gap: 0.25rem;

		button {
			max-width: 5rem;
			display: inline-flex;
			gap: 0.2rem;
			align-items: center;
			border: 1px solid theme('colors.gray.200');
			background: white;
			color: theme('colors.gray.500');
			font-size: 0.72rem;
			padding: 0.2rem 0.3rem;
			cursor: pointer;
			overflow: hidden;
			text-overflow: ellipsis;
			white-space: nowrap;

			span {
				font-weight: 700;
				color: theme('colors.gray.400');
			}

			&.active {
				border-color: theme('colors.purple.300');
				background: theme('colors.purple.50');
				color: theme('colors.purple.700');
			}
		}
	}

	.equation-row,
	.readout-row {
		display: flex;
		justify-content: center;
		align-items: center;
		gap: 1rem;
		color: theme('colors.gray.700');
	}

	.calc-row {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 0.55rem;
	}

	.calc-card {
		display: grid;
		justify-items: center;
		gap: 0.24rem;
		min-width: 5.4rem;
	}

	.calc-title {
		font-size: 0.82rem;
		font-weight: 700;
		color: theme('colors.gray.700');
	}

	.calc-size {
		font-size: 0.68rem;
		color: theme('colors.gray.400');
	}

	.energy-strip {
		width: 3.1rem;
		height: 0.32rem;
		border: 1px solid theme('colors.gray.200');
		background: theme('colors.gray.100');
		overflow: hidden;

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

	.operator {
		margin-top: -1.6rem;
		color: theme('colors.gray.600');
		font-size: 1.35rem;
		font-weight: 700;
	}

	.manifold-view {
		display: grid;
		justify-items: center;
		gap: 0.45rem;
	}

	.manifold-mode-tabs {
		display: inline-flex;
		border: 1px solid theme('colors.gray.200');
		background: theme('colors.gray.50');

		button {
			border: 0;
			background: transparent;
			color: theme('colors.gray.500');
			font-size: 0.68rem;
			font-weight: 700;
			line-height: 1;
			padding: 0.22rem 0.55rem;
			cursor: pointer;

			&.active {
				background: white;
				color: theme('colors.purple.700');
				box-shadow: 0 1px 2px rgb(0 0 0 / 0.08);
			}
		}
	}

	.manifold-svg {
		width: 32rem;
		max-width: 100%;
		height: auto;
		overflow: visible;
	}

	.manifold-plane {
		fill: url('#rwkv-state-plane-gradient');
		stroke: theme('colors.gray.300');
		stroke-width: 1;
	}

	.manifold-grid {
		stroke: white;
		stroke-width: 1;
		opacity: 0.8;
	}

	.manifold-axis {
		stroke: theme('colors.gray.400');
		stroke-width: 1.2;
	}

	.axis-label {
		fill: theme('colors.gray.500');
		font-size: 0.48rem;
		font-weight: 700;
		letter-spacing: 0;
		text-anchor: middle;
	}

	.trajectory {
		fill: none;
		stroke: theme('colors.gray.400');
		stroke-width: 2.2;
		stroke-linecap: round;
		stroke-linejoin: round;
		opacity: 0.75;
	}

	.trajectory-active {
		fill: none;
		stroke: theme('colors.purple.700');
		stroke-width: 3.2;
		stroke-linecap: round;
		stroke-linejoin: round;
	}

	.state-target-ring {
		fill: theme('colors.purple.100');
		stroke: theme('colors.purple.400');
		stroke-width: 2;
		opacity: 0.55;
	}

	.influence-line {
		fill: none;
		stroke-width: 1.8;
		stroke-linecap: round;
		stroke-linejoin: round;
		opacity: 0.7;

		&.sky {
			stroke: theme('colors.sky.500');
		}
		&.rose {
			stroke: theme('colors.rose.500');
		}
		&.red {
			stroke: theme('colors.red.500');
		}
		&.emerald {
			stroke: theme('colors.emerald.500');
		}
		&.purple {
			stroke: theme('colors.purple.500');
		}
		&.subtractive {
			stroke-dasharray: 4 3;
		}
	}

	:global(#rwkv-state-arrow path) {
		fill: currentColor;
	}

	.multi-axis {
		stroke: theme('colors.gray.300');
		stroke-width: 1.2;
	}

	.multi-axis-label {
		fill: theme('colors.gray.600');
		font-size: 0.55rem;
		font-weight: 700;
		letter-spacing: 0;
		text-anchor: middle;
	}

	.multi-state-path {
		fill: none;
		stroke: theme('colors.gray.300');
		stroke-width: 1.4;
		stroke-linecap: round;
		stroke-linejoin: round;
		opacity: 0.45;

		&.active {
			stroke: theme('colors.purple.700');
			stroke-width: 3;
			opacity: 1;
		}
	}

	.multi-state-value {
		fill: theme('colors.purple.600');
		stroke: white;
		stroke-width: 2;
	}

	.token-point {
		text {
			text-anchor: middle;
			font-size: 0.55rem;
			font-weight: 700;
			fill: white;
			pointer-events: none;
		}

		&:not(.active) text {
			fill: theme('colors.gray.700');
		}
	}

	.term-point {
		circle {
			stroke: white;
			stroke-width: 2;
			opacity: 0.9;
		}

		text {
			fill: theme('colors.gray.700');
			font-size: 0.54rem;
			font-weight: 700;
			paint-order: stroke;
			stroke: white;
			stroke-width: 3px;
			stroke-linejoin: round;
			pointer-events: none;
		}
	}

	.manifold-legend {
		display: flex;
		flex-wrap: wrap;
		justify-content: center;
		gap: 0.35rem;
	}

	.legend-chip {
		display: inline-flex;
		align-items: center;
		gap: 0.24rem;
		font-size: 0.72rem;
		font-weight: 600;
		color: theme('colors.gray.600');

		span {
			width: 0.55rem;
			height: 0.55rem;
			background: currentColor;
			border: 1px solid rgb(255 255 255 / 0.8);
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
		&.state-token {
			color: theme('colors.gray.500');

			span {
				background: theme('colors.gray.300');
			}

			&.active {
				color: theme('colors.purple.700');

				span {
					background: theme('colors.purple.600');
				}
			}
		}
	}

	.readout-values {
		display: flex;
		gap: 0.4rem;
	}

	.readout-chip {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		border: 1px solid theme('colors.gray.200');
		padding: 0.18rem 0.35rem;
		background: white;

		span {
			color: theme('colors.gray.500');
		}

		&.violet {
			border-color: theme('colors.violet.200');
			background: theme('colors.violet.50');
		}
		&.amber {
			border-color: theme('colors.amber.200');
			background: theme('colors.amber.50');
		}
		&.purple {
			border-color: theme('colors.purple.200');
			background: theme('colors.purple.50');
		}
	}

	.readout-bar {
		width: 2.5rem;
		height: 0.28rem;
		border: 1px solid theme('colors.gray.200');
		background: white;
		overflow: hidden;

		span {
			display: block;
			width: calc(var(--signal) * 100%);
			height: 100%;
			background: theme('colors.purple.500');
		}
	}

	.readout-chip.violet .readout-bar span {
		background: theme('colors.violet.500');
	}
	.readout-chip.amber .readout-bar span {
		background: theme('colors.amber.500');
	}
	.readout-chip.purple .readout-bar span {
		background: theme('colors.purple.500');
	}
</style>
