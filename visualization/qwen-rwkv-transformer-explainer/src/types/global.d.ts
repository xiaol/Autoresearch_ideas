type Flow = {
	type?: 'path' | 'node';
	id: string;
	from?: string;
	to?: string;
	match?: boolean;
	arrow?: boolean;
	anchor?: { from?: 'l' | 'r' | 't' | 'b'; to?: 'l' | 'r' | 't' | 'b' };
	stroke?: { width?: number };
	transition?: gsap.TweenVars & { position?: gsap.Position };
	curve?: 'vh' | 'hv';
};

type MatrixData = number[][];

type TraceModelKey = 'qwen' | 'rwkv';

type ModelMetaData = {
	layer_num: number;
	attention_head_num: number;
	dimension: number;
	vocab_size?: number;
	chunkTotal?: number;
};

type HighlightedToken = {
	index: number | string | null;
	value?: string | null;
	fix?: boolean;
};

type HighlightedHead = {
	index: number | string | null;
	value?: string | null;
	fix?: boolean;
};

type ExpandedBlock = { id: number | string | null };

type Probability = {
	rank: number;
	tokenId: number;
	token: string;
	logit: number;
	scaledLogit: number;
	expLogit: number;
	probability: number;
	topKLogit?: number;
	topPProbability?: number;
	cumulativeProbability?: number;
	cutoffIndex?: number;
};
type Probabilities = Probability[];

type RwkvHeadSeries = number[][];
type RwkvStateMatrixSeries = number[][][][];
type RwkvBlockTrace = {
	layer: number;
	timeMix: {
		receptanceNorms: RwkvHeadSeries;
		decayMeans: RwkvHeadSeries;
		keyNorms: RwkvHeadSeries;
		replacementKeyNorms: RwkvHeadSeries;
		removalKeyNorms: RwkvHeadSeries;
		valueNorms: RwkvHeadSeries;
		valueResidualGateMeans: RwkvHeadSeries;
		writeGateMeans: RwkvHeadSeries;
		gateNorms: RwkvHeadSeries;
		stateBeforeNorms: RwkvHeadSeries;
		decayedStateNorms: RwkvHeadSeries;
		writeNorms: RwkvHeadSeries;
		eraseNorms: RwkvHeadSeries;
		stateAfterNorms: RwkvHeadSeries;
		readNorms: RwkvHeadSeries;
		bonusNorms: RwkvHeadSeries;
		outputNorms: RwkvHeadSeries;
	};
	state: {
		hiddenNorms: number[];
		norms: RwkvHeadSeries;
		deltas: RwkvHeadSeries;
		rms: RwkvHeadSeries;
		stableRanks: RwkvHeadSeries;
	};
	stateTransition?: {
		stateBefore: RwkvStateMatrixSeries;
		decayed: RwkvStateMatrixSeries;
		erase: RwkvStateMatrixSeries;
		write: RwkvStateMatrixSeries;
		stateAfter: RwkvStateMatrixSeries;
	};
	channelMix: {
		previousStateNorms: number[];
		keyActivationNorms: number[];
		outputNorms: number[];
	};
};

type ModelData = {
	logits: number[];
	outputs: Record<string, { data: number[][]; dims: number[]; size: number }>;
	probabilities: Probabilities;
	sampled: Probability;
	rwkvState?: unknown;
	rwkvBlocks?: RwkvBlockTrace[];
};

type Sampling = { type: 'top-k' | 'top-p'; value: number };
