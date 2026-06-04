export const qwenDefaultModelId = 'Qwen/Qwen3.5-0.8B-Base';

export const qwenTraceExamples = [
	'The sun rises in the',
	'Machine learning models learn from training',
	'A language model predicts the next',
	'Attention helps tokens focus on relevant',
	'A small language model can answer simple'
];

export const qwenModelMetaMap: Record<string, ModelMetaData> = {
	'qwen3-0.6b': { layer_num: 28, attention_head_num: 16, dimension: 1024, vocab_size: 151936 },
	'qwen3.5-0.8b': { layer_num: 24, attention_head_num: 8, dimension: 1024, vocab_size: 248320 }
};

export const qwenPlaceholderData: ModelData = {
	logits: [],
	outputs: {},
	probabilities: [
		{
			rank: 0,
			tokenId: 0,
			token: '',
			logit: 0,
			scaledLogit: 0,
			expLogit: 1,
			probability: 1,
			topKLogit: 0
		}
	],
	sampled: {
		rank: 0,
		tokenId: 0,
		token: '',
		logit: 0,
		scaledLogit: 0,
		expLogit: 1,
		probability: 1,
		topKLogit: 0
	}
};
