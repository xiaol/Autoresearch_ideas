import { json, type RequestHandler } from '@sveltejs/kit';
import { spawn } from 'node:child_process';
import { existsSync, readdirSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

const cachedQwen35Path = () => {
	const snapshotsRoot = join(
		homedir(),
		'.cache',
		'huggingface',
		'hub',
		'models--Qwen--Qwen3.5-0.8B-Base',
		'snapshots'
	);
	if (!existsSync(snapshotsRoot)) return null;

	return readdirSync(snapshotsRoot)
		.map((name) => join(snapshotsRoot, name))
		.filter((path) => existsSync(join(path, 'config.json')))
		.sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs)[0];
};

const defaultModelId =
	process.env.QWEN_MODEL_ID || cachedQwen35Path() || 'Qwen/Qwen3.5-0.8B-Base';
const defaultTimeoutMs = Number(process.env.QWEN_TRACE_TIMEOUT_MS || 900_000);

type TraceRequest = {
	input?: string;
	modelId?: string;
	temperature?: number;
	samplingType?: 'top-k' | 'top-p';
	samplingValue?: number;
	selectionStrategy?: 'greedy' | 'sample';
	topN?: number;
};

const runTraceScript = (body: TraceRequest) => {
	const root = process.cwd();
	const script = join(root, 'scripts', 'qwen_trace.py');
	const venvPython = join(root, '.venv', 'bin', 'python');
	const command = process.env.QWEN_TRACE_PYTHON || (existsSync(venvPython) ? venvPython : 'uv');
	const scriptArgs = [
		script,
		'--input',
		body.input || ' ',
		'--model-id',
		body.modelId || defaultModelId,
		'--temperature',
		String(body.temperature ?? 0.8),
		'--sampling-type',
		body.samplingType || 'top-k',
		'--sampling-value',
		String(body.samplingValue ?? 5),
		'--selection-strategy',
		body.selectionStrategy || process.env.QWEN_SELECTION_STRATEGY || 'greedy',
		'--top-n',
		String(body.topN ?? 50)
	];
	const args =
		command === 'uv'
			? [
					'run',
					'--with',
					'torch',
					'--with',
					'transformers>=4.57.0',
					'--with',
					'accelerate',
					...scriptArgs
				]
			: scriptArgs;

	return new Promise<Record<string, unknown>>((resolve, reject) => {
		const child = spawn(command, args, {
			cwd: root,
			env: {
				...process.env,
				PYTHONUNBUFFERED: '1'
			}
		});

		let stdout = '';
		let stderr = '';
		const timer = setTimeout(() => {
			child.kill('SIGTERM');
			reject(new Error(`Qwen trace timed out after ${defaultTimeoutMs}ms.`));
		}, defaultTimeoutMs);

		child.stdout.on('data', (chunk) => {
			stdout += chunk;
		});
		child.stderr.on('data', (chunk) => {
			stderr += chunk;
		});
		child.on('error', (error) => {
			clearTimeout(timer);
			reject(error);
		});
		child.on('close', (code) => {
			clearTimeout(timer);
			if (code !== 0) {
				reject(new Error(stderr || `qwen_trace.py exited with code ${code}.`));
				return;
			}

			try {
				resolve(JSON.parse(stdout));
			} catch (error) {
				reject(new Error(`Failed to parse qwen_trace.py output: ${error}\n${stderr}`));
			}
		});
	});
};

export const POST: RequestHandler = async ({ request }) => {
	try {
		const body = (await request.json()) as TraceRequest;
		const trace = await runTraceScript(body);
		return json(trace);
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		return new Response(message, { status: 500 });
	}
};
