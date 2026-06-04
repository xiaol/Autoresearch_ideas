import { json, type RequestHandler } from '@sveltejs/kit';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join } from 'node:path';

const defaultModelPath =
	process.env.RWKV_MODEL_PATH ||
	'/home/xiaol/X/models/rwkv7-g1/rwkv7a-g1d-0.1b-20260212-ctx8192.pth';
const defaultManifoldSrc =
	process.env.RWKV_MANIFOLD_SRC || '/home/xiaol/X/rwkv-manifold-steering/src';
const defaultTimeoutMs = Number(process.env.RWKV_TRACE_TIMEOUT_MS || 900_000);

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
	const script = join(root, 'scripts', 'rwkv_trace.py');
	const venvPython = join(root, '.venv', 'bin', 'python');
	const command = process.env.RWKV_TRACE_PYTHON || (existsSync(venvPython) ? venvPython : 'uv');
	const scriptArgs = [
		script,
		'--input',
		body.input || ' ',
		'--model-path',
		body.modelId || defaultModelPath,
		'--temperature',
		String(body.temperature ?? 0.8),
		'--sampling-type',
		body.samplingType || 'top-k',
		'--sampling-value',
		String(body.samplingValue ?? 5),
		'--selection-strategy',
		body.selectionStrategy || process.env.RWKV_SELECTION_STRATEGY || 'greedy',
		'--top-n',
		String(body.topN ?? 50)
	];
	const args = command === 'uv' ? ['run', '--with', 'torch', 'python', ...scriptArgs] : scriptArgs;

	return new Promise<Record<string, unknown>>((resolve, reject) => {
		const child = spawn(command, args, {
			cwd: root,
			env: {
				...process.env,
				PYTHONPATH: [defaultManifoldSrc, process.env.PYTHONPATH].filter(Boolean).join(':'),
				PYTHONUNBUFFERED: '1'
			}
		});

		let stdout = '';
		let stderr = '';
		const timer = setTimeout(() => {
			child.kill('SIGTERM');
			reject(new Error(`RWKV trace timed out after ${defaultTimeoutMs}ms.`));
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
				reject(new Error(stderr || `rwkv_trace.py exited with code ${code}.`));
				return;
			}

			try {
				resolve(JSON.parse(stdout));
			} catch (error) {
				reject(new Error(`Failed to parse rwkv_trace.py output: ${error}\n${stderr}`));
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
