import { mkdir, rm } from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { chromium } from "playwright";

const defaults = {
  baseUrl: process.env.BASE_URL ?? "http://localhost:5173/",
  executablePath: process.env.CHROME_EXECUTABLE ?? "/usr/bin/google-chrome",
  outputDir: process.env.OUTPUT_DIR ?? "video-captures",
  prompt:
    process.env.PROMPT ??
    "Explain why attention heads, feed-forward layers, and recurrent state can specialize differently in compact language models.",
  scenario: process.env.SCENARIO ?? "all",
  preset: process.env.PRESET ?? "1080p",
  fps: Number(process.env.FPS ?? 15),
  duration: Number(process.env.DURATION ?? 8),
  width: Number(process.env.WIDTH ?? 1920),
  height: Number(process.env.HEIGHT ?? 1080),
  camera: process.env.CAMERA ?? "orbit",
  tokenMode: process.env.TOKEN_MODE ?? "step",
  keepFrames: process.env.KEEP_FRAMES === "1",
};

const args = parseArgs(process.argv.slice(2), defaults);
applyPreset(args, defaults);
const scenarios = selectScenarios(args.scenario);
const frameCount = Math.max(1, Math.round(args.duration * args.fps));

await mkdir(args.outputDir, { recursive: true });

const browser = await chromium.launch({
  executablePath: args.executablePath,
  headless: true,
  args: ["--disable-dev-shm-usage", "--no-sandbox"],
});

try {
  for (const scenario of scenarios) {
    await recordScenario(browser, scenario);
  }
} finally {
  await browser.close();
}

async function recordScenario(browser, scenario) {
  const framesDir = path.join(args.outputDir, "frames", scenario.name);
  const outputPath = path.join(args.outputDir, `${scenario.name}.mp4`);
  await rm(framesDir, { recursive: true, force: true });
  await mkdir(framesDir, { recursive: true });

  const page = await browser.newPage({
    viewport: { width: args.width, height: args.height },
    deviceScaleFactor: 1,
  });
  const browserErrors = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(`${message.type()}: ${message.text()}`);
    }
  });

  try {
    await page.goto(args.baseUrl, { waitUntil: "networkidle" });
    await page.waitForSelector("#networkCanvas");
    await page.waitForFunction(() => {
      const canvas = document.querySelector("#networkCanvas");
      return canvas && canvas.width > 64 && canvas.height > 64;
    });
    await page.waitForTimeout(700);

    await scenario.setup(page);
    await page.waitForTimeout(700);

    const maxStep = await getSliderMax(page, "#stepSlider");
    const maxToken = await getSliderMax(page, "#inputTokenSlider");
    const heldToken = await getRangeValue(page, "#inputTokenSlider");
    for (let frame = 0; frame < frameCount; frame += 1) {
      const t = frameCount > 1 ? frame / (frameCount - 1) : 0;
      const step = Math.round(t * maxStep);
      const token = Math.round(t * maxToken);
      await setRangeValue(page, "#stepSlider", step);
      if (args.tokenMode === "sweep" && maxToken > 0) {
        await setRangeValue(page, "#inputTokenSlider", token);
      } else if (args.tokenMode === "hold") {
        await setRangeValue(page, "#inputTokenSlider", heldToken);
      }
      await applyCameraMotion(page, t);
      await page.waitForTimeout(20);
      await page.screenshot({
        path: path.join(framesDir, `${String(frame + 1).padStart(4, "0")}.png`),
        fullPage: false,
      });
    }

    await encodeMp4(framesDir, outputPath);
    console.log(`Wrote ${outputPath}`);
  } finally {
    await page.close();
    if (!args.keepFrames) {
      await rm(framesDir, { recursive: true, force: true });
    }
  }

  if (browserErrors.length) {
    throw new Error(`${scenario.name}: browser errors:\n${browserErrors.join("\n")}`);
  }
}

function selectScenarios(value) {
  const allScenarios = [
    {
      name: "qwen35-08b",
      setup: async (page) => {
        await page.locator("#modelSelect").selectOption("qwen35-08b");
        await page.waitForFunction(() => document.querySelector("#modelName")?.textContent?.includes("Qwen"));
        await buildPromptTrace(page);
        await page.locator("#viewMlpButton").click();
      },
    },
    {
      name: "rwkv7-01b",
      setup: async (page) => {
        await page.locator("#modelSelect").selectOption("rwkv7-01b");
        await page.waitForFunction(() => document.querySelector("#modelName")?.textContent?.includes("RWKV"));
        await buildPromptTrace(page);
        await page.locator("#viewMlpButton").click();
      },
    },
    {
      name: "comparison",
      setup: async (page) => {
        await page.locator("#modelSelect").selectOption("qwen35-08b");
        await page.waitForFunction(() => document.querySelector("#modelName")?.textContent?.includes("Qwen"));
        await page.locator("#promptInput").fill(args.prompt);
        await page.locator("#viewCompareButton").click();
        await page.waitForFunction(() => document.querySelector("#modelName")?.textContent === "Qwen3.5 vs RWKV-7");
      },
    },
  ];

  if (value === "all") return allScenarios;
  const requested = new Set(value.split(",").map((item) => item.trim()).filter(Boolean));
  const selected = allScenarios.filter((scenario) => requested.has(scenario.name));
  if (!selected.length) {
    throw new Error(`No matching scenarios for "${value}". Use all, qwen35-08b, rwkv7-01b, comparison, or a comma list.`);
  }
  return selected;
}

async function buildPromptTrace(page) {
  await page.locator("#promptInput").fill(args.prompt);
  await page.locator("#applyPromptButton").click();
  await page.waitForFunction(() => document.querySelector("#traceStatus")?.textContent?.includes("Local prompt trace built"));
}

async function getSliderMax(page, selector) {
  const value = await page.locator(selector).getAttribute("max");
  const max = Number(value);
  return Number.isFinite(max) ? Math.max(0, max) : 0;
}

async function getRangeValue(page, selector) {
  const value = await page.locator(selector).inputValue();
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

async function setRangeValue(page, selector, value) {
  await page.evaluate(
    ({ selector, value }) => {
      const input = document.querySelector(selector);
      if (!input) return;
      input.value = String(value);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    },
    { selector, value },
  );
}

async function applyCameraMotion(page, progress) {
  if (args.camera === "static") return;
  await page.evaluate((progress) => {
    window.__LLM_VISUALIZER__?.setCameraOrbit(progress);
  }, progress);
}

async function encodeMp4(framesDir, outputPath) {
  const inputPattern = path.join(framesDir, "%04d.png");
  await new Promise((resolve, reject) => {
    const ffmpeg = spawn("ffmpeg", [
      "-y",
      "-framerate",
      String(args.fps),
      "-i",
      inputPattern,
      "-c:v",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      "-movflags",
      "+faststart",
      outputPath,
    ]);
    let stderr = "";
    ffmpeg.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    ffmpeg.on("error", reject);
    ffmpeg.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`ffmpeg exited with ${code}\n${stderr}`));
      }
    });
  });
}

function parseArgs(rawArgs, fallback) {
  const parsed = { ...fallback };
  for (let index = 0; index < rawArgs.length; index += 1) {
    const arg = rawArgs[index];
    const next = rawArgs[index + 1];
    if (arg === "--base-url") {
      parsed.baseUrl = next;
      index += 1;
    } else if (arg === "--out") {
      parsed.outputDir = next;
      index += 1;
    } else if (arg === "--prompt") {
      parsed.prompt = next;
      index += 1;
    } else if (arg === "--scenario") {
      parsed.scenario = next;
      index += 1;
    } else if (arg === "--preset") {
      parsed.preset = next;
      index += 1;
    } else if (arg === "--fps") {
      parsed.fps = Number(next);
      index += 1;
    } else if (arg === "--duration") {
      parsed.duration = Number(next);
      index += 1;
    } else if (arg === "--width") {
      parsed.width = Number(next);
      index += 1;
    } else if (arg === "--height") {
      parsed.height = Number(next);
      index += 1;
    } else if (arg === "--camera") {
      parsed.camera = next;
      index += 1;
    } else if (arg === "--token-mode") {
      parsed.tokenMode = next;
      index += 1;
    } else if (arg === "--keep-frames") {
      parsed.keepFrames = true;
    } else if (arg === "--help") {
      printHelpAndExit();
    }
  }
  if (!Number.isFinite(parsed.fps) || parsed.fps <= 0) throw new Error("--fps must be a positive number.");
  if (!Number.isFinite(parsed.duration) || parsed.duration <= 0) throw new Error("--duration must be a positive number.");
  if (!Number.isFinite(parsed.width) || parsed.width <= 0) throw new Error("--width must be a positive number.");
  if (!Number.isFinite(parsed.height) || parsed.height <= 0) throw new Error("--height must be a positive number.");
  if (!["orbit", "static"].includes(parsed.camera)) throw new Error("--camera must be orbit or static.");
  if (!["step", "sweep", "hold"].includes(parsed.tokenMode)) throw new Error("--token-mode must be step, sweep, or hold.");
  return parsed;
}

function applyPreset(parsed, fallback) {
  if (parsed.preset === "4k" || parsed.preset === "2160p") {
    if (parsed.width === fallback.width) parsed.width = 3840;
    if (parsed.height === fallback.height) parsed.height = 2160;
    if (parsed.fps === fallback.fps) parsed.fps = 12;
    return;
  }
  if (parsed.preset !== "1080p") {
    throw new Error(`Unknown preset "${parsed.preset}". Use 1080p or 4k.`);
  }
}

function printHelpAndExit() {
  console.log(`Usage: node scripts/record_videos.mjs [options]

Options:
  --preset 1080p|4k
  --scenario all|qwen35-08b|rwkv7-01b|comparison
  --prompt "Prompt used for all recordings"
  --duration 8
  --fps 15
  --width 1920
  --height 1080
  --camera orbit|static
  --token-mode step|sweep|hold
  --out video-captures
  --base-url http://localhost:5173/
  --keep-frames
`);
  process.exit(0);
}
