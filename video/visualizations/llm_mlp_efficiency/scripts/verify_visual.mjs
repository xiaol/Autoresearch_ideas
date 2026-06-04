import { mkdir } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";

const baseUrl = process.env.BASE_URL ?? "http://localhost:5173/";
const executablePath = process.env.CHROME_EXECUTABLE ?? "/usr/bin/google-chrome";
const screenshotDir = path.resolve("qa-screenshots");

const viewports = [
  { name: "desktop", width: 1440, height: 920 },
  { name: "mobile", width: 390, height: 844 },
];

await mkdir(screenshotDir, { recursive: true });

const browser = await chromium.launch({
  executablePath,
  headless: true,
  args: ["--disable-dev-shm-usage", "--no-sandbox"],
});

try {
  for (const viewport of viewports) {
    await verifyViewport(browser, viewport);
  }
  console.log(`Visual QA passed for ${viewports.map((viewport) => viewport.name).join(", ")}.`);
} finally {
  await browser.close();
}

async function verifyViewport(browser, viewport) {
  const page = await browser.newPage({ viewport });
  const browserErrors = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(`${message.type()}: ${message.text()}`);
    }
  });

  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.waitForSelector("#networkCanvas");
  await page.waitForFunction(() => {
    const canvas = document.querySelector("#networkCanvas");
    return canvas && canvas.width > 64 && canvas.height > 64;
  });
  await page.waitForTimeout(900);

  const firstLabel = await page.locator("#stepLabel").textContent();
  await page.locator("#nextStepButton").click();
  await page.waitForTimeout(120);
  const secondLabel = await page.locator("#stepLabel").textContent();
  if (firstLabel === secondLabel) {
    throw new Error(`${viewport.name}: step button did not change the active step.`);
  }
  await page.locator("#viewBlockButton").click();
  await page.waitForTimeout(160);
  const blockPressed = await page.locator("#viewBlockButton").getAttribute("aria-pressed");
  if (blockPressed !== "true") {
    throw new Error(`${viewport.name}: block-flow mode did not activate.`);
  }
  await page.locator("#viewMlpButton").click();
  await page.waitForTimeout(220);
  const mlpPressed = await page.locator("#viewMlpButton").getAttribute("aria-pressed");
  if (mlpPressed !== "true") {
    throw new Error(`${viewport.name}: MLP-only mode did not activate.`);
  }
  const tokenSlider = page.locator("#inputTokenSlider");
  const tokenSliderMax = Number(await tokenSlider.getAttribute("max"));
  if (Number.isFinite(tokenSliderMax) && tokenSliderMax > 1) {
    await tokenSlider.fill("1");
    await page.waitForTimeout(120);
    const tokenValue = await page.locator("#inputTokenValue").textContent();
    if (!tokenValue?.startsWith("1:")) {
      throw new Error(`${viewport.name}: input-token control did not update (${tokenValue}).`);
    }
  }
  await page.locator("#channelSampleSelect").selectOption("spread");
  await page.locator("#outputReadoutSelect").selectOption("tokens");
  await page.waitForTimeout(160);
  const outputTitle = await page.locator("#outputPanelTitle").textContent();
  if (outputTitle !== "Next Token") {
    throw new Error(`${viewport.name}: output readout control did not update (${outputTitle}).`);
  }
  await page.locator("#channelSampleSelect").selectOption("strong");
  await page.locator("#outputReadoutSelect").selectOption("channels");
  await page.waitForTimeout(160);

  await page.locator("#modelSelect").selectOption("rwkv7-01b");
  await page.waitForFunction(() => document.querySelector("#modelName")?.textContent?.includes("RWKV"));
  await page.waitForTimeout(260);
  const rwkvModelName = await page.locator("#modelName").textContent();
  const rwkvLayerCount = await page.locator("#layerCount").textContent();
  const rwkvModeLabel = await page.locator("#viewMlpButton").textContent();
  const rwkvOutputTitle = await page.locator("#outputPanelTitle").textContent();
  if (!rwkvModelName?.includes("RWKV") || rwkvLayerCount !== "12" || rwkvModeLabel !== "RWKV Mix") {
    throw new Error(
      `${viewport.name}: RWKV trace did not load expected metadata (${rwkvModelName}, ${rwkvLayerCount}, ${rwkvModeLabel}).`,
    );
  }
  if (rwkvOutputTitle !== "Output State") {
    throw new Error(`${viewport.name}: RWKV output readout did not use state wording (${rwkvOutputTitle}).`);
  }
  await page.screenshot({
    path: path.join(screenshotDir, `${viewport.name}-rwkv.png`),
    fullPage: true,
  });
  await page.locator("#modelSelect").selectOption("qwen35-08b");
  await page.waitForFunction(() => document.querySelector("#modelName")?.textContent?.includes("Qwen"));
  await page.waitForTimeout(220);
  await page.locator("#viewCompareButton").click();
  await page.waitForFunction(() => document.querySelector("#modelName")?.textContent?.includes("vs"));
  await page.waitForTimeout(420);
  const comparePressed = await page.locator("#viewCompareButton").getAttribute("aria-pressed");
  const compareModelName = await page.locator("#modelName").textContent();
  const compareOutputTitle = await page.locator("#outputPanelTitle").textContent();
  if (comparePressed !== "true" || compareModelName !== "Qwen3.5 vs RWKV-7" || compareOutputTitle !== "Compare Outputs") {
    throw new Error(
      `${viewport.name}: comparison view did not activate (${comparePressed}, ${compareModelName}, ${compareOutputTitle}).`,
    );
  }
  await page.screenshot({
    path: path.join(screenshotDir, `${viewport.name}-compare.png`),
    fullPage: true,
  });
  await page.locator("#viewMlpButton").click();
  await page.waitForTimeout(220);

  const canvasStats = await page.evaluate(async () => {
    const canvas = document.querySelector("#networkCanvas");
      const sample = () => {
        const probe = document.createElement("canvas");
        probe.width = 96;
        probe.height = 96;
        const context = probe.getContext("2d", { willReadFrequently: true });
        context.drawImage(canvas, 0, 0, probe.width, probe.height);
        const data = context.getImageData(0, 0, probe.width, probe.height).data;
        let bright = 0;
        let nonBackground = 0;
        const colors = new Set();
        const fingerprint = [];
        for (let index = 0; index < data.length; index += 16) {
          const red = data[index];
          const green = data[index + 1];
          const blue = data[index + 2];
          if (red + green + blue > 96) bright += 1;
        if (Math.abs(red - 16) + Math.abs(green - 17) + Math.abs(blue - 15) > 16) {
          nonBackground += 1;
          }
          colors.add(`${red >> 4},${green >> 4},${blue >> 4}`);
          fingerprint.push(red, green, blue);
        }
        return { bright, nonBackground, colors: colors.size, fingerprint };
      };
      const before = sample();
      await new Promise((resolve) => setTimeout(resolve, 650));
      const after = sample();
      let pixelDelta = 0;
      for (let index = 0; index < before.fingerprint.length; index += 1) {
        pixelDelta += Math.abs(before.fingerprint[index] - after.fingerprint[index]);
      }
      return {
        width: canvas.width,
        height: canvas.height,
        before: {
          bright: before.bright,
          nonBackground: before.nonBackground,
          colors: before.colors,
        },
        after: {
          bright: after.bright,
          nonBackground: after.nonBackground,
          colors: after.colors,
        },
        pixelDelta,
      };
    });

  if (canvasStats.before.nonBackground < 120 || canvasStats.before.colors < 8) {
    throw new Error(`${viewport.name}: canvas appears blank (${JSON.stringify(canvasStats.before)}).`);
  }
  if (canvasStats.pixelDelta > 6000) {
    throw new Error(`${viewport.name}: MLP canvas is unexpectedly unstable (${JSON.stringify(canvasStats)}).`);
  }

  const layout = await page.evaluate(() => {
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const selectors = [".left-panel", ".visual-stage", ".right-panel", ".stage-toolbar", ".token-strip"];
    return selectors.map((selector) => {
      const element = document.querySelector(selector);
      const rect = element.getBoundingClientRect();
      return {
        selector,
        width: rect.width,
        height: rect.height,
        top: rect.top,
        left: rect.left,
        inside:
          rect.width > 0 &&
          rect.height > 0 &&
          rect.left < viewportWidth &&
          rect.right > 0 &&
          rect.top < Math.max(viewportHeight, document.documentElement.scrollHeight) &&
          rect.bottom > -1,
      };
    });
  });
  const badLayout = layout.filter((item) => !item.inside);
  if (badLayout.length) {
    throw new Error(`${viewport.name}: layout elements outside viewport: ${JSON.stringify(badLayout)}`);
  }

  await page.screenshot({
    path: path.join(screenshotDir, `${viewport.name}.png`),
    fullPage: true,
  });

  await page.close();
  if (browserErrors.length) {
    throw new Error(`${viewport.name}: browser errors:\n${browserErrors.join("\n")}`);
  }
}
