import { InferenceEngine, PrismaClient, Visibility } from '@prisma/client';

const prisma = new PrismaClient();

const DEFAULT_CREATOR_USER_ID = 'clkht01d40000jv08hvalcvly';

function numberFromEnv(name: string, fallback: number): number {
  const value = process.env[name];
  if (!value) {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${name} must be a number`);
  }
  return parsed;
}

function engineFromEnv(): InferenceEngine {
  const value = process.env.RWKV_JLENS_ENGINE || InferenceEngine.RWKV;
  if (value === InferenceEngine.RWKV || value === InferenceEngine.RWKV_MS) {
    return value;
  }
  throw new Error(`RWKV_JLENS_ENGINE must be ${InferenceEngine.RWKV} or ${InferenceEngine.RWKV_MS}`);
}

async function main() {
  const creatorId = process.env.DEFAULT_CREATOR_USER_ID || DEFAULT_CREATOR_USER_ID;
  const modelId = process.env.RWKV_JLENS_MODEL_ID || 'rwkv7-g1d-0-1b';
  const hostUrl = process.env.RWKV_JLENS_HOST_URL || 'http://127.0.0.1:5003';
  const engine = engineFromEnv();

  await prisma.user.upsert({
    where: { id: creatorId },
    update: {},
    create: {
      id: creatorId,
      name: process.env.RWKV_JLENS_CREATOR_NAME || 'bot',
      bot: true,
      admin: true,
      emailNewsletterNotification: false,
      emailUnsubscribeAll: true,
    },
  });

  const model = await prisma.model.upsert({
    where: { id: modelId },
    update: {
      displayNameShort: process.env.RWKV_JLENS_DISPLAY_SHORT || 'RWKV7 0.1B',
      displayName: process.env.RWKV_JLENS_DISPLAY_NAME || 'RWKV7 G1D 0.1B',
      dimension: numberFromEnv('RWKV_JLENS_DIMENSION', 768),
      inferenceEnabled: true,
      instruct: true,
      layers: numberFromEnv('RWKV_JLENS_LAYERS', 12),
      neuronsPerLayer: numberFromEnv('RWKV_JLENS_NEURONS_PER_LAYER', 0),
      owner: process.env.RWKV_JLENS_OWNER || 'RWKV',
      visibility: Visibility.UNLISTED,
      website: process.env.RWKV_JLENS_WEBSITE || 'https://huggingface.co/BlinkDL/rwkv7-g1',
    },
    create: {
      id: modelId,
      displayNameShort: process.env.RWKV_JLENS_DISPLAY_SHORT || 'RWKV7 0.1B',
      displayName: process.env.RWKV_JLENS_DISPLAY_NAME || 'RWKV7 G1D 0.1B',
      creatorId,
      dimension: numberFromEnv('RWKV_JLENS_DIMENSION', 768),
      inferenceEnabled: true,
      instruct: true,
      layers: numberFromEnv('RWKV_JLENS_LAYERS', 12),
      neuronsPerLayer: numberFromEnv('RWKV_JLENS_NEURONS_PER_LAYER', 0),
      owner: process.env.RWKV_JLENS_OWNER || 'RWKV',
      visibility: Visibility.UNLISTED,
      website: process.env.RWKV_JLENS_WEBSITE || 'https://huggingface.co/BlinkDL/rwkv7-g1',
    },
  });

  const existingHost = await prisma.inferenceHostSource.findFirst({
    where: {
      modelId,
      engine,
      hostUrl,
    },
  });

  const host = existingHost
    ? await prisma.inferenceHostSource.update({
        where: { id: existingHost.id },
        data: {
          name: process.env.RWKV_JLENS_HOST_NAME || 'Local RWKV JLens adapter',
          hostUrl,
          engine,
        },
      })
    : await prisma.inferenceHostSource.create({
        data: {
          name: process.env.RWKV_JLENS_HOST_NAME || 'Local RWKV JLens adapter',
          hostUrl,
          engine,
          modelId,
        },
      });

  console.log({ model, host });
}

main()
  .then(async () => {
    await prisma.$disconnect();
  })
  .catch(async (error) => {
    console.error(error);
    await prisma.$disconnect();
    process.exit(1);
  });
