import { prisma } from '@/lib/db';
import { DEFAULT_JLENS_MODEL_ID } from '@/lib/utils/lens';
import { getAllLensHostsForModel, JLENS_INFERENCE_ENGINES } from './inference-host-source';

const LOCAL_RWKV_JLENS_MODEL_ID = 'rwkv7-g1d-0-1b';
const PREFERRED_JLENS_MODEL_IDS = [DEFAULT_JLENS_MODEL_ID, LOCAL_RWKV_JLENS_MODEL_ID];

const getExistingModelId = async (modelId: string) => {
  const model = await prisma.model.findUnique({
    where: { id: modelId },
    select: { id: true },
  });
  return model?.id ?? null;
};

const getHostedModelId = async (modelId: string) => {
  const existingModelId = await getExistingModelId(modelId);
  if (!existingModelId) {
    return null;
  }

  const { hosts } = await getAllLensHostsForModel(modelId);
  return hosts.length > 0 ? existingModelId : null;
};

const getFirstHostedJlensModelId = async () => {
  for (const engine of JLENS_INFERENCE_ENGINES) {
    // Engine order is the same priority list used by live lens inference.
    // eslint-disable-next-line no-await-in-loop
    const host = await prisma.inferenceHostSource.findFirst({
      where: { engine },
      select: { modelId: true },
      orderBy: { modelId: 'asc' },
    });
    if (host) {
      return host.modelId;
    }
  }

  return null;
};

export const getDefaultJlensModelId = async () => {
  for (const modelId of PREFERRED_JLENS_MODEL_IDS) {
    // eslint-disable-next-line no-await-in-loop
    const hostedModelId = await getHostedModelId(modelId);
    if (hostedModelId) {
      return hostedModelId;
    }
  }

  const hostedModelId = await getFirstHostedJlensModelId();
  if (hostedModelId) {
    return hostedModelId;
  }

  for (const modelId of PREFERRED_JLENS_MODEL_IDS) {
    // eslint-disable-next-line no-await-in-loop
    const existingModelId = await getExistingModelId(modelId);
    if (existingModelId) {
      return existingModelId;
    }
  }

  const inferenceEnabledModel = await prisma.model.findFirst({
    where: { inferenceEnabled: true },
    select: { id: true },
    orderBy: { id: 'asc' },
  });
  if (inferenceEnabledModel) {
    return inferenceEnabledModel.id;
  }

  const model = await prisma.model.findFirst({
    select: { id: true },
    orderBy: { id: 'asc' },
  });
  return model?.id ?? DEFAULT_JLENS_MODEL_ID;
};

export const getDefaultJlensHref = async () => `/${await getDefaultJlensModelId()}/jlens`;
